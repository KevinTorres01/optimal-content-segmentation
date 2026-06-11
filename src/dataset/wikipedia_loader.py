"""Load real Wikipedia articles as a segmentation dataset.

Each Wikipedia article has natural section boundaries (headings authored by
human editors). This module fetches articles via the MediaWiki API, extracts
plain text with section markers, splits it into sentences, and produces the
same on-disk layout used by the synthetic generator (documents/, boundaries/,
metadata.json) so existing experiments work unchanged.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.dataset.schemas import WikipediaDatasetConfig

_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"

_SECTION_HEADER_RE = re.compile(r"^={2,}\s*(.+?)\s*={2,}\s*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")

_SKIP_SECTIONS_ES = {
    "véase también",
    "referencias",
    "enlaces externos",
    "bibliografía",
    "notas",
    "fuentes",
    "obras",
}
_SKIP_SECTIONS_EN = {
    "see also",
    "references",
    "external links",
    "bibliography",
    "notes",
    "sources",
    "further reading",
}


@dataclass
class WikipediaDatasetMetadata:
    """Statistics about a loaded Wikipedia dataset."""

    dataset_name: str
    n_documents: int
    n_skipped: int
    avg_segments_per_doc: float
    avg_sentences_per_segment: float
    config: dict[str, Any]


class WikipediaLoader:
    """Fetches Wikipedia articles and converts them into a segmentation dataset."""

    def __init__(self, config: WikipediaDatasetConfig) -> None:
        self._config = config
        self._skip_sections = (
            _SKIP_SECTIONS_ES if config.language == "es" else _SKIP_SECTIONS_EN
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "optimal-content-segmentation/1.0 "
                    "(academic research; contact via repository)"
                )
            }
        )

    def load(self, output_dir: Path) -> WikipediaDatasetMetadata:
        """Download and process all configured articles.

        Args:
            output_dir: Directory where documents/, boundaries/ and
                metadata.json will be written.

        Returns:
            WikipediaDatasetMetadata with counts of accepted and skipped docs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "documents").mkdir(exist_ok=True)
        (output_dir / "boundaries").mkdir(exist_ok=True)

        accepted = 0
        skipped = 0
        total_segments = 0
        total_sentences = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Fetching {len(self._config.titles)} Wikipedia articles...",
                total=len(self._config.titles),
            )

            for title in self._config.titles:
                progress.update(task, description=f"Fetching: {title}")
                try:
                    sentences, boundaries = self._process_article(title)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ Skipped '{title}': {exc}")
                    skipped += 1
                    progress.advance(task)
                    continue

                if not self._passes_filters(sentences, boundaries):
                    skipped += 1
                    progress.advance(task)
                    continue

                doc_id = f"doc_{accepted + 1:04d}"
                (output_dir / "documents" / f"{doc_id}.txt").write_text(
                    "\n".join(sentences), encoding="utf-8"
                )
                (output_dir / "boundaries" / f"{doc_id}.json").write_text(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "source_title": title,
                            "boundaries": boundaries,
                            "n_segments": len(boundaries),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                accepted += 1
                total_segments += len(boundaries)
                total_sentences += len(sentences)
                progress.advance(task)

                if self._config.request_delay_seconds > 0:
                    time.sleep(self._config.request_delay_seconds)

        metadata = WikipediaDatasetMetadata(
            dataset_name=self._config.dataset_name,
            n_documents=accepted,
            n_skipped=skipped,
            avg_segments_per_doc=(total_segments / accepted) if accepted else 0.0,
            avg_sentences_per_segment=(
                total_sentences / total_segments if total_segments else 0.0
            ),
            config=self._config.model_dump(),
        )

        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset_name": metadata.dataset_name,
                    "n_documents": metadata.n_documents,
                    "n_skipped": metadata.n_skipped,
                    "avg_segments_per_doc": round(metadata.avg_segments_per_doc, 2),
                    "avg_sentences_per_segment": round(
                        metadata.avg_sentences_per_segment, 2
                    ),
                    "config": metadata.config,
                    "source": "wikipedia",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return metadata

    def _process_article(self, title: str) -> tuple[list[str], list[int]]:
        """Download a single article and return its sentences and boundaries."""
        extract = self._fetch_extract(title)
        if not extract:
            raise ValueError("empty extract")
        return self._extract_to_segments(extract)

    def _fetch_extract(self, title: str) -> str:
        """Call the MediaWiki API and return plain text with section markers.

        Retries on HTTP 429 (rate limit) with exponential backoff. Public
        Wikipedia is generous but bursts of parallel-looking requests can
        trip its throttle.
        """
        url = _API_URL_TEMPLATE.format(lang=self._config.language)
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "redirects": 1,
        }
        backoff = 2.0
        for attempt in range(4):
            response = self._session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return ""
            page = next(iter(pages.values()))
            if "missing" in page:
                raise ValueError(f"article '{title}' not found")
            return page.get("extract", "")
        raise RuntimeError(f"rate-limited after retries for '{title}'")

    def _extract_to_segments(self, extract: str) -> tuple[list[str], list[int]]:
        """Split the API extract into sentences and detect section boundaries.

        Article extracts come back as plain text where headings look like
        ``== Section ==`` / ``=== Subsection ===``. We treat only top-level
        sections (``==``) as boundaries — sub-sections stay inside their parent
        segment so the granularity matches the synthetic dataset.

        Each section is truncated to ``max_sentences_per_segment`` sentences and
        the whole article is truncated to ``max_segments_per_doc`` sections so
        the resulting documents have a size comparable to the synthetic dataset.
        """
        sentences: list[str] = []
        boundaries: list[int] = []
        cap_sents = self._config.max_sentences_per_segment
        cap_segs = self._config.max_segments_per_doc

        intro_text, sections = self._split_top_level_sections(extract)

        intro_sentences = self._split_sentences(intro_text)[:cap_sents]
        if intro_sentences:
            boundaries.append(0)
            sentences.extend(intro_sentences)

        for heading, body in sections:
            if len(boundaries) >= cap_segs:
                break
            if heading.strip().lower() in self._skip_sections:
                continue
            section_sentences = self._split_sentences(body)[:cap_sents]
            if not section_sentences:
                continue
            boundaries.append(len(sentences))
            sentences.extend(section_sentences)

        return sentences, boundaries

    def _split_top_level_sections(
        self, extract: str
    ) -> tuple[str, list[tuple[str, str]]]:
        """Return (intro_text, [(heading, body), ...]) using only ``==`` markers."""
        lines = extract.splitlines()
        intro_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current_heading: str | None = None
        current_body: list[str] = []

        for line in lines:
            match = re.match(r"^(=+)\s*(.+?)\s*\1\s*$", line)
            # Only top-level "==" headings open a new segment.
            if match and len(match.group(1)) == 2:
                if current_heading is None:
                    intro_lines = current_body
                else:
                    sections.append((current_heading, current_body))
                current_heading = match.group(2)
                current_body = []
            else:
                # Drop sub-section headings (===, ====, ...) entirely so they
                # don't pollute the sentence list with non-sentence tokens.
                if match:
                    continue
                current_body.append(line)

        if current_heading is None:
            intro_lines = current_body
        else:
            sections.append((current_heading, current_body))

        intro_text = "\n".join(intro_lines)
        section_pairs = [(h, "\n".join(b)) for h, b in sections]
        return intro_text, section_pairs

    def _split_sentences(self, text: str) -> list[str]:
        """Split a block of text into clean sentences."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sentences: list[str] = []
        for paragraph in paragraphs:
            for raw in _SENTENCE_SPLIT_RE.split(paragraph):
                clean = raw.strip()
                if len(clean) >= self._config.min_sentence_chars:
                    sentences.append(clean)
        return sentences

    def _passes_filters(self, sentences: list[str], boundaries: list[int]) -> bool:
        """Drop articles that fall outside the configured size envelope."""
        cfg = self._config
        n_sent = len(sentences)
        n_seg = len(boundaries)
        if n_sent < cfg.min_sentences_per_doc:
            return False
        if n_sent > cfg.max_sentences_per_doc:
            return False
        if n_seg < cfg.min_segments_per_doc:
            return False
        if n_seg > cfg.max_segments_per_doc:
            return False
        return True


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Download Wikipedia articles and build a segmentation dataset"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = WikipediaDatasetConfig.model_validate(raw)

    loader = WikipediaLoader(config)
    meta = loader.load(Path(args.output))

    print(f"\n✓ Wikipedia dataset '{meta.dataset_name}' loaded at {args.output}")
    print(f"  Accepted documents:      {meta.n_documents}")
    print(f"  Skipped documents:       {meta.n_skipped}")
    print(f"  Avg segments/doc:        {meta.avg_segments_per_doc:.2f}")
    print(f"  Avg sentences/segment:   {meta.avg_sentences_per_segment:.2f}")
