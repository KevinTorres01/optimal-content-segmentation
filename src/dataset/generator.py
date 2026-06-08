from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.dataset.schemas import DatasetConfig

# ── Topic sentence templates ──────────────────────────────────────────────────

_TEMPLATES_ES: dict[str, list[str]] = {
    "deportes": [
        "El equipo de fútbol ganó el partido con un marcador histórico.",
        "El entrenador del equipo anunció nuevas tácticas para el próximo partido.",
        "Los jugadores del equipo celebraron la victoria en el estadio.",
        "El torneo de fútbol reunió a los mejores equipos de la región.",
        "El estadio registró una asistencia récord en el partido del equipo.",
        "El capitán del equipo marcó el gol definitivo en el partido.",
        "La selección de fútbol clasifica al campeonato mundial.",
        "El partido de fútbol fue transmitido en cadena nacional.",
        "El delantero del equipo anotó tres goles en el partido.",
        "La liga de fútbol profesional inicia su nueva temporada.",
    ],
    "tecnologia": [
        "El nuevo software de inteligencia artificial procesa datos a gran velocidad.",
        "Los programadores desarrollaron una aplicación de software innovadora.",
        "El sistema de inteligencia artificial mejora la eficiencia del software.",
        "La empresa lanzó una nueva versión del software con inteligencia artificial.",
        "El procesador de última generación ejecuta el software más rápido.",
        "La inteligencia artificial del software aprende de los datos automáticamente.",
        "Los desarrolladores de software integran inteligencia artificial en la plataforma.",
        "El software utiliza inteligencia artificial para optimizar resultados.",
        "La nueva actualización del software incorpora módulos de inteligencia artificial.",
        "El equipo de desarrollo presentó el software de inteligencia artificial.",
    ],
    "ciencia": [
        "Los científicos descubrieron una nueva especie en el océano profundo.",
        "El experimento científico fue realizado en laboratorios especializados.",
        "Los investigadores científicos publicaron su descubrimiento en revista.",
        "La nueva especie descubierta por científicos cambia la taxonomía conocida.",
        "El laboratorio científico analizó muestras de la nueva especie.",
        "Los científicos replicaron el experimento para confirmar los resultados.",
        "El descubrimiento científico de la nueva especie fue premiado.",
        "Los datos del laboratorio científico respaldan la nueva hipótesis.",
        "La comunidad científica debatió sobre el descubrimiento de la especie.",
        "Los investigadores del laboratorio científico presentaron sus hallazgos.",
    ],
    "politica": [
        "El gobierno aprobó una nueva ley de política económica.",
        "El parlamento debatió la propuesta de política de gobierno.",
        "Los representantes del gobierno discutieron la nueva política social.",
        "La política del gobierno generó debate en el parlamento nacional.",
        "El presidente anunció cambios en la política económica del gobierno.",
        "El partido de gobierno presentó su propuesta de política educativa.",
        "La nueva política del parlamento impacta la economía nacional.",
        "Los ministros del gobierno defendieron la política ante el parlamento.",
        "La política económica del gobierno fue aprobada por el parlamento.",
        "El gobierno implementó medidas de política social urgentes.",
    ],
    "arte": [
        "El artista presentó su nueva obra en la galería de arte.",
        "La exposición de arte reunió obras de artistas nacionales.",
        "El museo de arte inauguró una nueva exposición de pintura.",
        "Los artistas de la galería participaron en el festival de arte.",
        "La obra de arte del pintor fue adquirida por el museo.",
        "La galería de arte exhibe la colección del artista contemporáneo.",
        "El festival de arte convocó a pintores de toda la región.",
        "El artista creó una obra de arte inspirada en la naturaleza.",
        "La pintura del artista domina la exposición del museo de arte.",
        "El museo presentó una retrospectiva del artista más destacado.",
    ],
    "economia": [
        "El banco central ajustó las tasas de interés de la economía.",
        "Los mercados financieros reaccionaron al informe económico del banco.",
        "La economía del país creció según los indicadores del banco central.",
        "El banco publicó el informe trimestral de indicadores económicos.",
        "Los analistas económicos revisaron las proyecciones del banco central.",
        "La tasa de interés del banco impacta la economía doméstica.",
        "El informe del banco central describe el crecimiento económico.",
        "Los mercados financieros reflejan la estabilidad económica del banco.",
        "El banco central implementó medidas para estabilizar la economía.",
        "Los indicadores económicos del banco señalan recuperación del mercado.",
    ],
    "salud": [
        "Los médicos desarrollaron un nuevo tratamiento médico para la enfermedad.",
        "El hospital implementó un protocolo médico para tratar la enfermedad.",
        "Los pacientes del hospital recibieron el nuevo tratamiento médico.",
        "La investigación médica del hospital encontró cura para la enfermedad.",
        "El tratamiento médico redujo los síntomas de la enfermedad.",
        "El hospital de salud pública aplica el tratamiento médico experimental.",
        "Los médicos del hospital especializados en la enfermedad publicaron estudio.",
        "El nuevo tratamiento médico fue aprobado por las autoridades de salud.",
        "Los pacientes con la enfermedad mejoraron con el tratamiento médico.",
        "El hospital investiga variantes del tratamiento médico eficaz.",
    ],
    "historia": [
        "Los historiadores descubrieron documentos históricos del siglo pasado.",
        "El archivo histórico conserva documentos de la historia nacional.",
        "Los investigadores de historia analizaron los documentos del archivo.",
        "La historia del país fue reescrita con los nuevos documentos históricos.",
        "El museo de historia exhibe los documentos del archivo nacional.",
        "Los historiadores del archivo presentaron nuevos datos históricos.",
        "El documento histórico revela aspectos desconocidos de la historia.",
        "La investigación de historia se basa en documentos del archivo.",
        "Los expertos en historia revisaron los documentos del archivo nacional.",
        "El archivo histórico digitalizó los documentos para preservar la historia.",
    ],
}

_TEMPLATES_EN: dict[str, list[str]] = {
    "sports": [
        "The football team won the championship with a record-breaking score.",
        "The coach of the football team announced new tactics for the game.",
        "The players of the football team celebrated their victory at the stadium.",
        "The football tournament brought together top teams from the region.",
        "The stadium recorded record attendance during the football team's match.",
        "The captain of the football team scored the decisive goal.",
        "The national football team qualified for the world championship.",
        "The football match was broadcast on national television.",
        "The striker of the football team scored three goals in the match.",
        "The professional football league begins its new season.",
    ],
    "technology": [
        "The new artificial intelligence software processes data at high speed.",
        "The programmers developed an innovative artificial intelligence application.",
        "The artificial intelligence system improves software efficiency.",
        "The company launched a new version of the artificial intelligence software.",
        "The latest processor runs the artificial intelligence software faster.",
        "The software's artificial intelligence learns automatically from data.",
        "Developers integrate artificial intelligence into the software platform.",
        "The software uses artificial intelligence to optimize results.",
        "The new software update incorporates artificial intelligence modules.",
        "The development team presented the artificial intelligence software.",
    ],
    "science": [
        "Scientists discovered a new species in the deep ocean.",
        "The scientific experiment was conducted in specialized laboratories.",
        "Scientific researchers published their discovery in a journal.",
        "The new species discovered by scientists changes known taxonomy.",
        "The scientific laboratory analyzed samples of the new species.",
        "Scientists replicated the experiment to confirm the results.",
        "The scientific discovery of the new species was awarded.",
        "Laboratory data from scientists supports the new hypothesis.",
        "The scientific community debated the discovery of the species.",
        "Researchers at the scientific laboratory presented their findings.",
    ],
}


@dataclass
class DatasetMetadata:
    """Metadata about a generated dataset."""

    dataset_name: str
    n_documents: int
    avg_segments_per_doc: float
    avg_sentences_per_segment: float
    config: dict[str, Any]
    generation_seed: int


class DatasetGenerator:
    """Generates synthetic text segmentation datasets from a DatasetConfig."""

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config
        self._templates = _TEMPLATES_ES if config.language == "es" else _TEMPLATES_EN
        self._topics = list(self._templates.keys())

    def generate(self, output_dir: Path) -> DatasetMetadata:
        """Generate the full dataset and save it to output_dir.

        Args:
            output_dir: Directory where documents/, boundaries/ and
                metadata.json will be created.

        Returns:
            DatasetMetadata with generation statistics.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "documents").mkdir(exist_ok=True)
        (output_dir / "boundaries").mkdir(exist_ok=True)

        rng = np.random.default_rng(self._config.random_seed)
        total_segments = 0
        total_sentences = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Generating {self._config.n_documents} documents...",
                total=self._config.n_documents,
            )

            for idx in range(self._config.n_documents):
                doc_id = f"doc_{idx + 1:04d}"
                sentences, boundaries = self._generate_document(rng)

                (output_dir / "documents" / f"{doc_id}.txt").write_text(
                    "\n".join(sentences), encoding="utf-8"
                )
                (output_dir / "boundaries" / f"{doc_id}.json").write_text(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "boundaries": boundaries,
                            "n_segments": len(boundaries),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                total_segments += len(boundaries)
                total_sentences += len(sentences)
                progress.advance(task)

        n = self._config.n_documents
        metadata = DatasetMetadata(
            dataset_name=self._config.dataset_name,
            n_documents=n,
            avg_segments_per_doc=total_segments / n,
            avg_sentences_per_segment=total_sentences / total_segments,
            config=self._config.model_dump(),
            generation_seed=self._config.random_seed,
        )

        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset_name": metadata.dataset_name,
                    "n_documents": metadata.n_documents,
                    "avg_segments_per_doc": round(metadata.avg_segments_per_doc, 2),
                    "avg_sentences_per_segment": round(
                        metadata.avg_sentences_per_segment, 2
                    ),
                    "config": metadata.config,
                    "generation_seed": metadata.generation_seed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return metadata

    def _generate_document(
        self, rng: np.random.Generator
    ) -> tuple[list[str], list[int]]:
        """Generate one document with ground-truth boundaries.

        Args:
            rng: NumPy random generator seeded from the dataset seed.

        Returns:
            Tuple of (sentences, boundaries) where boundaries[0] == 0.
        """
        cfg = self._config
        # Each segment must have a distinct topic so the ground-truth boundary
        # corresponds to a real semantic shift. Cap n_segments at the available
        # pool size — otherwise EN configs (3 topics) crash with replace=False.
        max_possible = len(self._topics)
        n_segments = int(
            rng.integers(cfg.segments_per_doc.min, cfg.segments_per_doc.max + 1)
        )
        n_segments = min(n_segments, max_possible)
        topic_indices = rng.choice(max_possible, size=n_segments, replace=False)
        chosen_topics = [self._topics[i] for i in topic_indices]

        sentences: list[str] = []
        boundaries: list[int] = []

        for topic in chosen_topics:
            boundaries.append(len(sentences))
            n_sents = int(
                rng.integers(
                    cfg.sentences_per_segment.min, cfg.sentences_per_segment.max + 1
                )
            )
            topic_sentences = self._get_topic_sentences(topic, n_sents, rng)
            sentences.extend(topic_sentences)

        return sentences, boundaries

    def _get_topic_sentences(
        self, topic: str, n: int, rng: np.random.Generator
    ) -> list[str]:
        """Sample n sentences from the given topic's template pool.

        Uses overlap_level to optionally inject shared vocabulary sentences
        at the boundaries between topics (medium/high overlap).

        Args:
            topic: Topic key (e.g. 'deportes').
            n: Number of sentences to return.
            rng: Seeded random generator.

        Returns:
            List of n sentence strings.
        """
        pool = self._templates[topic]
        indices = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        selected = [pool[i] for i in indices]

        if len(selected) < n:
            extra = rng.choice(len(pool), size=n - len(selected), replace=True)
            selected.extend(pool[i] for i in extra)

        # Overlap injection: for medium/high levels, insert slightly ambiguous
        # sentences at the end of the block to simulate fuzzy topic transitions.
        if self._config.overlap_level in ("medium", "high") and len(selected) > 1:
            n_overlap = 1 if self._config.overlap_level == "medium" else 2
            n_overlap = min(n_overlap, len(selected) - 1)
            other_topics = [t for t in self._topics if t != topic]
            for i in range(n_overlap):
                other = str(rng.choice(other_topics))
                other_pool = self._templates[other]
                selected[-(i + 1)] = other_pool[int(rng.integers(0, len(other_pool)))]

        return selected[:n]


if __name__ == "__main__":
    import argparse

    import yaml

    from src.dataset.schemas import DatasetConfig

    parser = argparse.ArgumentParser(description="Generate a synthetic dataset")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = DatasetConfig.model_validate(raw)

    generator = DatasetGenerator(config)
    meta = generator.generate(Path(args.output))

    print(f"\n✓ Dataset '{meta.dataset_name}' generated at {args.output}")
    print(f"  Documents:               {meta.n_documents}")
    print(f"  Avg segments/doc:        {meta.avg_segments_per_doc:.2f}")
    print(f"  Avg sentences/segment:   {meta.avg_sentences_per_segment:.2f}")
