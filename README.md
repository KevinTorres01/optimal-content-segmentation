# Optimal Content Segmentation

Research project on **optimal text segmentation**: given a long document, divide it into semantically coherent segments using combinatorial optimisation algorithms, optionally evaluated by an LLM for semantic cohesion scoring.

---

## Overview

The core problem is: given a document of *n* sentences, find the *k* boundaries that maximise internal semantic cohesion and minimise inter-segment similarity.

Four algorithms are implemented and compared under the same objective function:

| Algorithm | Complexity | Description |
|---|---|---|
| Brute Force | O(C(n-1,k-1)·n) | Exhaustive enumeration — exact, only for n ≤ 15 |
| Dynamic Programming | O(n²·k) | Exact optimal solution for arbitrary n |
| Greedy | O(n·w) | TextTiling-inspired — fast baseline heuristic |
| Simulated Annealing | O(n² + iters) | Metaheuristic — near-optimal, escapes local optima |

All algorithms share the same **length-weighted cosine cohesion** objective:

```
cohesion(i, j) = mean_pairwise_cosine_similarity(sentences[i..j]) × (j−i+1) / n
```

Sentence vectors are computed using TF-IDF. The length weighting prevents degenerate solutions where many single-sentence segments always appear optimal.

---

## Project Structure

```
src/
├── core/
│   ├── models.py          # Pydantic data models: Document, Segment, SegmentationResult, CohesionScore, ...
│   ├── interfaces.py      # ABCs: BaseSegmenter, BaseLLMEvaluator
│   ├── config.py          # YAML config loading and validation
│   └── env.py             # Loads .env automatically via python-dotenv
├── algorithms/
│   ├── _cohesion.py       # Shared TF-IDF cohesion matrix builder
│   ├── brute_force.py     # BruteForceSegmenter
│   ├── dynamic_programming.py  # DPSegmenter
│   ├── greedy.py          # GreedySegmenter
│   └── simulated_annealing.py  # SASegmenter
├── llm/
│   ├── groq_provider.py   # Groq (primary, free tier)
│   ├── mistral_provider.py # Mistral (fallback online)
│   ├── fallback_provider.py # FallbackEvaluator: tries primary → fallback → neutral
│   ├── factory.py         # get_llm_provider(config) factory
│   ├── rate_limit.py      # Exponential backoff retry + request throttle
│   ├── prompts.py         # Shared prompt template + JSON response parser
│   └── check.py           # CLI connectivity checker
├── dataset/
│   ├── generator.py       # DatasetGenerator — synthetic datasets in ES/EN
│   ├── wikipedia_loader.py # WikipediaLoader — fetches real articles via MediaWiki API
│   └── schemas.py         # DatasetConfig & WikipediaDatasetConfig Pydantic schemas
├── evaluation/
│   └── metrics.py         # WindowDiff, Pk, F1-boundary (no external deps)
└── experiments/
    └── runner.py          # Orchestrates: load config → generate/load data → run algorithms → save results

config/
├── datasets/
│   ├── small.yaml                    # 20-document synthetic dataset, 12–40 sentences/doc
│   ├── tiny.yaml                     # 20-document synthetic dataset, 4–12 sentences/doc (BF-friendly)
│   └── wikipedia.yaml                # 28 Spanish-Wikipedia titles → 25 accepted after filtering
└── experiments/
    ├── smoke_test.yaml               # End-to-end test without LLM
    ├── exp_compare_algorithms.yaml   # DP vs Greedy vs SA on `small`, no LLM
    ├── exp_bf_vs_dp.yaml             # Brute Force vs DP on `tiny` (correctness validation)
    ├── exp_llm_groq.yaml             # Full experiment with Groq LLM evaluation
    └── exp_wikipedia.yaml            # DP vs Greedy vs SA on real Wikipedia text

tests/
├── unit/
│   ├── test_models.py
│   ├── test_dp_segmenter.py
│   ├── test_algorithms.py  # BruteForce, Greedy, SA
│   ├── test_generator.py
│   ├── test_llm_factory.py
│   └── test_wikipedia_loader.py
├── integration/
│   ├── test_smoke.py
│   └── test_llm_runner.py
└── mocks.py               # MockLLMEvaluator — no API calls in tests
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (optional — needed only for LLM evaluation)
cp .env.example .env
# Edit .env and fill in your keys
```

### Environment Variables

```bash
# .env

# Groq (primary LLM evaluator — free tier, no credit card required)
# Get your key at: console.groq.com
GROQ_API_KEY=

# Mistral (fallback evaluator — free tier)
# Get your key at: console.mistral.ai
MISTRAL_API_KEY=

# Fallback behaviour
LLM_FALLBACK_ENABLED=true          # auto-fallback to Mistral if Groq fails
LLM_TIMEOUT_SECONDS=30             # timeout before trying fallback
LLM_MIN_REQUEST_INTERVAL_SECONDS=1.5  # throttle to avoid rate limits
```

If you don't have API keys, all experiments can be run with `provider: none` to skip LLM scoring — all structural metrics (WindowDiff, Pk, F1) still work.

---

## Generating a Dataset

```bash
python -m src.dataset.generator \
    --config config/datasets/small.yaml \
    --output data/small/
```

This creates:
- `data/small/documents/` — one `.txt` file per document
- `data/small/boundaries/` — one `.json` file per document with ground-truth boundary positions
- `data/small/metadata.json` — dataset statistics and config

The generator creates synthetic multi-topic documents in Spanish (or English). Topics include sports, technology, politics, science, art, economics, health, and history. The `overlap_level` parameter controls how much vocabulary is shared between topics.

### Wikipedia dataset (real text, requires internet)

In addition to the synthetic datasets, you can build a dataset from real Spanish Wikipedia articles. The level-2 section headings (`==`) act as ground-truth boundaries.

```bash
python -m src.dataset.wikipedia_loader \
    --config config/datasets/wikipedia.yaml \
    --output data/wikipedia/
```

The loader downloads the configured titles via the MediaWiki API, splits each article into sentences, truncates to match the size envelope of the `small` dataset, and writes the same `documents/`, `boundaries/`, `metadata.json` layout. It honours a 1.5 s inter-request delay and retries with exponential backoff on HTTP 429.

---

## Running an Experiment

```bash
# Without LLM (no API key needed)
python -m src.experiments.runner --config config/experiments/smoke_test.yaml

# With Groq LLM evaluation (requires GROQ_API_KEY in .env)
python -m src.experiments.runner --config config/experiments/exp_llm_groq.yaml
```

Results are written to the path specified in `output.path`:
- `results.json` — one entry per document with all metric values
- `summary.csv` — aggregated statistics per algorithm
- `run_metadata.json` — experiment config, timestamp, random seed

---

## Experiment Configuration

```yaml
# config/experiments/example.yaml
experiment_id: example
description: "Compare all algorithms on small dataset"

dataset:
  path: data/small/

algorithms:
  - name: dynamic_programming
    params:
      max_segments: 5
  - name: greedy
    params:
      max_segments: 5
  - name: simulated_annealing
    params:
      max_segments: 5
      n_iterations: 2000
      random_seed: 42

llm_evaluator:
  provider: groq          # groq | mistral | none
  model: llama-3.3-70b-versatile
  temperature: 0.0
  max_tokens: 512

evaluation:
  metrics: [windowdiff, pk, f1_boundary, llm_score, runtime_seconds]
  random_seed: 42

output:
  path: results/example/
  save_raw: true
```

### Available Algorithms

| Name in config | Class | Key params |
|---|---|---|
| `brute_force` | `BruteForceSegmenter` | `max_segments` (n ≤ 15 only) |
| `dynamic_programming` | `DPSegmenter` | `max_segments` |
| `greedy` | `GreedySegmenter` | `max_segments`, `window_size` (default 2) |
| `simulated_annealing` | `SASegmenter` | `max_segments`, `n_iterations`, `initial_temp`, `cooling_rate`, `random_seed` |

### LLM Providers

| Name in config | Model | Notes |
|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Free tier, primary evaluator |
| `mistral` | `mistral-large-latest` | Free tier, fallback |
| `none` | — | Disables LLM scoring; structural metrics only |

---

## Evaluation Metrics

| Metric | Range | Lower = Better | Description |
|---|---|---|---|
| `pk` | [0, 1] | Yes | Pk error: probability of mis-classifying a random sentence pair |
| `windowdiff` | [0, 1] | Yes | WindowDiff: penalises near-miss boundaries more than Pk |
| `f1_boundary` | [0, 1] | No | F1 score on detected boundary positions (tolerance ±1) |
| `llm_score` | [1, 5] | No | Mean LLM cohesion score across segments |
| `runtime_seconds` | ≥ 0 | Yes | Wall-clock time for the algorithm |

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Only unit tests (no API calls, fast)
python -m pytest tests/unit/ -v

# Only integration tests
python -m pytest tests/integration/ -v
```

All tests use mocked LLM evaluators — no API keys required to run the full test suite.

---

## Checking LLM Connectivity

```bash
python -m src.llm.check --provider groq
python -m src.llm.check --provider mistral
```

This sends one real API call to verify the key and endpoint are reachable.

---

## Adding a New Algorithm

1. Create `src/algorithms/my_algorithm.py` extending `BaseSegmenter`:

```python
from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult

class MySegmenter(BaseSegmenter):
    @property
    def name(self) -> str:
        return "my_algorithm"

    def segment(self, document: Document, max_segments: int | None = None) -> SegmentationResult:
        ...
```

2. Register it in `src/algorithms/__init__.py`:

```python
from src.algorithms.my_algorithm import MySegmenter

ALGORITHM_REGISTRY["my_algorithm"] = MySegmenter
```

3. Use it in any experiment config:

```yaml
algorithms:
  - name: my_algorithm
    params:
      max_segments: 5
```
