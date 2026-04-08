# rag-eval-harness

A pipeline-agnostic evaluation framework for RAG (Retrieval-Augmented Generation) systems. Swap retrievers, generators, and metrics via YAML config. Benchmark different pipeline combinations and get JSON + HTML reports.

## What This Is

`rag-eval-harness` evaluates RAG pipelines by:
- Loading a dataset of questions and ground truth answers
- Retrieving context documents from a vector database (Pinecone, pgvector, etc.)
- Generating answers using an LLM (OpenAI, Anthropic, etc.)
- Scoring results with multiple metrics (latency, RAGAS, LLM judge)
- Comparing pipeline configurations side-by-side

All components (retriever, generator, metrics) are swappable via YAML config. Add new implementations without touching the core orchestrator.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ CLI (run / compare)                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ EvalRunner (orchestrator)                                       │
│  • Loads YAML config                                           │
│  • Instantiates components via registry pattern                │
│  • Runs eval loop: retrieve → generate → score                 │
└────────────────────┬──────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┬───────────────────┐
         ▼           ▼           ▼                   ▼
    ┌────────────────────┐  ┌─────────────┐  ┌──────────────────┐
    │ Retriever          │  │ Generator   │  │ Metrics          │
    ├────────────────────┤  ├─────────────┤  ├──────────────────┤
    │ • Pinecone         │  │ • OpenAI    │  │ • Latency        │
    │ • pgvector         │  │ • Anthropic │  │ • LLM Judge      │
    │ • Custom           │  │ • Custom    │  │ • RAGAS Metrics  │
    └────────────────────┘  └─────────────┘  └──────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ ReportBuilder (generates output)                                │
├─────────────────────────────────────────────────────────────────┤
│ • JSON: structured results + summary statistics                │
│ • HTML: interactive report with metric tables                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Clone and install

```bash
git clone <repo>
cd rag-eval-harness
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY or ANTHROPIC_API_KEY
# - PINECONE_API_KEY (if using Pinecone)
# - PGVECTOR_DSN (if using pgvector)
```

### 3. Run an evaluation

```bash
python cli.py run --config configs/pipeline_a.yaml --dataset data/sample_dataset.jsonl
```

Output:
```
JSON report: reports/pipeline_a_20260408T120000_abc123.json
HTML report: reports/pipeline_a_20260408T120000_abc123.html
```

### 4. Compare pipelines

```bash
python cli.py compare \
  --configs configs/pipeline_a.yaml configs/pipeline_b.yaml configs/pipeline_c.yaml \
  --dataset data/sample_dataset.jsonl
```

Output:
```
[pipeline_a] JSON: reports/...json  HTML: reports/...html
[pipeline_b] JSON: reports/...json  HTML: reports/...html
[pipeline_c] JSON: reports/...json  HTML: reports/...html

--- Comparison Summary ---
configs/pipeline_a.yaml: latency=125.5ms latency=0.125 ragas_faithfulness=0.842 llm_judge=0.800
configs/pipeline_b.yaml: latency=89.3ms latency=0.089 ragas_faithfulness=0.756 llm_judge=0.720
configs/pipeline_c.yaml: latency=156.2ms latency=0.156 ragas_faithfulness=0.891 llm_judge=0.850
```

## Configuration

YAML configuration drives everything. Structure:

```yaml
name: pipeline_name

retriever:
  type: pinecone                # or: pgvector, custom_retriever
  index_name: rag-eval-index    # Pinecone-specific
  top_k: 3

generator:
  type: openai                  # or: anthropic, custom_generator
  model: gpt-4o                 # or: claude-3-5-sonnet-20241022

metrics:
  - type: latency
  - type: ragas_faithfulness
  - type: ragas_answer_relevancy
  - type: llm_judge
```

### Supported retriever types

| Type | Config | Notes |
|---|---|---|
| `pinecone` | `index_name`, `top_k` | Uses OpenAI `text-embedding-3-small` |
| `pgvector` | `table`, `top_k` | Cosine similarity via `<=>` operator |

### Supported generator types

| Type | Config | Notes |
|---|---|---|
| `openai` | `model` | Default: `gpt-4o` |
| `anthropic` | `model` | Default: `claude-3-5-sonnet-20241022` |

### Supported metric types

| Type | Notes |
|---|---|
| `latency` | Retrieve + generate time (in seconds) |
| `llm_judge` | GPT-4o scores 1-5, normalized to [0, 1] |
| `ragas_faithfulness` | RAGAS faithfulness metric |
| `ragas_answer_relevancy` | RAGAS answer relevancy metric |

## Adding a Custom Retriever

1. **Create a new file** `retrievers/my_retriever.py`:

```python
from core.base import BaseRetriever
from core.registry import register_retriever

@register_retriever("my_type")
class MyRetriever(BaseRetriever):
    def retrieve(self, question: str) -> list[str]:
        # Your logic here
        return ["context1", "context2"]
```

2. **Register the import** in `core/runner.py`:

```python
import retrievers.my_retriever  # Add this line
```

3. **Use it in YAML**:

```yaml
retriever:
  type: my_type
  custom_setting: value
```

Same pattern for generators and metrics (extend `BaseGenerator` or `BaseMetric`).

## Docker

Run with Docker Compose (includes PostgreSQL + pgvector):

```bash
# Set up environment
cp .env.example .env
# ... edit .env with keys ...

# Start services
docker-compose up --build

# Run evaluation in another terminal
docker-compose run harness run --config configs/pipeline_a.yaml --dataset data/sample_dataset.jsonl

# View reports
ls reports/
```

Or build the image yourself:

```bash
docker build -t rag-eval-harness:latest .
docker run \
  --env-file .env \
  -v $(pwd)/reports:/app/reports \
  rag-eval-harness:latest \
  run --config configs/pipeline_a.yaml --dataset data/sample_dataset.jsonl
```

## Kubernetes

### 1. Create secrets

```bash
kubectl create secret generic rag-eval-secrets --from-env-file=.env
```

### 2. Apply manifests

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Swap pipelines without rebuilding

Edit the ConfigMap to use a different pipeline:

```bash
kubectl create configmap rag-eval-pipeline-config \
  --from-file=pipeline.yaml=configs/pipeline_b.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# Rolling update picks up the new config
kubectl rollout restart deployment/rag-eval-harness
```

### 4. View reports

Reports are written to the `emptyDir` volume. Copy from a pod:

```bash
kubectl cp rag-eval-harness-xxxx:/app/reports ./local-reports
```

## Tests

Run the test suite:

```bash
pytest tests/ -v
```

Run tests excluding RAGAS (if not installed):

```bash
pytest tests/ -v -k "not ragas"
```

Key test files:

- `test_base.py`: ABC enforcement
- `test_loader.py`: JSONL parsing and validation
- `test_runner.py`: end-to-end orchestration (with fake components)
- `test_report.py`: JSON/HTML generation
- `test_llm_judge.py`: LLM judge metric scoring and clamping
- `conftest.py`: shared fixtures

All external APIs are mocked. Tests are isolated (use temp files/directories).

## Why Each Decision Was Made

### Registry Pattern (not `importlib.import_module` from strings)

**Why:** Explicit import graph in `runner.py` makes dependencies traceable. Fails fast if a module has a syntax error. No dynamic module discovery magic.

**Trade-off:** Requires adding one import line to `runner.py` per custom component. Worth it for clarity and debugging.

### `argparse` (not Click/Typer)

**Why:** Zero extra dependencies. The two subcommands (`run`, `compare`) are simple enough that Click's features (command groups, parameter types, shell completion) add unnecessary complexity.

### `text-embedding-3-small` for both retrievers

**Why:** Consistent embedding space across pipelines. Cost-efficient (~$0.02 per 1M tokens vs. $0.13 for `text-embedding-3-large`). 1536 dimensions sufficient for semantic search.

### LatencyMetric has `set_latency()` method

**Why:** Keeps `BaseMetric.score()` synchronous while allowing the runner to inject latency measurements. Without this, latency would either (a) be computed twice (in metric and runner), or (b) require a protocol change to `score()`. The `set_latency()` approach is the least invasive.

### RAGAS async wrapped in ThreadPoolExecutor

**Why:** RAGAS uses async internally, but we need a synchronous interface for `BaseMetric.score()`. Using `asyncio.run()` in a thread pool safely handles both cases:
- Caller has no event loop → `asyncio.run()` in main thread
- Caller has event loop → `asyncio.run()` in thread pool (avoids "RuntimeError: asyncio.run() cannot be called from a running event loop")

### ConfigMap mounts pipeline YAML in K8s

**Why:** Decouples configuration from container image. Swap pipelines with `kubectl apply` (no rebuild, no redeployment, just a rolling update that picks up the new ConfigMap). Enables fast A/B testing in production.

### JSONL format for datasets

**Why:** One question/answer pair per line. Easy to stream, grep, and append. No need for SQL or HDF5.

### HTML reports (not Jupyter notebooks)

**Why:** Self-contained, portable, no kernel dependencies. Can be viewed in any browser or emailed. Easy to archive alongside JSON for reproducibility.

## Project Structure

```
rag-eval-harness/
├── core/                      # Core abstractions
│   ├── base.py               # ABC classes + ResultRecord
│   ├── registry.py           # Decorator-based registration
│   ├── runner.py             # EvalRunner orchestrator
│   └── report.py             # JSON + HTML generation
├── retrievers/               # Pluggable retrievers
│   ├── pinecone_retriever.py
│   └── pgvector_retriever.py
├── generators/               # Pluggable generators
│   ├── openai_generator.py
│   └── anthropic_generator.py
├── metrics/                  # Pluggable metrics
│   ├── latency.py
│   ├── llm_judge.py
│   └── ragas_metrics.py
├── datasets/                 # Dataset loading
│   └── loader.py
├── configs/                  # Pipeline configurations
│   ├── pipeline_a.yaml
│   ├── pipeline_b.yaml
│   └── pipeline_c.yaml
├── data/                     # Sample datasets
│   └── sample_dataset.jsonl
├── reports/                  # Output directory
├── tests/                    # Test suite
│   ├── conftest.py
│   ├── test_base.py
│   ├── test_loader.py
│   ├── test_runner.py
│   ├── test_report.py
│   └── test_llm_judge.py
├── k8s/                      # Kubernetes manifests
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── cli.py                    # CLI entry point
├── Dockerfile                # Container image
├── docker-compose.yml        # Local dev setup
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Standard Python gitignore
└── README.md                 # This file
```

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes, write tests
3. Open a pull request to `main`
4. Ensure tests pass: `pytest tests/ -v`

## License

MIT