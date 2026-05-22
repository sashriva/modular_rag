# Modular RAG - Free Cloud Hybrid (1 User)

This project gives you a practical RAG stack where compute runs locally (Python app) and core infra uses free cloud tiers.

If your goal is to understand the code before running it, start with [docs/code-flow.md](docs/code-flow.md).

## Architecture

- Vector DB: Qdrant Cloud
- Cache: Upstash Redis for exact, embedding, and semantic reuse
- Metadata DB: Neon Postgres (optional)
- Embeddings: Google Gemini embedding API
- LLM: Groq (Llama 3)
- Reranker: Cohere (optional, can be disabled)
- App: FastAPI + CLI

## How To Read The Code

Read in this order:

1. [docs/code-flow.md](docs/code-flow.md)
2. [src/rag_cloud/config.py](src/rag_cloud/config.py)
3. [src/rag_cloud/clients.py](src/rag_cloud/clients.py)
4. [src/rag_cloud/cache.py](src/rag_cloud/cache.py)
5. [src/rag_cloud/pipeline.py](src/rag_cloud/pipeline.py)
6. [src/rag_cloud/api.py](src/rag_cloud/api.py) or [src/rag_cloud/cli.py](src/rag_cloud/cli.py)
7. [scripts/ingest_docs.py](scripts/ingest_docs.py)
8. [scripts/bootstrap_cloud.py](scripts/bootstrap_cloud.py)

## 1. Setup

1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and fill values.
4. Tune cache behavior if needed:
	- `CACHE_TTL_EXACT`
	- `CACHE_TTL_EMBED`
	- `CACHE_TTL_SEMANTIC`
	- `SIMILARITY_THRESHOLD`

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## 2. Bootstrap Cloud Resources

This validates env variables, checks Redis and Qdrant, optionally verifies Neon,
Groq, Gemini, Cohere, and Langfuse, and runs a small end-to-end smoke test.

```powershell
python scripts/bootstrap_cloud.py
```

## 3. Add Documents and Ingest

Put documents inside `data/docs`.
Supported: `.txt`, `.md`, `.pdf`, `.docx`

```powershell
python scripts/ingest_docs.py --docs data/docs
```

For the bundled evaluation example, you can ingest the starter corpus instead:

```powershell
python scripts/ingest_docs.py --docs eval/sample_docs
```

## 4. Query Once (CLI)

```powershell
python scripts/query_once.py "What is our refund policy?"
```

Interactive CLI:

```powershell
python -m rag_cloud.cli
```

Both the CLI and API return the answer, source list, and cache outcome (`exact`, `semantic`, or `miss`).

## 5. Run API

```powershell
$env:PYTHONPATH="src"
uvicorn rag_cloud.api:app --host 127.0.0.1 --port 8000 --reload
```

Swagger docs:

- http://127.0.0.1:8000/docs

## 6. Run Evaluation Harness

This project includes a small evaluation harness with three starter samples and
three metrics: faithfulness, answer relevancy, and context recall.

```powershell
python scripts/run_eval.py
```

The harness saves JSON reports into `eval/results/`. See [eval/README.md](eval/README.md)
for the sample-doc workflow and dataset format.

## Notes

- Keep `COHERE_API_KEY` empty to disable paid reranking calls.
- Semantic cache uses scan-based cosine similarity for learning simplicity; for scale, move to a Redis vector index.
- Exact cache keys are tenant-scoped, so one tenant cannot reuse another tenant's cached answer.
- The cache TTLs and semantic threshold are configurable through `.env`.
- For multi-tenant safety, keep using `tenant_id` in query calls.
- Langfuse tracing is optional. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable spans for cache checks, retrieval, reranking, and generation.
- Neon logging is optional. Set `DATABASE_URL` to persist query logs (`rag_query_logs`) and ingestion metadata (`rag_ingested_files`).
