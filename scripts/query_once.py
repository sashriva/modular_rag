from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_cloud.clients import Clients
from rag_cloud.config import get_settings
from rag_cloud.pipeline import RAGPipeline


"""Small non-interactive entrypoint used to test the query path quickly."""


def main() -> None:
    """Run a single query and print the answer and cache result."""

    parser = argparse.ArgumentParser(description="Run one RAG query")
    parser.add_argument("query", help="User query")
    parser.add_argument("--tenant", default="default", help="Tenant id for cache keying")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RAGPipeline(Clients(settings), settings)
    result = pipeline.query_dict(args.query, tenant_id=args.tenant)

    print("Answer:\n")
    print(result["answer"])
    print(f"\nCache: {result['cache']}")


if __name__ == "__main__":
    main()
