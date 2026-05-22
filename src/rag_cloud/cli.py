from __future__ import annotations

from rag_cloud.clients import Clients
from rag_cloud.config import get_settings
from rag_cloud.pipeline import RAGPipeline


"""Interactive terminal entrypoint for exploring the pipeline manually."""


def main() -> None:
    """Read questions from stdin and print answers plus source metadata."""

    settings = get_settings()
    clients = Clients(settings)
    pipeline = RAGPipeline(clients, settings)

    print("RAG CLI ready. Type 'quit' to exit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = pipeline.query_dict(question)
        print("\nAnswer:\n")
        print(result["answer"])
        print(f"\nCache: {result['cache']}")

        print("\nSources:")
        for idx, src in enumerate(result["sources"], start=1):
            score = src.get("rerank_score", src.get("retrieval_score", 0.0))
            print(f"{idx}. {src.get('source', 'unknown')} (score={score:.4f})")


if __name__ == "__main__":
    main()
