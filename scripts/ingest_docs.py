from __future__ import annotations

import argparse
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import google.generativeai as genai
from docx import Document
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag_cloud.config import get_settings
from rag_cloud.metadata_store import MetadataStore


"""Document ingestion script.

This script converts local files into chunk embeddings and writes them to the
configured Qdrant collection. Read this file to understand how retrieval data is
created before it is queried by the runtime pipeline.
"""


def read_txt(path: Path) -> str:
    """Read a UTF-8 text or markdown file."""

    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    """Extract concatenated text from a PDF file."""

    pdf = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in pdf.pages)


def read_docx(path: Path) -> str:
    """Extract paragraph text from a DOCX file."""

    doc = Document(str(path))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_any(path: Path) -> str:
    """Dispatch file parsing based on extension."""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return ""


def chunk_words(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows for simple dense retrieval."""

    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size]).strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks


def embed(model: str, text: str, embed_dim: int) -> list[float]:
    """Embed one chunk as a retrieval document vector using Gemini."""

    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document",
        output_dimensionality=embed_dim,
    )
    return result["embedding"]


def main() -> None:
    """Load local files, chunk them, embed them, and upsert them to Qdrant."""

    parser = argparse.ArgumentParser(description="Ingest local docs into Qdrant cloud")
    parser.add_argument("--docs", default="data/docs", help="Folder containing docs")
    args = parser.parse_args()

    load_dotenv()
    settings = get_settings()

    genai.configure(api_key=settings.google_api_key)
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_version=False)
    metadata = MetadataStore(settings.database_url) if settings.database_url else None

    docs_dir = Path(args.docs)
    docs_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}
    ]

    if not files:
        print("No files found. Add .txt, .md, .pdf, or .docx into data/docs")
        return

    total_chunks = 0
    for file in files:
        text = read_any(file)
        if not text.strip():
            print(f"Skipped empty/unsupported file: {file.name}")
            continue

        chunks = chunk_words(text)
        points: list[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            vector = embed(settings.embed_model, chunk, settings.embed_dim)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": file.name,
                        "chunk_id": idx,
                    },
                )
            )

        qdrant.upsert(collection_name=settings.collection, points=points, wait=True)
        if metadata:
            try:
                metadata.upsert_ingested_file(file.name, len(points))
            except Exception:
                pass
        total_chunks += len(points)
        print(f"Indexed {len(points)} chunks from {file.name}")

    print(f"Done. Indexed {total_chunks} chunks from {len(files)} files.")


if __name__ == "__main__":
    main()
