"""
Ingestion script for PolyMentor documentation into Pinecone vector database.

Reads repository markdown files, performs intelligent header-aware chunking,
generates embeddings using RAGClient (local sentence-transformers), and upserts to Pinecone.
"""

import os
import re
import sys
import glob
import hashlib
import logging
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add repository root directory to sys.path
repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root_dir not in sys.path:
    sys.path.insert(0, repo_root_dir)

from src.inference.rag_client import RAGClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_docs")


def clean_text(text: str) -> str:
    """Clean and normalize whitespace in markdown text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_markdown(
    content: str,
    file_path: str,
    max_chunk_size: int = 800,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Intelligently split markdown content into chunks based on headers and token/character limits.

    Args:
        content: Full text of markdown file.
        file_path: Relative or absolute path to the file.
        max_chunk_size: Maximum character count per chunk.
        overlap: Character overlap between consecutive split chunks.

    Returns:
        List of chunk dictionaries with metadata.
    """
    cleaned = clean_text(content)
    file_name = os.path.basename(file_path)

    # Split by markdown headers (# Header)
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    matches = list(header_pattern.finditer(cleaned))

    sections: List[Tuple[str, str]] = []

    if not matches:
        sections.append(("General", cleaned))
    else:
        # Preamble before first header
        first_start = matches[0].start()
        if first_start > 0:
            preamble = cleaned[:first_start].strip()
            if preamble:
                sections.append(("Preamble", preamble))

        for idx, match in enumerate(matches):
            header_title = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            section_text = cleaned[start_pos:end_pos].strip()

            if section_text:
                sections.append((header_title, section_text))

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for header, section_body in sections:
        # If section fits within max_chunk_size, keep as single chunk
        if len(section_body) <= max_chunk_size:
            chunk_id = hashlib.md5(f"{file_name}:{header}:{chunk_counter}".encode('utf-8')).hexdigest()[:16]
            chunks.append({
                "id": f"doc-{chunk_id}",
                "text": f"### {header}\n{section_body}",
                "metadata": {
                    "source_path": file_path,
                    "file_name": file_name,
                    "header": header,
                    "chunk_index": chunk_counter,
                    "character_count": len(section_body),
                }
            })
            chunk_counter += 1
        else:
            # Overlapping window splitting for larger sections
            start = 0
            while start < len(section_body):
                end = start + max_chunk_size
                piece = section_body[start:end].strip()

                if piece:
                    chunk_id = hashlib.md5(f"{file_name}:{header}:{chunk_counter}".encode('utf-8')).hexdigest()[:16]
                    chunks.append({
                        "id": f"doc-{chunk_id}",
                        "text": f"### {header}\n{piece}",
                        "metadata": {
                            "source_path": file_path,
                            "file_name": file_name,
                            "header": header,
                            "chunk_index": chunk_counter,
                            "character_count": len(piece),
                        }
                    })
                    chunk_counter += 1

                start += (max_chunk_size - overlap)

    return chunks


def discover_markdown_files(repo_root: str, docs_dir: str = "docs") -> List[str]:
    """Discover all relevant markdown documentation files in the repo."""
    target_files = []

    # 1. Root level markdown files
    root_path = Path(repo_root)
    for root_md in root_path.glob("*.md"):
        target_files.append(str(root_md))

    # 2. Files in docs directory
    docs_path = root_path / docs_dir
    if docs_path.exists():
        for doc_md in docs_path.rglob("*.md"):
            target_files.append(str(doc_md))

    # Remove duplicates and sort
    unique_files = sorted(list(set(target_files)))
    return unique_files


def ingest_documentation(
    repo_root: str = ".",
    docs_dir: str = "docs",
    namespace: str = "docs",
    batch_size: int = 50,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute full ingestion process: read files, chunk, embed, and upsert to Pinecone.
    """
    logger.info("Initializing RAG Client...")
    rag_client = RAGClient()

    files = discover_markdown_files(repo_root, docs_dir)
    logger.info(f"Discovered {len(files)} markdown file(s) for ingestion.")

    all_chunks: List[Dict[str, Any]] = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            rel_path = os.path.relpath(file_path, repo_root)
            file_chunks = chunk_markdown(content, rel_path)
            all_chunks.extend(file_chunks)
            logger.info(f"Parsed '{rel_path}': generated {len(file_chunks)} chunk(s).")
        except Exception as e:
            logger.error(f"Failed to read or chunk '{file_path}': {e}")

    logger.info(f"Total chunks created across all documents: {len(all_chunks)}")

    if not all_chunks:
        logger.warning("No chunks created. Ingestion finished empty.")
        return {"total_files": len(files), "total_chunks": 0, "status": "empty"}

    logger.info("Generating embeddings locally using sentence-transformers...")
    vectors_payload = []
    
    # Process embedding generation in batches
    texts_to_embed = [chunk["text"] for chunk in all_chunks]
    embeddings = rag_client.embed_text(texts_to_embed)

    for chunk, vector in zip(all_chunks, embeddings):
        vectors_payload.append({
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                **chunk["metadata"],
                "text": chunk["text"]
            }
        })

    logger.info(f"Successfully generated {len(vectors_payload)} vector embeddings (dim={len(embeddings[0]) if embeddings else 0}).")

    if dry_run:
        logger.info("[DRY RUN] Skipping Pinecone network upsert.")
        return {
            "total_files": len(files),
            "total_chunks": len(all_chunks),
            "sample_chunk": all_chunks[0] if all_chunks else None,
            "status": "dry_run_success"
        }

    logger.info(f"Upserting {len(vectors_payload)} vector payloads to Pinecone namespace '{namespace}'...")
    result = rag_client.upsert_documents(vectors_payload, namespace=namespace, batch_size=batch_size)

    return {
        "total_files": len(files),
        "total_chunks": len(all_chunks),
        "upsert_result": result,
        "status": result.get("status", "completed")
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest markdown documentation into Pinecone RAG knowledge base.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root directory.")
    parser.add_argument("--docs-dir", default="docs", help="Documentation subdirectory name.")
    parser.add_argument("--namespace", default="docs", help="Pinecone namespace.")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for vector upserts.")
    parser.add_argument("--dry-run", action="store_true", help="Chunk and embed without uploading to Pinecone.")

    args = parser.parse_args()

    summary = ingest_documentation(
        repo_root=args.repo_root,
        docs_dir=args.docs_dir,
        namespace=args.namespace,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

    logger.info(f"Ingestion complete! Summary: {summary}")


if __name__ == "__main__":
    main()
