"""
Unit tests for RAG Client and Document Ingestion Pipeline.
"""

import os
import pytest
from src.inference.rag_client import RAGClient
from scripts.ingest_docs import chunk_markdown, discover_markdown_files, ingest_documentation


def test_rag_client_embed_single_string():
    client = RAGClient(api_key="mock_key")
    vec = client.embed_text("Test embedding sentence for PolyMentor.")
    
    assert isinstance(vec, list)
    assert len(vec) == client.embedding_dimension
    assert all(isinstance(val, float) for val in vec)


def test_rag_client_embed_batch_strings():
    client = RAGClient(api_key="mock_key")
    sentences = ["First sentence.", "Second sentence for embedding."]
    vecs = client.embed_text(sentences)
    
    assert isinstance(vecs, list)
    assert len(vecs) == 2
    assert len(vecs[0]) == client.embedding_dimension
    assert len(vecs[1]) == client.embedding_dimension


def test_rag_client_query_mock_mode():
    client = RAGClient(api_key="mock_key")
    results = client.query_knowledge_base("how does analyzer work?", top_k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3
    for match in results:
        assert "id" in match
        assert "score" in match
        assert "metadata" in match
        assert "text" in match["metadata"]


def test_markdown_chunking_headers():
    md_content = """# PolyMentor Overview
PolyMentor is an AI mentor.

## Feature 1
Supports local static code analysis.

## Feature 2
Supports grounded LLM responses using Groq.
"""
    chunks = chunk_markdown(md_content, file_path="docs/test.md", max_chunk_size=500)
    
    assert len(chunks) >= 2
    headers = [chunk["metadata"]["header"] for chunk in chunks]
    assert "PolyMentor Overview" in headers or "Feature 1" in headers
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["source_path"] == "docs/test.md"


def test_discover_markdown_files(tmp_path):
    # Create temporary doc files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_file = docs_dir / "guide.md"
    doc_file.write_text("# Guide\nSample documentation.")

    discovered = discover_markdown_files(repo_root=str(tmp_path), docs_dir="docs")
    assert len(discovered) == 1
    assert "guide.md" in discovered[0]


def test_ingest_docs_dry_run():
    summary = ingest_documentation(
        repo_root=".",
        docs_dir="docs",
        namespace="test-namespace",
        batch_size=10,
        dry_run=True
    )
    
    assert summary["status"] == "dry_run_success"
    assert summary["total_files"] > 0
    assert summary["total_chunks"] > 0
    assert summary["sample_chunk"] is not None
