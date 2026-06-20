"""
Export and clean PolyCode prompt conversations from MongoDB.

The production collection is named "prompts" in the "polycode" database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATABASE = "polycode"
DEFAULT_COLLECTION = "prompts"
DEFAULT_OUTPUT = "data/processed/mongodb_prompts.json"


@dataclass(frozen=True)
class ExportStats:
    scanned: int
    exported: int
    skipped: int
    output_path: str


def build_mongodb_uri() -> str:
    """Build a MongoDB URI from either MONGODB_URI or Atlas-style env pieces."""
    load_dotenv()

    direct_uri = os.getenv("MONGODB_URI")
    if direct_uri:
        return direct_uri

    user = os.getenv("MONGODB_USER")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    if not all([user, password, cluster]):
        raise RuntimeError(
            "Set MONGODB_URI or set MONGODB_USER, MONGODB_PASSWORD, and MONGODB_CLUSTER."
        )

    return (
        f"mongodb+srv://{quote_plus(user or '')}:{quote_plus(password or '')}"
        f"@{cluster}/?retryWrites=true&w=majority"
    )


def clean_prompt_document(document: dict[str, Any]) -> dict[str, Any] | None:
    """Return the training fields needed from one MongoDB prompt document."""
    user_message = str(document.get("userMessage") or "").strip()
    assistant_message = str(document.get("assistantMessage") or "").strip()
    if not user_message or not assistant_message:
        return None

    return {
        "userMessage": user_message,
        "assistantMessage": assistant_message,
        "liked": bool(document.get("liked", False)),
    }


def _dedupe_key(sample: dict[str, Any]) -> str:
    raw = f"{sample['userMessage']}\n---\n{sample['assistantMessage']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dedupe_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for sample in samples:
        key = _dedupe_key(sample)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def export_prompts(
    output_path: str = DEFAULT_OUTPUT,
    database: str = DEFAULT_DATABASE,
    collection: str = DEFAULT_COLLECTION,
    limit: int = 0,
    only_liked: bool = False,
) -> ExportStats:
    uri = build_mongodb_uri()
    query: dict[str, Any] = {
        "userMessage": {"$type": "string", "$ne": ""},
        "assistantMessage": {"$type": "string", "$ne": ""},
    }
    if only_liked:
        query["liked"] = True

    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    try:
        source = client[database][collection]
        cursor = source.find(query, {"userMessage": 1, "assistantMessage": 1, "liked": 1})
        if limit > 0:
            cursor = cursor.limit(limit)

        scanned = 0
        cleaned: list[dict[str, Any]] = []
        for document in cursor:
            scanned += 1
            sample = clean_prompt_document(document)
            if sample:
                cleaned.append(sample)

        exported = dedupe_samples(cleaned)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")

        stats = ExportStats(
            scanned=scanned,
            exported=len(exported),
            skipped=scanned - len(exported),
            output_path=str(output),
        )
        logger.info(
            "Exported %s/%s MongoDB prompt conversations to %s",
            stats.exported,
            stats.scanned,
            stats.output_path,
        )
        return stats
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned MongoDB prompts for training.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--database", default=os.getenv("MONGODB_DB", DEFAULT_DATABASE))
    parser.add_argument("--collection", default=os.getenv("MONGODB_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-liked", action="store_true")
    args = parser.parse_args()

    stats = export_prompts(
        output_path=args.output,
        database=args.database,
        collection=args.collection,
        limit=args.limit,
        only_liked=args.only_liked,
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
