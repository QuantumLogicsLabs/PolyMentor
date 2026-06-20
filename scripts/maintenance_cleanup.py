from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

LARGE_FOLDER_CANDIDATES = [
    "data",
    "models_saved",
    "experiments/logs",
    "venv",
    "venv312",
    "polymentor.egg-info",
    "website/node_modules",
]


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def remove_generated_files(root: Path, dry_run: bool) -> list[str]:
    removed: list[str] = []

    for cache_dir in root.rglob("*"):
        if not cache_dir.is_dir() or cache_dir.name not in CACHE_DIR_NAMES:
            continue
        removed.append(str(cache_dir.relative_to(root)))
        if not dry_run:
            shutil.rmtree(cache_dir, ignore_errors=True)

    for pyc_file in root.rglob("*.pyc"):
        removed.append(str(pyc_file.relative_to(root)))
        if not dry_run:
            pyc_file.unlink(missing_ok=True)

    return sorted(set(removed))


def report_large_folders(root: Path, min_mb: float) -> list[tuple[str, int]]:
    threshold = int(min_mb * 1024 * 1024)
    report: list[tuple[str, int]] = []

    for relative in LARGE_FOLDER_CANDIDATES:
        path = root / relative
        if not path.exists():
            continue
        size = path_size(path)
        if size >= threshold:
            report.append((relative, size))

    return sorted(report, key=lambda item: item[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated caches and report large local folders.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-mb", type=float, default=50)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    removed = remove_generated_files(root, dry_run=args.dry_run)
    large_folders = report_large_folders(root, min_mb=args.min_mb)

    mode = "Would remove" if args.dry_run else "Removed"
    print(f"{mode} {len(removed)} generated cache path(s).")
    for item in removed[:50]:
        print(f"- {item}")
    if len(removed) > 50:
        print(f"- ... {len(removed) - 50} more")

    print("")
    print("Large local folders to review:")
    if not large_folders:
        print("- None above threshold.")
    for relative, size in large_folders:
        print(f"- {relative}: {format_bytes(size)}")


if __name__ == "__main__":
    main()
