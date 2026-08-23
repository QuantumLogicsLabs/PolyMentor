import os
import sys

# Add repository root directory to sys.path
repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root_dir not in sys.path:
    sys.path.insert(0, repo_root_dir)

from src.data_pipeline.prompt_exporter import main


if __name__ == "__main__":
    main()
