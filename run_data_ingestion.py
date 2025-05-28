#!/usr/bin/env python3
"""
Data Ingestion Launcher - Run from project root.
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
src_root = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_root))

# Import and run the data ingestion
from src.rag_engine.data_processing.data_ingestion import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data ingestion with environment-aware model selection"
    )
    parser.add_argument(
        "--root-directory",
        default="data/aiap17-gitlab-data",
        help="Root directory containing data",
    )
    parser.add_argument(
        "--limit-people",
        type=int,
        help="Limit number of people to process (for testing)",
    )
    parser.add_argument(
        "--limit-files", type=int, help="Limit files per person (for testing)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (3 people, 10 files each)"
    )

    args = parser.parse_args()

    if args.test:
        print("🧪 Running in test mode")
        main(args.root_directory, limit_people=3, limit_files_per_person=10)
    else:
        main(args.root_directory, args.limit_people, args.limit_files)
