from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_cloud.config import get_settings
from rag_cloud.eval_harness import EvalHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small RAG evaluation harness")
    parser.add_argument("--dataset", default="eval/golden_set.json", help="Path to golden dataset JSON")
    parser.add_argument("--out", default="eval/results", help="Folder to store eval reports")
    args = parser.parse_args()

    harness = EvalHarness(get_settings())
    report = harness.run(args.dataset)
    output_path = harness.save(report, args.out)

    print("Evaluation summary:")
    for metric, score in report["summary"].items():
        print(f"- {metric}: {score}")
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
