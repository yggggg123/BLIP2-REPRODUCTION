"""CPU-only paired analysis of the six small result files shipped with this repo."""

from pathlib import Path
import json
import sys


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "improvement"))
from analyze_improvement_results import paired_interval


def main():
    rows = {variant: json.loads((root / "results/test" / variant / "per_question_scores.json").read_text(encoding="utf-8"))
            for variant in ("plain", "constant", "question")}
    for baseline, candidate in (("plain", "constant"), ("plain", "question"), ("constant", "question")):
        print(f"{candidate}-{baseline}: " + json.dumps(paired_interval(rows[baseline], rows[candidate]), indent=2))


if __name__ == "__main__":
    main()
