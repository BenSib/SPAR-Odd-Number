"""
Check what percentage of a results file's trials picked a specific number
(default 7) per condition — useful for spotting "favorite number" bias
(e.g. qwen3-8b-instruct saturating on 42, others on 7).
"""

import argparse

import reward_gaming_experiment as rge
from summarize_results import find_latest_result
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results", help="Directory to search for result files")
    parser.add_argument("--file", default=None, help="Summarize this specific file instead of the latest in --dir")
    parser.add_argument("--number", type=int, default=7, help="Number to check for (default: 7)")
    args = parser.parse_args()

    path = args.file or find_latest_result(args.dir)
    with open(path) as f:
        data = json.load(f)

    print(f"File:   {path}")
    print(f"Model:  {data.get('model', 'unknown')}")
    print(f"Target: {args.number}\n")

    header = f"{'condition':<32}{'attempts':>10}{'valid':>8}{f'={args.number}':>8}{'% of valid':>12}{'% of attempts':>15}"
    print(header)
    print("-" * len(header))
    for name, result in data["results"].items():
        raw = result["raw_outputs"]
        attempts = len(raw)
        parsed = [rge.parse_number(x) for x in raw]
        valid = [n for n in parsed if n is not None]
        matches = sum(1 for n in valid if n == args.number)

        pct_valid = f"{matches / len(valid):.1%}" if valid else "n/a"
        pct_attempts = f"{matches / attempts:.1%}" if attempts else "n/a"

        print(f"{name:<32}{attempts:>10}{len(valid):>8}{matches:>8}{pct_valid:>12}{pct_attempts:>15}")


if __name__ == "__main__":
    main()
