"""
Summarize every result file in a directory, one after another.

By default, skips files tagged _DEBUG (small smoke-test runs) so real
experiment data isn't lost in the noise. Pass --include-debug to see those too.
"""

import argparse
import glob
import os

from summarize_results import summarize_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results", help="Directory to search for result files")
    parser.add_argument(
        "--include-debug", action="store_true",
        help="Also summarize files tagged _DEBUG (small smoke-test runs)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")), key=os.path.getmtime)
    if not args.include_debug:
        files = [f for f in files if "_DEBUG" not in os.path.basename(f)]

    if not files:
        print(f"No result files found in {args.dir}")
        return

    for i, path in enumerate(files):
        if i > 0:
            print("\n" + "=" * 78 + "\n")
        summarize_file(path)


if __name__ == "__main__":
    main()
