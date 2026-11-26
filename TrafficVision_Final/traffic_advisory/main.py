"""main.py - command-line demo.

  python main.py                      run every sample scenario
  python main.py --scenario heavy     run one sample (free/moderate/heavy/rising/easing)
  python main.py --offline            skip the LLM, use the fallback only
  python main.py --input t.json [--prediction p.json]   use your own JSON files
"""
import argparse
import json
import sys

from advisory import generate_advisory
from sample_data import SAMPLES


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="TrafficVision AI Advisory demo")
    parser.add_argument("--scenario", choices=SAMPLES.keys())
    parser.add_argument("--input", help="JSON file with traffic data")
    parser.add_argument("--prediction", help="JSON file with prediction data")
    parser.add_argument("--offline", action="store_true", help="do not call the LLM")
    args = parser.parse_args()

    if args.input:
        try:
            runs = [("custom input", _load(args.input),
                     _load(args.prediction) if args.prediction else None)]
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read JSON file: {exc}")
            return 1
    else:
        names = [args.scenario] if args.scenario else list(SAMPLES)
        runs = [(n, *SAMPLES[n]) for n in names]

    for name, traffic, prediction in runs:
        print(f"\n=== {name} ===")
        result = generate_advisory(traffic, prediction, use_llm=not args.offline)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
