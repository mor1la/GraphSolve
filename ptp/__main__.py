"""CLI: audit incomplete GraphML, then solve explicitly completed input data."""

import argparse
import json
from pathlib import Path

from .data import audit_and_template, load_instance
from .solver import placement, ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("graphml")
    audit.add_argument("--output", default="results/sample")
    for command in ("placement", "ranking"):
        p = sub.add_parser(command)
        p.add_argument("input", help="Completed input JSON")
        p.add_argument("--output", required=True)
        p.add_argument(
            "--seconds", type=float, default=60, help="Time limit per MILP solve"
        )
        if command == "ranking":
            p.add_argument("--placement", required=True, help="Placement result JSON")
            p.add_argument("--max-packages", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit_and_template(args.graphml, args.output)
        else:
            if args.seconds <= 0:
                raise ValueError("seconds must be positive")
            instance = load_instance(args.input)
            if args.command == "placement":
                result = placement(instance, args.seconds)
            else:
                from .solver import verify

                saved = json.loads(Path(args.placement).read_text(encoding="utf-8"))
                selected = saved["selected_gms"]
                verify(instance, selected, saved["assignments"], require_all=True)
                result = ranking(instance, selected, args.seconds, args.max_packages)
                result["placement_proven_optimal"] = saved.get("proven_optimal", False)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("validation", "assignments", "packages", "rows")
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"Input/validation error: {error}\n")


if __name__ == "__main__":
    main()
