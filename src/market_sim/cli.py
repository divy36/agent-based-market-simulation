"""Run the three experiments and save tables, plots and execution metadata."""

import argparse
import contextlib
import importlib
import json
import platform
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["all", "constraints", "mixed_market", "head_to_head"],
        default="all",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small verification runs; not research estimates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = (
        ["constraints", "mixed_market", "head_to_head"]
        if args.experiment == "all"
        else [args.experiment]
    )
    settings = (
        {
            "constraints": {"n_runs": 3, "n_ticks": 30, "sizes": (20, 40, 60)},
            "mixed_market": {"n_ticks": 40, "seeds": (11, 22, 33)},
            "head_to_head": {"n_runs": 3, "n_ticks": 40},
        }
        if args.quick
        else {}
    )
    manifest = {
        "mode": "quick" if args.quick else "full",
        "python": platform.python_version(),
        "experiments": {},
    }
    for name in names:
        start = time.perf_counter()
        folder = args.output / name
        folder.mkdir()
        try:
            with (folder / "run.txt").open("w") as log, contextlib.redirect_stdout(log):
                tables = importlib.import_module(f"market_sim.experiments.{name}").run(
                    **settings.get(name, {})
                )
            for index, table in enumerate(tables):
                table.to_csv(folder / f"table_{index+1}.csv", index=False)
            for index, number in enumerate(plt.get_fignums(), 1):
                plt.figure(number).savefig(
                    folder / f"figure_{index}.png", dpi=120, bbox_inches="tight"
                )
            manifest["experiments"][name] = {
                "status": "passed",
                "seconds": round(time.perf_counter() - start, 2),
                "rows": [len(t) for t in tables],
                "overrides": settings.get(name, {}),
            }
            print(f"{name}: completed", flush=True)
        except Exception:
            manifest["experiments"][name] = {"status": "failed"}
            raise
        finally:
            plt.close("all")
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
