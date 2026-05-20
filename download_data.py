"""Download the HarmBench behavior seed set for the Colab MVP."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
DEFAULT_OUTPUT = Path("data/harmbench_behaviors.json")


def download_harmbench(output_path: Path = DEFAULT_OUTPUT, overwrite: bool = False) -> Path:
    """Download HarmBench behaviors and validate that the result is JSON."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"[data] Found existing dataset: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading HarmBench behaviors from:\n       {HARMBENCH_URL}")

    try:
        with urlopen(HARMBENCH_URL, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(
            "Failed to download HarmBench behaviors. "
            f"Manual URL: {HARMBENCH_URL}"
        ) from exc

    data = parse_harmbench_csv(raw)

    if not isinstance(data, list) or not data:
        raise RuntimeError("Downloaded HarmBench file did not contain a non-empty list.")

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data] Saved {len(data)} behaviors to {output_path}")
    return output_path


def parse_harmbench_csv(raw: str) -> list[dict[str, str]]:
    """Parse the upstream CSV and keep a JSON shape compatible with colab_main."""
    reader = csv.DictReader(StringIO(raw))
    if not reader.fieldnames or "Behavior" not in reader.fieldnames:
        raise RuntimeError(
            "Downloaded HarmBench CSV did not include a Behavior column. "
            f"Columns found: {reader.fieldnames}"
        )

    rows: list[dict[str, str]] = []
    for row in reader:
        behavior = (row.get("Behavior") or "").strip()
        if behavior:
            rows.append({key: (value or "") for key, value in row.items()})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HarmBench behavior data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_harmbench(Path(args.output), overwrite=args.overwrite)
