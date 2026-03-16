from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate shadow and occlusion adversarial variants for GTSRB."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
