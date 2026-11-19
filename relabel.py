#!/usr/bin/env python3
"""Re-render a finished transcript with real speaker names (no re-transcribing).

Example:
    python relabel.py ~/scribini/output/meeting/meeting.transcript.json \
        --names ~/scribini/output/meeting/meeting.names.txt
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from render import load_hints, render_all  # noqa: E402
from speakers import load_names  # noqa: E402

SUFFIX = ".transcript.json"


def main():
    p = argparse.ArgumentParser(description="Re-render a transcript with speaker names.")
    p.add_argument("transcript", help="the <name>.transcript.json file")
    p.add_argument("--names", default=None, help="names file (Speaker 1: Alice ...)")
    p.add_argument("--hints", default=None, help="hint list file (default: hints.txt next to this script)")
    p.add_argument("--format", choices=["both", "txt", "docx"], default="both")
    p.add_argument("--ts-interval", type=int, default=60)
    p.add_argument("--outdir", default=None, help="default: the folder of the transcript file")
    args = p.parse_args()

    json_path = Path(args.transcript).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"Transcript file not found: {json_path}")
    if not json_path.name.endswith(SUFFIX):
        raise SystemExit(f"Expected a file ending in {SUFFIX}")
    stem = json_path.name[: -len(SUFFIX)]
    data = json.loads(json_path.read_text(encoding="utf-8"))

    mapping = load_names(args.names)
    hints_path = args.hints
    if hints_path is None and (HERE / "hints.txt").exists():
        hints_path = str(HERE / "hints.txt")
    _, fixups = load_hints(hints_path)

    outdir = Path(args.outdir).expanduser() if args.outdir else json_path.parent
    formats = ("txt", "docx") if args.format == "both" else (args.format,)
    written = render_all(data, outdir, stem, formats=formats, interval=args.ts_interval,
                         mapping=mapping, fixups=fixups)
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
