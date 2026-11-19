#!/usr/bin/env python3
"""Quick environment check. Run this before your first transcription.

    python check_setup.py                  # checks libraries, GPU (if any), ffmpeg, model folder
    python check_setup.py --device cpu     # CPU-only machine: GPU lines are skipped
    python check_setup.py --load-model     # also loads the diarization model

A missing GPU is not a failure: the system still works on the CPU, only more slowly.
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transcribe import DEFAULT_DIAR_MODEL, detect_gpu  # noqa: E402


def report(ok, label, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return ok


def note(label, detail=""):
    print(f"  [note] {label}" + (f": {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto (default): report what is found; cuda: a GPU is required; "
                             "cpu: skip the GPU checks")
    parser.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}")
    all_ok = True

    all_ok &= report(shutil.which("ffmpeg") is not None, "ffmpeg on PATH")

    for module, label in [("torch", "PyTorch"), ("faster_whisper", "faster-whisper"),
                          ("pyannote.audio", "pyannote.audio"), ("docx", "python-docx"),
                          ("soundfile", "soundfile")]:
        try:
            __import__(module)
            all_ok &= report(True, f"import {label}")
        except Exception as exc:  # noqa: BLE001
            all_ok &= report(False, f"import {label}", str(exc))

    cpu_only = False
    if args.device == "cpu":
        note("GPU checks skipped (--device cpu)")
        cpu_only = True
    else:
        gpu = detect_gpu()
        if gpu["whisper"]:
            report(True, "faster-whisper (CTranslate2) sees an NVIDIA GPU")
        elif args.device == "cuda":
            all_ok &= report(False, "faster-whisper (CTranslate2) sees an NVIDIA GPU")
        else:
            cpu_only = True
        if gpu["torch"]:
            report(True, "PyTorch sees the GPU", gpu["name"] or "")
        elif gpu["whisper"] or args.device == "cuda":
            note("PyTorch does not see the GPU", "speaker identification will use the CPU")

    model_dir = Path(args.diar_model).expanduser()
    all_ok &= report((model_dir / "config.yaml").exists(), "diarization model folder", str(model_dir))

    if args.load_model and (model_dir / "config.yaml").exists():
        try:
            from pyannote.audio import Pipeline
            Pipeline.from_pretrained(str(model_dir))
            all_ok &= report(True, "diarization model loads")
        except Exception as exc:  # noqa: BLE001
            all_ok &= report(False, "diarization model loads", str(exc))

    print()
    if not all_ok:
        print("Some checks failed; see the Troubleshooting section of the README.")
    elif cpu_only:
        print("All checks passed, in CPU-only mode.")
        print("Transcription will work but is much slower than on a GPU. Tips: use")
        print("--model large-v3-turbo for speed, and --threads N to limit CPU use.")
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
