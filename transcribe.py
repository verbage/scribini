#!/usr/bin/env python3
"""Local audio -> speaker-labeled, timestamped transcript (.txt and .docx).

Pipeline: ffmpeg preprocess -> Whisper (faster-whisper) -> pyannote diarization
-> merge words with speakers -> save result -> write outputs and speaker clips.

Examples:
    python transcribe.py meeting.mp3 --language en --speakers 3
    python transcribe.py meeting.mp3 --device cpu --threads 6 --model large-v3-turbo
"""

import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from render import load_hints, render_all  # noqa: E402
from speakers import make_clips, write_names_template  # noqa: E402

DEFAULT_DIAR_MODEL = "~/models/pyannote-community-1"
DEFAULT_OUTPUT_ROOT = "~/scribini/output"

PRESET_FILTERS = {
    "off": None,
    "light": "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
    # "strong" uses ffmpeg's built-in spectral denoiser. A true AI denoiser can be added later.
    "strong": "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
}


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def preload_cuda_libs():
    """Load the pip-installed cuBLAS/cuDNN libraries so faster-whisper can find them.

    faster-whisper (CTranslate2) needs cuBLAS 12 and cuDNN 9. When they come from
    pip packages they are not on the library path by default, so we load them
    into the process ourselves before importing faster_whisper.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    except Exception:
        roots = []
    wanted = [("cublas", "libcublasLt.so.12"), ("cublas", "libcublas.so.12"),
              ("cudnn", "libcudnn.so.9")]
    for package, lib in wanted:
        for root in roots:
            candidate = Path(root) / package / "lib" / lib
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def detect_gpu():
    """Report which parts of the pipeline can use a GPU.

    Returns {"whisper": bool, "torch": bool, "name": str or None}:
      whisper - faster-whisper (CTranslate2) can see an NVIDIA GPU
      torch   - PyTorch (used for speaker identification) can see a GPU
    """
    preload_cuda_libs()
    info = {"whisper": False, "torch": False, "name": None}
    try:
        import ctranslate2
        info["whisper"] = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["torch"] = True
            info["name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def thread_count(requested):
    """CPU threads to use: all of them by default, or fewer if asked."""
    total = os.cpu_count() or 4
    if not requested or requested <= 0:
        return total
    return min(requested, total)


def resolve_devices(args):
    """Decide where each stage runs; sets args.whisper_device and args.diar_device."""
    if args.device == "cpu":
        args.whisper_device = args.diar_device = "cpu"
        return

    gpu = detect_gpu()
    if args.device == "cuda":
        if not gpu["whisper"]:
            raise SystemExit(
                "--device cuda was requested, but no usable NVIDIA GPU was found.\n"
                "Run check_setup.py to see why, or use --device cpu."
            )
        args.whisper_device = "cuda"
        args.diar_device = "cuda" if gpu["torch"] else "cpu"
    elif gpu["whisper"]:
        args.whisper_device = "cuda"
        args.diar_device = "cuda" if gpu["torch"] else "cpu"
        print(f"GPU detected: {gpu['name'] or 'NVIDIA GPU'}")
    else:
        print("\nWARNING: no usable NVIDIA GPU was detected.")
        print("  Transcription will run on the CPU only, which is much slower than on a GPU")
        print("  (with the large-v3 model it is often slower than real time).")
        print("  Tips: --model large-v3-turbo is much faster, and --threads N limits CPU use.")
        print("  To skip this question next time, add --device cpu.")
        if sys.stdin.isatty():
            answer = input("Continue on the CPU? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                raise SystemExit("Stopped. Nothing was processed.")
        else:
            print("  (not an interactive session, so continuing on the CPU)")
        args.whisper_device = "cpu"
        args.diar_device = "cuda" if gpu["torch"] else "cpu"

    if args.whisper_device == "cuda" and args.diar_device == "cpu":
        print("Note: PyTorch cannot use the GPU here, so speaker identification will run on the CPU.")


def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def step(number, total, message):
    print(f"\n[{number}/{total}] {message}", flush=True)


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

def preprocess(src, dst, preset):
    """Convert any input to clean 16 kHz mono WAV (optionally filtered)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    audio_filter = PRESET_FILTERS[preset]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def transcribe_words(wav, args, terms, duration):
    """Run Whisper and return (words, detected_language)."""
    preload_cuda_libs()
    try:
        import torch  # noqa: F401  (loads CUDA libraries early)
    except ImportError:
        pass
    from faster_whisper import WhisperModel

    if args.whisper_device == "cuda":
        compute_type = "int8_float16" if args.low_vram else "float16"
    else:
        compute_type = "int8"
    print(f"    loading {args.model} ({args.whisper_device}, {compute_type}) ...", flush=True)
    model = WhisperModel(args.model, device=args.whisper_device, compute_type=compute_type,
                         cpu_threads=args.threads_used)

    kwargs = dict(
        language=None if args.language == "auto" else args.language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    glossary = ", ".join(terms[:60])
    if glossary:
        # Newer faster-whisper has a hotwords option; older ones only take a prompt.
        import inspect
        if "hotwords" in inspect.signature(model.transcribe).parameters:
            kwargs["hotwords"] = glossary
        else:
            kwargs["initial_prompt"] = "Glossary: " + glossary + "."

    # Hand Whisper the samples directly instead of a file path. The WAV was made by
    # ffmpeg in step 1 (16 kHz mono), so faster-whisper's own decoder (PyAV) is not needed.
    import soundfile as sf
    audio, _sample_rate = sf.read(str(wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    segments, info = model.transcribe(audio, **kwargs)

    print(f"    language: {info.language} (probability {info.language_probability:.2f})", flush=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
        print(f"\r    {seg.end / 60:6.1f} / {duration / 60:.1f} min", end="", flush=True)
    print()
    language = info.language
    del model
    free_gpu_memory()
    return words, language


def diarize(wav, args):
    """Run pyannote and return a list of (start, end, label) speaker turns."""
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    torch.set_num_threads(args.threads_used)

    model_dir = Path(args.diar_model).expanduser()
    if not (model_dir / "config.yaml").exists():
        raise SystemExit(
            f"Diarization model not found at {model_dir} (no config.yaml).\n"
            "Download pyannote/speaker-diarization-community-1 there, or pass --diar-model PATH."
        )
    print(f"    loading diarization model from {model_dir} ...", flush=True)
    pipeline = Pipeline.from_pretrained(str(model_dir))
    if args.diar_device == "cuda":
        pipeline.to(torch.device("cuda"))

    data, sample_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = {"waveform": torch.from_numpy(data.T.copy()), "sample_rate": sample_rate}

    options = {}
    if args.speakers:
        options["num_speakers"] = args.speakers
    else:
        if args.min_speakers:
            options["min_speakers"] = args.min_speakers
        if args.max_speakers:
            options["max_speakers"] = args.max_speakers

    print("    identifying speakers ...", flush=True)
    result = pipeline(audio, **options)

    annotation = result
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        candidate = getattr(result, attr, None)
        if candidate is not None:
            annotation = candidate
            break
    turns = [(float(t.start), float(t.end), str(label))
             for t, _, label in annotation.itertracks(yield_label=True)]
    del pipeline
    free_gpu_memory()
    return turns


def assign_speakers(words, turns):
    """Give every word the speaker whose turn overlaps it most (or the nearest turn)."""
    if not turns:
        for w in words:
            w["speaker"] = "SPEAKER_00"
        return
    for w in words:
        best, best_overlap = None, 0.0
        for start, end, label in turns:
            overlap = min(end, w["end"]) - max(start, w["start"])
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is None:
            mid = (w["start"] + w["end"]) / 2

            def distance(turn):
                if turn[0] <= mid <= turn[1]:
                    return 0.0
                return min(abs(mid - turn[0]), abs(mid - turn[1]))

            best = min(turns, key=distance)[2]
        w["speaker"] = best


def relabel_by_appearance(words):
    """Rename raw labels to 'Speaker 1', 'Speaker 2', ... in order of first appearance."""
    order = {}
    for w in words:
        raw = w["speaker"]
        if raw not in order:
            order[raw] = f"Speaker {len(order) + 1}"
        w["speaker"] = order[raw]
    return list(order.values())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Local speaker-labeled transcription.")
    p.add_argument("input", help="audio file (mp3, wav, m4a, ...)")
    p.add_argument("--language", default="auto", help="auto, en, es, ... (default auto)")
    p.add_argument("--speakers", type=int, default=None, help="exact number of speakers, if known")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--preset", choices=list(PRESET_FILTERS), default="off",
                   help="noise-reduction preset (default off)")
    p.add_argument("--format", choices=["both", "txt", "docx"], default="both")
    p.add_argument("--ts-interval", type=int, default=60,
                   help="seconds between inline timestamps in a long single-speaker stretch (0 = off)")
    p.add_argument("--hints", default=None, help="hint list file (default: hints.txt next to this script)")
    p.add_argument("--model", default="large-v3", help="Whisper model (default large-v3)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="auto (default): use the NVIDIA GPU if there is one, otherwise warn and ask "
                        "before using the CPU; cuda: require the GPU; cpu: use the CPU without asking")
    p.add_argument("--threads", type=int, default=0,
                   help="limit the CPU threads used (default 0 = all cores). Fewer threads run "
                        "slower but keep the CPU cooler and quieter. Only affects CPU work.")
    p.add_argument("--low-vram", action="store_true", help="GPU only: use int8_float16 to save GPU memory")
    p.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL, help="path to the pyannote model folder")
    p.add_argument("--no-diarize", action="store_true", help="skip speaker identification (single speaker)")
    p.add_argument("--outdir", default=None, help=f"output folder (default {DEFAULT_OUTPUT_ROOT}/<name>)")
    p.add_argument("--keep-wav", action="store_true", help="keep the temporary preprocessed WAV")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Run: sudo apt install ffmpeg")

    # Thread limit first (before numpy/torch load), then decide where each stage runs.
    args.threads_used = thread_count(args.threads)
    if args.threads and args.threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads_used)
    resolve_devices(args)
    print(f"Transcription on: {args.whisper_device} | speaker identification on: {args.diar_device} "
          f"| CPU threads: {args.threads_used}")

    stem = src.stem
    outdir = Path(args.outdir).expanduser() if args.outdir else Path(DEFAULT_OUTPUT_ROOT).expanduser() / stem
    outdir.mkdir(parents=True, exist_ok=True)

    hints_path = args.hints
    if hints_path is None and (HERE / "hints.txt").exists():
        hints_path = str(HERE / "hints.txt")
    terms, fixups = load_hints(hints_path)
    if hints_path:
        print(f"Hint list: {hints_path} ({len(terms)} terms, {len(fixups)} fix-ups)")

    formats = ("txt", "docx") if args.format == "both" else (args.format,)
    total_steps = 5
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="transcribe_"))
    try:
        wav = work / "audio.wav"

        step(1, total_steps, f"Preprocessing audio (preset: {args.preset})")
        preprocess(src, wav, args.preset)
        import soundfile as sf
        duration = float(sf.info(str(wav)).duration)
        print(f"    audio length: {duration / 60:.1f} min")

        step(2, total_steps, "Transcribing speech")
        t = time.time()
        words, language = transcribe_words(wav, args, terms, duration)
        print(f"    {len(words)} words in {(time.time() - t) / 60:.1f} min")
        if not words:
            raise SystemExit("No speech was detected in this file.")

        if args.no_diarize:
            step(3, total_steps, "Skipping speaker identification (--no-diarize)")
            for w in words:
                w["speaker"] = "SPEAKER_00"
        else:
            step(3, total_steps, "Identifying speakers")
            t = time.time()
            turns = diarize(wav, args)
            print(f"    {len(turns)} speaker turns in {(time.time() - t) / 60:.1f} min")
            assign_speakers(words, turns)
        speaker_labels = relabel_by_appearance(words)
        print(f"    speakers found: {len(speaker_labels)}")

        step(4, total_steps, "Saving result")
        data = {
            "format_version": 1,
            "source": str(src),
            "created": datetime.now().isoformat(timespec="seconds"),
            "language": language,
            "duration": round(duration, 2),
            "whisper_model": args.model,
            "diarization": None if args.no_diarize else args.diar_model,
            "preset": args.preset,
            "speakers": speaker_labels,
            "words": [{"start": round(w["start"], 2), "end": round(w["end"], 2),
                       "word": w["word"], "speaker": w["speaker"]} for w in words],
        }
        json_path = outdir / f"{stem}.transcript.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

        step(5, total_steps, "Writing outputs and speaker clips")
        written = render_all(data, outdir, stem, formats=formats, interval=args.ts_interval,
                             fixups=fixups)
        written.append(json_path)

        names_path = outdir / f"{stem}.names.txt"
        if len(speaker_labels) >= 1 and not args.no_diarize:
            clips_dir_name = f"{stem}_speakers"
            clips = make_clips(src, data["words"], outdir / clips_dir_name)
            if names_path.exists():
                print(f"    keeping your existing names file: {names_path.name}")
            else:
                write_names_template(names_path, data["words"], clips, clips_dir_name)
                written.append(names_path)

        print(f"\nDone in {(time.time() - started) / 60:.1f} min. Files in {outdir}:")
        for path in written:
            print(f"  {path.name}")
        if not args.no_diarize:
            print("\nTo use real names: listen to the clips, edit the names file, then run:")
            print(f"  python relabel.py \"{json_path}\" --names \"{names_path}\"")
    finally:
        if args.keep_wav:
            print(f"Temporary WAV kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
