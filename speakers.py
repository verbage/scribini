"""Speaker helpers: names file, talk-time stats, and sample audio clips.

Workflow:
  1. transcribe.py labels speakers "Speaker 1", "Speaker 2", ... in order of
     first appearance and cuts a few short clips per speaker.
  2. You listen to the clips and fill in the names file.
  3. relabel.py re-renders the transcript with the real names (no re-transcribing).

Names file format (lines starting with # are ignored):
    Speaker 1: Alice
    Speaker 2: Bob
    Speaker 3 = Speaker 1        <- same person as Speaker 1 (merge)
"""

import re
import subprocess
from pathlib import Path

from render import build_turns, fmt_ts

LABEL = re.compile(r"^\s*speaker\s*(\d+)\s*$", re.IGNORECASE)
NAME_LINE = re.compile(r"^\s*(speaker\s*\d+)\s*:\s*(.*)$", re.IGNORECASE)
MERGE_LINE = re.compile(r"^\s*(speaker\s*\d+)\s*=\s*(speaker\s*\d+)\s*$", re.IGNORECASE)


def normalize_label(text):
    match = LABEL.match(text)
    if not match:
        return text.strip()
    return f"Speaker {int(match.group(1))}"


def load_names(path):
    """Return a mapping {generic label: display name} from a names file."""
    mapping = {}
    if not path:
        return mapping
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"Names file not found: {p}")
    names, alias = {}, {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        merge = MERGE_LINE.match(line)
        if merge:
            alias[normalize_label(merge.group(1))] = normalize_label(merge.group(2))
            continue
        named = NAME_LINE.match(line)
        if named and named.group(2).strip():
            names[normalize_label(named.group(1))] = named.group(2).strip()

    def canonical(label):
        seen = set()
        while label in alias and label not in seen:
            seen.add(label)
            label = alias[label]
        return label

    for label in set(names) | set(alias):
        canon = canonical(label)
        mapping[label] = names.get(canon, canon)
    return mapping


def speaker_stats(words):
    """Talk time (seconds) and word count per speaker label."""
    stats = {}
    for w in words:
        entry = stats.setdefault(w["speaker"], {"seconds": 0.0, "words": 0})
        entry["seconds"] += max(0.0, w["end"] - w["start"])
        entry["words"] += 1
    return stats


def _label_number(label):
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def make_clips(source, words, clips_dir, per_speaker=3, clip_len=10.0, min_turn=5.0):
    """Cut a few short clips per speaker from the original audio file.

    Returns {label: [{"file": Path, "start": float, "text": str}, ...]}.
    """
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    turns = build_turns(words)
    by_speaker = {}
    for t in turns:
        by_speaker.setdefault(t["speaker"], []).append(t)

    result = {}
    for label, speaker_turns in by_speaker.items():
        candidates = [t for t in speaker_turns if t["end"] - t["start"] >= min_turn]
        if not candidates:
            candidates = sorted(speaker_turns, key=lambda t: t["end"] - t["start"],
                                reverse=True)[:per_speaker]
        candidates.sort(key=lambda t: t["start"])
        if len(candidates) > per_speaker:
            step = (len(candidates) - 1) / (per_speaker - 1) if per_speaker > 1 else 0
            picks = [candidates[round(i * step)] for i in range(per_speaker)]
        else:
            picks = candidates
        result[label] = []
        for k, turn in enumerate(picks, start=1):
            length = min(clip_len, max(1.0, turn["end"] - turn["start"]))
            out = clips_dir / f"speaker{_label_number(label)}_clip{k}.wav"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{turn['start']:.2f}",
                   "-t", f"{length:.2f}", "-i", str(source), "-vn", "-ac", "1",
                   "-ar", "16000", str(out)]
            try:
                subprocess.run(cmd, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"  warning: could not cut {out.name}: {exc}")
                continue
            text = re.sub(r"\s+", " ", "".join(w["word"] for w in turn["words"])).strip()
            result[label].append({"file": out, "start": turn["start"], "text": text})
    return result


def write_names_template(path, words, clips, clips_dir_name):
    stats = speaker_stats(words)
    total = sum(s["seconds"] for s in stats.values()) or 1.0
    lines = [
        "# Name the speakers below, then run:",
        "#   python relabel.py <stem>.transcript.json --names <this file>",
        "#",
        "# One name per line, like:   Speaker 1: Alice",
        "# If one person was split into two labels, merge them:   Speaker 3 = Speaker 1",
        "# Leave a line as 'Speaker 2:' (empty) to keep the generic label.",
        "",
    ]
    for label in sorted(stats, key=_label_number):
        minutes = stats[label]["seconds"] / 60.0
        share = 100.0 * stats[label]["seconds"] / total
        lines.append(f"# {label}: about {minutes:.1f} min of speech ({share:.0f}% of the total)")
        for clip in clips.get(label, []):
            snippet = clip["text"][:110] + ("..." if len(clip["text"]) > 110 else "")
            lines.append(f"#   {clips_dir_name}/{clip['file'].name}  at {fmt_ts(clip['start'])}: {snippet}")
        lines.append(f"{label}:")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
