"""Rendering for the local transcription system.

Turns a list of words (each with start, end, text and a speaker label) into
.txt and .docx transcripts. Also reads the hint list (terms to favor plus
"misheard => correct" fix-ups).

Layout rules (from the spec):
  * A timestamp starts every speaker turn.
  * Inside a long single-speaker stretch, an inline timestamp appears roughly
    every `interval` seconds (default 60), placed at a sentence boundary.
  * Lines inside one speaker's turn are single-spaced.
  * Two blank lines separate one speaker's turn from the next.
"""

import re
import textwrap
from datetime import datetime
from pathlib import Path

TXT_WIDTH = 100
SENTENCE_END = re.compile(r"[.!?\u2026]['\"\u201d\u2019)\]]*$")


def fmt_ts(seconds):
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------------
# Hint list
# --------------------------------------------------------------------------

def load_hints(path):
    """Read a hint file.

    Format, one entry per line:
        # comment
        Kubernetes              <- a term to favor
        cuban tay => Kubernetes <- a fix-up applied after transcription

    Returns (terms, fixups) where fixups is a list of (compiled_regex, replacement).
    """
    terms, fixups = [], []
    if not path:
        return terms, fixups
    p = Path(path).expanduser()
    if not p.exists():
        return terms, fixups
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            wrong, right = (part.strip() for part in line.split("=>", 1))
            if wrong:
                pattern = re.compile(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", re.IGNORECASE)
                fixups.append((pattern, right))
        else:
            terms.append(line)
    return terms, fixups


def apply_fixups(text, fixups):
    for pattern, replacement in fixups:
        text = pattern.sub(replacement, text)
    return text


# --------------------------------------------------------------------------
# Turn and segment building
# --------------------------------------------------------------------------

def build_turns(words, mapping=None):
    """Group consecutive words by speaker. `mapping` renames/merges labels."""
    mapping = mapping or {}
    turns = []
    for w in words:
        name = mapping.get(w["speaker"], w["speaker"])
        if not turns or turns[-1]["speaker"] != name:
            turns.append({"speaker": name, "words": []})
        turns[-1]["words"].append(w)
    for t in turns:
        t["start"] = t["words"][0]["start"]
        t["end"] = t["words"][-1]["end"]
    return turns


def split_segments(words, interval):
    """Split one turn's words into segments, each starting with a timestamp."""
    segments, current, seg_start = [], [], None
    for i, w in enumerate(words):
        if seg_start is None:
            seg_start = w["start"]
        current.append(w)
        is_last = i == len(words) - 1
        if (not is_last and interval and interval > 0
                and w["end"] - seg_start >= interval
                and SENTENCE_END.search(w["word"].strip())):
            segments.append((seg_start, current))
            current, seg_start = [], None
    if current:
        segments.append((seg_start, current))
    return segments


def segment_text(words, fixups):
    text = "".join(w["word"] for w in words)
    text = re.sub(r"\s+", " ", text).strip()
    return apply_fixups(text, fixups)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def render_txt(turns, interval, fixups, header_lines):
    lines = list(header_lines)
    if lines:
        lines += ["", ""]
    for ti, turn in enumerate(turns):
        for si, (start, ws) in enumerate(split_segments(turn["words"], interval)):
            prefix = f"[{fmt_ts(start)}] "
            if si == 0:
                prefix += f"{turn['speaker']}: "
            lines.append(textwrap.fill(segment_text(ws, fixups), width=TXT_WIDTH,
                                       initial_indent=prefix, subsequent_indent=""))
        if ti != len(turns) - 1:
            lines += ["", ""]
    return "\n".join(lines) + "\n"


def render_docx(turns, path, title, meta, interval, fixups):
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError as exc:
        raise SystemExit("python-docx is not installed. Run: pip install python-docx") from exc

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)

    doc.add_heading(title, level=1)
    meta_par = doc.add_paragraph()
    meta_run = meta_par.add_run(meta)
    meta_run.font.size = Pt(10)
    meta_run.font.color.rgb = RGBColor(0x6B, 0x6B, 0x6B)
    meta_par.paragraph_format.space_after = Pt(18)

    for ti, turn in enumerate(turns):
        segments = split_segments(turn["words"], interval)
        for si, (start, ws) in enumerate(segments):
            par = doc.add_paragraph()
            fmt = par.paragraph_format
            fmt.space_before = Pt(0)
            fmt.line_spacing = 1.0
            last_in_turn = si == len(segments) - 1
            fmt.space_after = Pt(20) if last_in_turn else Pt(2)
            ts_run = par.add_run(f"[{fmt_ts(start)}] ")
            ts_run.font.size = Pt(9.5)
            ts_run.font.color.rgb = RGBColor(0x6B, 0x6B, 0x6B)
            if si == 0:
                name_run = par.add_run(f"{turn['speaker']}: ")
                name_run.bold = True
            par.add_run(segment_text(ws, fixups))
    doc.save(str(path))


def render_all(data, outdir, stem, formats=("txt", "docx"), interval=60,
               mapping=None, fixups=None):
    """Write the requested outputs. Returns the list of files written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fixups = fixups or []
    turns = build_turns(data["words"], mapping)
    if not turns:
        raise SystemExit("No speech was found in the transcript data.")

    when = datetime.now().strftime("%Y-%m-%d")
    meta = (f"Transcribed {when} | language: {data.get('language', '?')} | "
            f"length: {fmt_ts(data.get('duration', 0))} | source: {Path(data.get('source', '')).name}")
    written = []
    if "txt" in formats:
        header = [stem, meta]
        path = outdir / f"{stem}.txt"
        path.write_text(render_txt(turns, interval, fixups, header), encoding="utf-8")
        written.append(path)
    if "docx" in formats:
        path = outdir / f"{stem}.docx"
        render_docx(turns, path, stem, meta, interval, fixups)
        written.append(path)
    return written
