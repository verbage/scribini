# Scribini

*Private, local audio transcription with speaker labels.*

Scribini turns recordings (mp3, wav, m4a, ...) of up to about 2 hours into a transcript that says **who spoke when**, as plain text and Word (.docx). Everything runs on your own computer. No audio, text, or voice data is sent anywhere. The internet is needed only for the one-time downloads of software and models.

**Pipeline:** ffmpeg (audio cleanup) → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (speech to text) → [pyannote.audio](https://github.com/pyannote/pyannote-audio) (who spoke) → merge → `.txt` / `.docx`.

**Features**

- Speaker labels (`Speaker 1`, `Speaker 2`, ...) with a timestamp at every speaker change, plus an inline timestamp about every minute inside a long single-speaker stretch
- Naming speakers *after* transcription: short audio clips per speaker, a names file, and a re-render step that does not re-transcribe
- English, Spanish and many other languages (set per file, or auto-detected)
- An editable hint list for names and jargon, with "misheard => correct" fix-ups
- Optional noise-reduction presets and an optional "number of speakers" hint
- Works on an NVIDIA GPU (fast) or on the CPU only (slower)

## What you need

| | NVIDIA GPU machine | CPU-only machine |
| --- | --- | --- |
| Hardware | NVIDIA GPU with 8 GB+ VRAM recommended | Any modern CPU; 16 GB+ RAM recommended |
| OS | Linux, or Windows with WSL2 | Linux, or Windows with WSL2 |
| Speed | Roughly 10x faster than real time (see below) | Slower than real time with the largest model |
| Software | Python 3.10+, ffmpeg, git, git-lfs | the same |

**Recording length.** Recordings up to about 2 hours are the design target. There is no hard limit in the code, but processing time and memory grow with length, and very long files are untested. Speaker labels are only consistent within a single run, so if you split a long recording into pieces, the labels restart for each piece. Split at a natural pause if you must.

## Install

Commands are for Linux or a WSL2 Ubuntu shell.

```bash
sudo apt update && sudo apt install -y ffmpeg git git-lfs python3-venv
git lfs install

mkdir -p ~/scribini
git clone https://github.com/verbage/scribini.git ~/scribini/app

python3 -m venv ~/scribini/venv
source ~/scribini/venv/bin/activate        # repeat in every new terminal
cd ~/scribini/app                          # the folder holding Scribini
pip install -U pip
```

**NVIDIA GPU machine** (on Windows, install the normal NVIDIA driver; do not install a Linux driver inside WSL):

```bash
pip install -r requirements.txt -r requirements-nvidia.txt
```

**CPU-only machine:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The Whisper model (about 3 GB) downloads by itself the first time you transcribe. It needs no login.

## Get the speaker model (one time)

The speaker-identification model, [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), is free (CC-BY-4.0 license) but **gated**: you must accept its terms on Hugging Face. It is not included in this project.

1. Create a Hugging Face account and open the model page. Fill in the access form.
2. Create a **read** access token at <https://huggingface.co/settings/tokens>. Keep it private.
3. Download the model once (this is the only step that needs the token):

   ```bash
   pip install -U huggingface_hub
   hf auth login
   hf download pyannote/speaker-diarization-community-1 --local-dir ~/models/pyannote-community-1
   ls ~/models/pyannote-community-1/config.yaml
   ```

After that the model loads from disk with no network. You can archive the folder and copy it to other machines. If the license asks for attribution, keep it with your copy.

## Check your setup

```bash
python check_setup.py                 # NVIDIA machine
python check_setup.py --device cpu    # CPU-only machine
```

Every line should say `[ok]`. A missing GPU is not an error; it just means CPU-only mode.

## Use it

```bash
python transcribe.py recording.mp3 --language en
python transcribe.py recording.mp3 --language es --speakers 4
```

Results go to `~/scribini/output/<name>/`:

| File | What it is |
| --- | --- |
| `<name>.txt`, `<name>.docx` | the transcripts |
| `<name>.transcript.json` | the full saved result (words, timings, speakers); used for re-rendering |
| `<name>.names.txt` | a template for naming speakers |
| `<name>_speakers/` | a few short audio clips per speaker, to help you identify who is who |

### Options

| Option | Meaning |
| --- | --- |
| `--language` | `auto` (default), `en`, `es`, ... Setting it explicitly is more reliable. |
| `--speakers N` | You know exactly how many people spoke. Improves accuracy. |
| `--min-speakers`, `--max-speakers` | A range, if you only roughly know. |
| `--device` | `auto` (default): use the NVIDIA GPU if present, otherwise warn and ask before using the CPU. `cuda`: require the GPU. `cpu`: use the CPU without asking. |
| `--threads N` | Limit CPU threads (default: all cores). Fewer threads run slower but keep the CPU cooler and quieter. Only affects CPU work; it does not switch the device. |
| `--model` | Whisper model. Default `large-v3`. `large-v3-turbo` is much faster with a small accuracy cost; `medium` and `small` are lighter still. |
| `--format` | `both` (default), `txt`, or `docx`. |
| `--ts-interval S` | Seconds between inline timestamps in a long single-speaker stretch. Default 60; 0 turns them off. |
| `--preset` | Noise reduction: `off` (default), `light` (high-pass + loudness), `strong` (adds a spectral denoiser). Test on a copy first; heavy cleanup can hurt accuracy. |
| `--hints FILE` | Hint list. Default: `hints.txt` next to the scripts. |
| `--diar-model PATH` | Where the speaker model lives. Default `~/models/pyannote-community-1`. |
| `--low-vram` | GPU only: use less video memory. |
| `--no-diarize` | Skip speaker identification (one speaker). |
| `--outdir DIR` | Output folder. |

Unattended runs (no keyboard): with `--device auto` and no GPU, the program prints the warning and continues on the CPU instead of waiting for an answer.

## Naming speakers

1. Listen to the clips in `<name>_speakers/` and open `<name>.names.txt`.
2. Type a name after each label, for example `Speaker 1: Alice`. If one person was split into two labels, merge them: `Speaker 3 = Speaker 1`. Lines you leave blank keep the generic label.
3. Re-render (takes a second or two, no GPU needed):

   ```bash
   python relabel.py ~/scribini/output/<name>/<name>.transcript.json --names ~/scribini/output/<name>/<name>.names.txt
   ```

This overwrites the `.txt` and `.docx` with the named versions. The generic version can always be regenerated from the saved `.transcript.json`.

## Hint list

Copy `hints.example.txt` to `hints.txt`. That file is your own plain-text list, and git ignores it, so your names and jargon stay private. Grow it over time:

```text
# a term to favor (names, jargon, acronyms)
Kubernetes
# a fix-up: misheard => correct (whole words, ignoring case)
kuber netties => Kubernetes
```

Terms nudge Whisper toward your spellings but do not guarantee them; fix-ups are applied afterwards as a safety net.

## Speed

One reference measurement: a 42-minute, 4-speaker recording took **4.1 minutes** in total on a desktop with an NVIDIA GeForce RTX 4060 (8 GB), including model loading. Second reference measurement: a 66-minute, 4-speaker recording took **6.1 minutes** in total on the same desktop system. Whisper `large-v3` on a CPU is typically slower than real time, so a long recording can take hours; `--model large-v3-turbo` and `--threads` are the two main dials. Measure on your own machine with a short clip first.

## Troubleshooting

- **`check_setup.py` says the diarization model folder is missing.** Download the model (see above) or pass `--diar-model PATH`.
- **Hugging Face keeps asking for a username/password during `git clone`.** Use the `hf auth login` and `hf download` commands above. Make sure you accepted the model's terms on the website with the same account, and that the token is of type *Read* (a fine-grained token must allow access to gated repos). Paste with Ctrl+Shift+V in most Linux terminals.
- **`open() got an unexpected keyword argument 'metadata_errors'`.** A mismatched audio-decoding library (PyAV). The current `transcribe.py` avoids it by passing audio samples to Whisper directly.
- **Warnings about TF32 or `std(): degrees of freedom`.** Harmless messages from pyannote.
- **Unexpected `Command 'python' not found` error on execution.** Reissue the `source ~/scribini/venv/bin/activate` command first.
- **Out of GPU memory.** Add `--low-vram`, or use a smaller `--model`.
- **A GPU is installed but not detected.** Run `nvidia-smi`. On WSL2, update the Windows NVIDIA driver. Then run `python check_setup.py`.

## Files

`transcribe.py` (the pipeline) · `relabel.py` (re-render with names) · `render.py` (text/Word output) · `speakers.py` (names file and speaker clips) · `check_setup.py` · `hints.example.txt` · `requirements.txt` · `requirements-nvidia.txt` · `LICENSE` · `CONTRIBUTING.md`

## Contributing

Bug reports and improvements are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Please never attach real recordings or transcripts to an issue.

## Credits and acknowledgements

Scribini was designed, specified, directed and tested by [verbage](https://github.com/verbage). The code was written with Claude (Anthropic). It is an independent project, not affiliated with or endorsed by Anthropic, OpenAI, SYSTRAN, or the pyannote team.

It stands on other people's work. These are installed or downloaded separately, and none of their code is included here:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT License) and OpenAI's Whisper speech-recognition models
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT License) and the [community-1 speaker model](https://huggingface.co/pyannote/speaker-diarization-community-1) (CC-BY-4.0), which each user downloads from Hugging Face after accepting its terms
- [FFmpeg](https://ffmpeg.org), [PyTorch](https://pytorch.org), [python-docx](https://github.com/python-openxml/python-docx) and [SoundFile](https://github.com/bastibe/python-soundfile)

## License

Scribini is released under the [BSD 3-Clause License](LICENSE). Copyright (c) 2026, verbage. The libraries and models above have their own licenses.
