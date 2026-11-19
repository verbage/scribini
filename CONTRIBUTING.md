# Contributing to Scribini

Thanks for your interest. Scribini is a small project, and bug reports, fixes and ideas are all welcome.

## Reporting a problem

Open an issue and include:

- your operating system (and whether you use WSL2)
- your GPU model, or say you are running CPU-only
- the exact command you ran
- the full error message or output
- the output of `python check_setup.py`

**Please do not attach real recordings, transcripts or speaker clips.** They may contain private conversations. If you need to show a problem, use a short non-sensitive sample, or describe it.

## Suggesting a change

For anything larger than a small fix, please open an issue first so we can agree on the approach before you spend time on it.

## Sending a pull request

- Keep it small and focused: one change per pull request.
- Keep the code simple and readable; the scripts are meant to be understood by non-experts.
- Before submitting, make sure the scripts still compile:

  ```bash
  python -m py_compile transcribe.py relabel.py render.py speakers.py check_setup.py
  ```

- If you change behavior or add an option, update `README.md`.
- Do not commit models, recordings, transcripts, tokens or your personal `hints.txt`.

## License of contributions

Scribini is released under the BSD 3-Clause License (see `LICENSE`). By submitting a contribution you agree that it is licensed under the same terms.
