# Patch: add a YouTube link box to `yue2_streamlit_app.py`

It adds a **🔗 Reference video / audio → cover score** panel in step ④. The flow:

1. Paste a YouTube link (or upload an audio file instead).
2. **Get audio** saves it to `reference_audio/` (needs `yt-dlp`).
3. **Transcribe to ABC** runs SheetSage2 (the separate environment from the YuE covers doc) and drops the melody into the **Score (ABC)** box.
4. Paste lyrics, choose planning mode **melody**, and generate.

## 0. Install the extras (once)

```bash
# in the Streamlit app's environment
python -m pip install yt-dlp
# ffmpeg must be on PATH (yt-dlp uses it to make WAV)

# SheetSage2: separate environment, set up exactly as in the YuE covers doc
python3.11 -m venv .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 --local-dir models/SheetSage2
```
Then install SheetSage2's torch and `requirements.txt` as the covers doc describes.

## 1. Paste these functions

Place them **after the `audio_seconds()` function** and before the `# Generation` section.

```python
# --------------------------------------------------------------------------- #
# Reference video (YouTube) / audio -> ABC melody via SheetSage2
# --------------------------------------------------------------------------- #
REF_DIR = APP_DIR / "reference_audio"
YT_RE = re.compile(r"^https?://((www|m|music)\.)?(youtube\.com|youtu\.be)/", re.I)
AUDIO_TYPES = ["wav", "mp3", "flac", "m4a", "ogg"]


def download_youtube_audio(url):
    """Download the audio of a YouTube link as WAV into reference_audio/. Returns (path, error)."""
    url = url.strip()
    if not YT_RE.match(url):
        return None, "That doesn't look like a YouTube link."
    if importlib.util.find_spec("yt_dlp") is None:
        return None, "yt-dlp is not installed. Run: python -m pip install yt-dlp"
    REF_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-simulate",
           "-x", "--audio-format", "wav",
           "-o", str(REF_DIR / "%(id)s.%(ext)s"),
           "--print", "after_move:filepath", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        return None, str(e)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    path = Path(lines[-1]) if lines else None
    if r.returncode != 0 or path is None or not path.exists():
        return None, (r.stderr or r.stdout or "yt-dlp failed")[-400:]
    return path, ""


def transcribe_to_abc(audio_path, sheetsage_python, sheetsage_dir):
    """Run SheetSage2 in its own environment -> melody-only ABC. Returns (abc_text, info_or_error)."""
    py, d = Path(sheetsage_python), Path(sheetsage_dir)
    py = py if py.is_absolute() else APP_DIR / py      # no .resolve(): that would break venv symlinks
    d = d if d.is_absolute() else APP_DIR / d
    infer = d / "infer.py"
    if not py.exists() or not infer.exists():
        return None, f"Check the SheetSage2 paths: {py}  /  {infer}"
    out = OUTPUT_DIR / f"cover-score_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), str(infer), str(audio_path), "--output", str(out), "--melody-only"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(APP_DIR))
    except Exception as e:
        return None, str(e)
    score = out / "score.abc"
    if r.returncode != 0 or not score.exists():
        return None, (r.stderr or r.stdout or "SheetSage2 failed")[-600:]
    return score.read_text(encoding="utf-8"), (r.stderr or "")[-300:]
```

## 2. Paste the UI block

Place it in step ④, **immediately above** this existing line:

```python
with st.expander("🎼 Score (ABC) — for covers and editing", expanded=bool(ss["abc_text"])):
```

```python
for _k, _v in {"ref_url": "", "ref_audio": "",
               "ss2_py": ".venv-sheetsage2/bin/python", "ss2_dir": "models/SheetSage2"}.items():
    ss.setdefault(_k, _v)

with st.expander("🔗 Reference video / audio → cover score", expanded=bool(ss["ref_url"] or ss["ref_audio"])):
    st.caption("For covers: paste the YouTube link of the song you want to cover (or upload the audio). "
               "The app gets the audio, SheetSage2 transcribes the melody to ABC, and it lands in the "
               "Score box below. Only use recordings you have the right to use.")
    st.text_input("YouTube link", key="ref_url", placeholder="https://www.youtube.com/watch?v=...")
    up_audio = st.file_uploader("…or upload the audio yourself", type=AUDIO_TYPES, key="ref_upload")
    st.text_input("SheetSage2 Python", key="ss2_py", help="Python of the separate SheetSage2 environment")
    st.text_input("SheetSage2 folder", key="ss2_dir", help="Folder that contains infer.py")

    b1, b2 = st.columns(2)
    if b1.button("⬇️ Get audio"):
        if up_audio is not None:
            REF_DIR.mkdir(exist_ok=True)
            dst = REF_DIR / Path(up_audio.name).name
            dst.write_bytes(up_audio.getvalue())
            ss["ref_audio"] = str(dst)
        elif ss["ref_url"].strip():
            with st.spinner("Downloading audio…"):
                p, err = download_youtube_audio(ss["ref_url"])
            if p:
                ss["ref_audio"] = str(p)
            else:
                st.error(err)
        else:
            st.warning("Paste a YouTube link or upload an audio file first.")

    if ss["ref_audio"] and Path(ss["ref_audio"]).exists():
        st.audio(ss["ref_audio"])
        st.caption(f"Reference audio: `{Path(ss['ref_audio']).name}`")

    if b2.button("🎼 Transcribe to ABC", disabled=not ss["ref_audio"]):
        unload_pipeline()   # SheetSage2 and YuE2 must share the GPU one after the other
        with st.spinner("Transcribing the melody with SheetSage2…"):
            abc_text, info = transcribe_to_abc(ss["ref_audio"], ss["ss2_py"], ss["ss2_dir"])
        if abc_text:
            ss["abc_pending"] = abc_text   # applied to the Score box on the next run
            st.rerun()
        else:
            st.error(info)
    st.caption("After transcribing: check the score against the original (notes, meter, section order), "
               "add matching lyrics, and set Planning mode to **melody**.")
```

## 3. Optional: one thing to know about your existing code

Your script puts the `Detected pipeline parameters` expander **inside** the `Duration & steps` expander.
Recent Streamlit versions raise *"Expanders may not be nested inside other expanders."* If you see that error,
change the inner one to `st.checkbox("Show detected pipeline parameters")` followed by `st.json(...)`.

## Notes

- A YouTube **429 / rate-limit** error means YouTube is blocking the download for now. Wait a bit, or download the
  audio another way and use the upload box.
- Transcription errors carry into the cover, so always listen to the original while reviewing the ABC.
- Run order matters for VRAM: the button unloads the YuE2 model before SheetSage2 runs. YuE2 reloads on the next generate.
