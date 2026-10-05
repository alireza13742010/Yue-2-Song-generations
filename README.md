# 🎤 YuE2-3B Streamlit App — Lyrics & Style → Song

A local Streamlit app that turns **lyrics + a style prompt** into a full song with vocals using
[`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B). It can also make **covers** (from an ABC melody)
and **edit** a generated score.

> **License:** the model weights are **CC BY-NC 4.0, non-commercial use only.**

---

## What the app does

The app is a four-step pipeline. Each step unlocks the next one.

| Step | What happens |
|------|--------------|
| **① Download** | Downloads `YuE2-3B` and `YuE2-Vae` (optionally `YuE2-Vae-legacy`) with resume support. |
| **② Verify** | Checks every file against the Hub's file list (size), validates `.safetensors` headers, with an optional SHA-256 deep check. |
| **③ Package** | Installs the `yue2_infer` wheel that ships inside the YuE2-3B repo. |
| **④ Generate** | Create, Cover, or Edit a song through `YuE2Pipeline`, with length controls. |

---

## Requirements

- **OS:** Linux
- **Python:** 3.10 or newer (a conda environment is recommended)
- **GPU:** NVIDIA, 24 GB VRAM class, with BF16 support (peak usage is about 11–14 GB)
- **RAM:** about 24 GB free host memory
- **Disk:** enough for the model weights (several GB, stored next to the script)
- **ffmpeg / ffprobe** on your `PATH`. These are needed for MP3/WAV output, the hard duration cap, and showing the audio length. FLAC output works without them.
- *Optional, for the YouTube link panel:* `yt-dlp` and a separate SheetSage2 environment (see "Adding a YouTube link" below).

---

## Installation

1. Create and activate a dedicated environment:

   ```bash
   conda create -n yue2 python=3.10 -y
   conda activate yue2
   ```

2. Install the two prerequisites. The app downloads everything else itself:

   ```bash
   python -m pip install huggingface-hub==0.36.2 streamlit
   ```

3. Put `yue2_streamlit_app.py` in an empty folder of your choice.

> Always use `python -m pip` rather than bare `pip`, so packages land in the environment you activated.

---

## Running

```bash
streamlit run yue2_streamlit_app.py
```

Then work through the page from top to bottom:

1. Click **⬇️ Download / resume**. If it is interrupted, press it again and it continues where it stopped.
2. Check that the **verify** banner turns green. Use **🔍 Re-check** or **🧪 Deep check (SHA-256)** if you want extra certainty.
3. Click **📦 Install yue2_infer-…whl**. This pulls in `torch`, `transformers` and the `yue2` code, which can take a few minutes.
4. **Stop Streamlit (Ctrl+C) and run it again** after the install, so the new `torch` and `transformers` versions load cleanly.
5. When the weights are verified, the package is installed and a CUDA GPU is visible, the **④ Generate** section unlocks.

---

## Where files go

Everything lives in the folder that contains the script. Nothing is written to `~/.cache`, because the app overrides `HF_HOME`, `HF_HUB_CACHE`, `XDG_CACHE_HOME` and similar variables before any library loads.

```
yue2_streamlit_app.py
yue2_infer-<version>.whl   inference wheel, copied here from the model repo
yue2_weights/              YuE2-3B, YuE2-Vae (+ legacy), Hugging Face cache layout
yue2_cache/                xet / torch / triton / inductor / HF-module caches, temp files
output/                    generated songs, ABC scores, artifacts
reference_audio/           audio fetched from a YouTube link or uploaded (needs the YouTube patch)
```

The one exception is that Python packages (`torch`, `transformers`, `yue2`) install into your environment's `site-packages`. pip cannot put them in this folder.

The sidebar shows the size of each folder and your GPU. **Unload model / free VRAM** releases GPU memory.

---

## Generating a song

### Fields

- **Style prompt:** comma-separated descriptors. Presets are provided (Calm, Pop, Cinematic, Folk, Lo-fi).
  Example: `Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums`
- **Lyrics:** organised into section tags. Click **📥 Load the official example** to see the expected format.

  ```
  [Verse]
  First line of the verse
  Second line of the verse

  [Chorus]
  First line of the chorus
  ...
  ```

- **Planning mode (`cot`):**
  - `full`: melody + chord planning (default)
  - `melody`: melody-only planning, **recommended for covers**
  - `off`: no symbolic plan
- **Output format:** `flac` (native), or `mp3` / `wav` (converted with ffmpeg).
- **Decoder (VAE):** shown only if you also downloaded `YuE2-Vae-legacy`. `YuE2-Vae` is the default and gives better perceptual audio quality. The legacy one scores higher on benchmark musicality.

### Advanced

- **cfg_scale:** the default is 1.0 for `full` and `melody`, and 1.01 for `off`. The model card suggests trying 1.2.
- **Seed:** Random, or Fixed. Reuse a fixed seed to reproduce or edit a result.

### Controlling length

YuE2's documented API has no duration argument. **Song length follows the lyrics and the score.** The app offers these controls:

| Control | Effect |
|---------|--------|
| **Lyric sections to use** | Sings only the first N sections. Shorter song and less compute. |
| **Hard cap (seconds)** | Cuts the finished song with a 2 s fade-out. The full song is still generated first, so no compute is saved. |
| **Native duration / steps** | Appear only if your installed `yue2` version really has such a parameter. |
| **Extra pipeline kwargs (JSON)** | Free-form passthrough, e.g. `{"num_steps": 32}`. Names are validated against the installed package. |

If you supply an ABC score, **keep all lyric sections**. Cutting sections may no longer match the score.

---

## Covers and editing (ABC scores)

Open the **🎼 Score (ABC)** expander.

**Cover**
1. Transcribe the source recording to melody-only ABC (for example with SheetSage2) and paste it, or upload an `.abc` file.
2. Paste matching lyrics. Section tags and lyric order must line up with the score.
3. Set the planning mode to **`melody`**.
4. Generate.

**Edit**
1. Generate once. The returned score appears under **Generated score (ABC)**.
2. Click **✏️ Load this score into the editor**, then change the notes.
3. Keep the **same fixed seed** and generate again.

### YouTube link: https://youtu.be/vUP56XZ7tt0

Apply `yue2_youtube_patch.md` to get a **🔗 Reference video / audio → cover score** panel above the Score box.

1. Paste the **YouTube link** (or upload the audio file yourself).
2. Click **⬇️ Get audio**. The audio is saved to `reference_audio/` and you can play it in the page.
3. Click **🎼 Transcribe to ABC**. SheetSage2 turns the melody into ABC and loads it into the Score box.
4. Review the score against the original (notes, meter, section order), paste matching lyrics, set Planning mode to **melody**, and generate.

Extra requirements for this panel:

- `python -m pip install yt-dlp` in the app's environment, plus ffmpeg on `PATH`
- SheetSage2 in its **own environment** (Python 3.10/3.11, FFmpeg 6.1), as in the [YuE covers doc](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md). Enter its Python and folder paths in the panel.
- The YuE2 model is unloaded while SheetSage2 runs because they share one GPU. It reloads on the next generate.

If YouTube blocks the download (HTTP 429), wait a while or download the audio another way and use the upload box. Only use recordings you have the right to use.

### Using `cover-request.json` + `score.abc` files from earlier

The app does not read the request JSON directly. Copy the fields across:

| In the JSON file | Put it in the app |
|------------------|-------------------|
| `style` | **Style prompt** |
| `lyrics` | **Lyrics** |
| `cot` | **Planning mode** |
| `seed` | **Advanced → Seed → Fixed** |
| `score.abc` | **Score (ABC)** (paste or upload) |

---

## Output

Each run writes to `output/`:

- `song_<timestamp>.flac`: the generated song
- `song_<timestamp>_cap.<fmt>` or `song_<timestamp>.<fmt>`: the converted or capped version, if requested
- `song_<timestamp>/`: artifacts including `score.abc`, tokens, latents, audio and settings

The page shows the audio player, generation time, audio length, the seed, a download button, and a history of earlier tracks from the session.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| *"Hugging Face cache points outside the app folder"* | Stop Streamlit fully and relaunch it, so the script can set the cache paths first. |
| *Generate section stays locked* | One of three is missing: verified weights, the `yue2` package, or a visible CUDA GPU. The caption under step ③ tells you which. |
| *Download fails or stalls* | Press **Download / resume** again. Check your connection. |
| *A file is BAD SIZE / CORRUPT / MISSING* | Press **Download / resume** to re-fetch it, then **Re-check**. |
| *Out of GPU memory* | Close other GPU programs, use shorter lyrics or fewer sections, or press **Unload model / free VRAM**. |
| *"Post-processing failed … ffmpeg not found"* | Install ffmpeg and make sure it is on `PATH`. The full FLAC is kept meanwhile. |
| *"Not accepted by `YuE2Pipeline.__call__`"* | A JSON kwarg name does not exist in your installed `yue2` version. Check **Detected pipeline parameters**. |
| *`No module named 'mpmath'`, `networkx`, `filelock`, etc.* | A dependency is missing from the environment, often after an interrupted install. Install just that package with `python -m pip install <name>` inside the same environment, then run `python -m pip check`. Avoid upgrading `torch`, `transformers` or `sympy` unless needed. |
| *`mpmath` version conflict with `sympy`* | Install `"mpmath>=1.1.0,<1.4"` so it matches the `sympy` version your `torch` pins. |

After fixing any import-level problem, **restart Streamlit** (or the Jupyter kernel). Python caches failed imports.

---

## Tips for good results

- Match the **number of syllables** per line to the melody when covering or translating lyrics.
- Use clear section tags (`[Verse]`, `[Chorus]`, `[Bridge]`) and keep their order the same as in any ABC score.
- Put the **genre, vocal type, main instruments, mood and BPM** in the style prompt.
- Try a few **seeds** with otherwise identical settings. Results vary noticeably.
- For a ~4 minute song, write enough sections to fill it. A 3–4 minute song takes about a minute on an RTX 4090.

---

## Credits

Model: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) · Decoder: `m-a-p/YuE2-Vae` · Code reference: the
[YuE repository](https://github.com/multimodal-art-projection/YuE).
