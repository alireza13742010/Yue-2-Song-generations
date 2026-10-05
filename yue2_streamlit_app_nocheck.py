"""
Streamlit app — YuE2-3B music generation (lyrics + style prompt -> full song with vocals).

Model card: https://huggingface.co/m-a-p/YuE2-3B   (weights: CC BY-NC 4.0, non-commercial)

This is the "no checks" edition. It assumes you have ALREADY:
    * downloaded YuE2-3B and YuE2-Vae (optionally YuE2-Vae-legacy), and
    * installed the yue2 inference package (plus torch / transformers) in this environment.

There is no download step, no verification step and no package-install step, and nothing blocks the
Generate button. The app runs fully offline (HF_HUB_OFFLINE=1).

Where the weights are looked for (first match wins), for each of YuE2-3B / YuE2-Vae / YuE2-Vae-legacy:
    0. WEIGHTS_ROOT below (set to /media/avidmech/data/Yue music Generations/yue2_weights)
    1. the paths you set in MODEL_PATHS below (edit them if your weights live somewhere unusual)
    2. <folder of this script>/<RepoName>                       e.g. .\\YuE2-3B   (huggingface-cli --local-dir .)
    3. <folder of this script>/yue2_weights/                    (the old app's cache folder)
    4. your normal Hugging Face cache (~/.cache/huggingface/hub, or HF_HUB_CACHE / HF_HOME if set)
    5. $YUE2_WEIGHTS_DIR, if you set that environment variable
If nothing is found, the app falls back to the plain repo id (e.g. "m-a-p/YuE2-3B") and lets
YuE2Pipeline resolve it from the Hugging Face cache.

Run:
    streamlit run yue2_streamlit_app_nocheck.py
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "yue2_cache"   # torch / triton / inductor / temp files (kept next to this script)

# ---- optional: point straight at your weight folders ----------------------- #
# Example (Windows):  r"D:\models\YuE2-3B"      Example (Linux):  "/data/YuE2-3B"
MODEL_PATHS = {
    "m-a-p/YuE2-3B": None,
    "m-a-p/YuE2-Vae": None,
    "m-a-p/YuE2-Vae-legacy": None,
}

# Folder that holds all the downloaded models (checked first). Works for both layouts:
#   <WEIGHTS_ROOT>/models--m-a-p--YuE2-3B/snapshots/<hash>/...   (Hugging Face cache layout)
#   <WEIGHTS_ROOT>/YuE2-3B/...                                   (plain folders)
WEIGHTS_ROOT = "/media/avidmech/data/Yue music Generations/yue2_weights"

# Must run BEFORE torch / transformers are imported (they read these at import time).
os.environ["HF_HUB_OFFLINE"] = "1"        # never contact the Hub: no checks, no downloads
if WEIGHTS_ROOT and Path(WEIGHTS_ROOT).is_dir():
    os.environ["HF_HUB_CACHE"] = WEIGHTS_ROOT          # so plain repo ids also resolve from this folder
    os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_ROOT
os.environ["TRANSFORMERS_OFFLINE"] = "1"
_LOCAL_CACHES = {
    "HF_MODULES_CACHE": CACHE_DIR / "hf_modules",
    "TORCH_HOME": CACHE_DIR / "torch",
    "TRITON_CACHE_DIR": CACHE_DIR / "triton",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_DIR / "inductor",
    "TMPDIR": CACHE_DIR / "tmp",
}
for _k, _v in _LOCAL_CACHES.items():
    _v.mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_v)

import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import streamlit as st  # noqa: E402

st.set_page_config(page_title="YuE2-3B · Lyrics & Style → Song", page_icon="🎤", layout="centered")

try:
    import torch
except Exception:
    torch = None

ss = st.session_state

MAIN_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
VAE_LEGACY_REPO = "m-a-p/YuE2-Vae-legacy"
OUTPUT_DIR = APP_DIR / "output"

STYLE_PRESETS = {
    "🌙 Calm": "Ambient, soft piano, gentle strings, warm pads, slow tempo, peaceful, airy reverb",
    "🎉 Pop": "Pop, upbeat, catchy, bright female lead vocal, synths, danceable, energetic drums",
    "🎬 Cinematic": "Cinematic, orchestral, epic, sweeping strings, brass, dramatic, tension building",
    "🎸 Folk": "Acoustic folk, warm male vocal, fingerpicked guitar, intimate, storytelling",
    "🌆 Lo-fi": "Lo-fi chillhop, mellow beats, vinyl crackle, jazzy piano, relaxed",
}
COT_HELP = {
    "full": "full — melody + chord planning (default)",
    "melody": "melody — melody-only planning (recommended for covers)",
    "off": "off — no symbolic plan",
}


# --------------------------------------------------------------------------- #
# Locating the weights (no network, no verification)
# --------------------------------------------------------------------------- #
def _hf_default_hub_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _looks_like_model_dir(p):
    p = Path(p)
    return p.is_dir() and (any(p.glob("*.safetensors")) or (p / "config.json").exists())


def find_model(repo):
    """Return a local folder holding `repo`'s files, or None. Pure filesystem lookup."""
    name = repo.split("/")[-1]
    explicit = MODEL_PATHS.get(repo)
    if explicit and _looks_like_model_dir(explicit):
        return Path(explicit)

    roots = ([Path(WEIGHTS_ROOT)] if WEIGHTS_ROOT else []) + [APP_DIR, APP_DIR / "yue2_weights", _hf_default_hub_dir()]
    if os.environ.get("YUE2_WEIGHTS_DIR"):
        roots.append(Path(os.environ["YUE2_WEIGHTS_DIR"]))

    for root in roots:
        if _looks_like_model_dir(root / name):            # plain folder, e.g. ./YuE2-3B
            return root / name
        cache_repo = root / ("models--" + repo.replace("/", "--"))
        snaps = sorted((cache_repo / "snapshots").glob("*"), key=lambda s: s.stat().st_mtime, reverse=True) \
            if (cache_repo / "snapshots").is_dir() else []
        for snap in snaps:                                # Hugging Face cache layout
            if _looks_like_model_dir(snap):
                return snap
    return None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YuE2 (BF16, ≈11 GB VRAM)…")
def load_pipeline(vae_repo):
    from yue2 import YuE2Pipeline

    main_local, vae_local = find_model(MAIN_REPO), find_model(vae_repo)
    attempts = []
    if main_local or vae_local:   # 1) local folders where we found them
        kw = {"device": "cuda", "vae": str(vae_local or vae_repo)}
        attempts.append((str(main_local or MAIN_REPO), kw))
    kw = {"device": "cuda"}       # 2) plain repo ids, resolved from the HF cache (offline)
    if vae_repo != VAE_REPO:
        kw["vae"] = vae_repo
    attempts.append((MAIN_REPO, kw))

    errors = []
    for src, kwargs in attempts:
        try:
            return YuE2Pipeline.from_pretrained(src, **kwargs)
        except Exception as exc:
            errors.append(f"{src}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load YuE2Pipeline.\n" + "\n".join(errors))


def unload_pipeline():
    if ss.get("pipe_vae"):
        try:
            load_pipeline(ss["pipe_vae"]).close()
        except Exception:
            pass
    st.cache_resource.clear()
    ss.pop("pipe_vae", None)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def convert_audio(src, dst, bitrate="192k"):
    try:
        r = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-b:a", bitrate, str(dst)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not Path(dst).exists():
            return False, (r.stderr or "unknown ffmpeg error")[-300:]
        return True, ""
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"
    except Exception as e:
        return False, str(e)


def generate(style, lyrics, abc, cot, cfg, seed, fmt, vae_repo):
    pipe = load_pipeline(vae_repo)
    ss["pipe_vae"] = vae_repo
    kwargs = dict(style=style.strip(), lyrics=lyrics, cot=cot, seed=int(seed))
    if abc.strip():
        kwargs["abc"] = abc
    if cfg:
        kwargs["cfg_scale"] = float(cfg)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flac, art = OUTPUT_DIR / f"song_{stamp}.flac", OUTPUT_DIR / f"song_{stamp}"

    t0 = time.time()
    song = pipe(**kwargs)
    song.save(str(flac))
    song.save_artifacts(str(art))  # ABC score, tokens, latents, audio, settings
    secs = time.time() - t0

    path, msg = flac, "Song generated (FLAC)."
    if fmt != "flac":
        dst = flac.with_suffix(f".{fmt}")
        ok, err = convert_audio(flac, dst)
        if ok:
            path, msg = dst, f"Song generated ({fmt.upper()})."
        else:
            msg = f"{fmt.upper()} conversion failed ({err}) — kept the FLAC."
    score = art / "score.abc"
    return {"path": str(path), "msg": msg, "seed": int(seed), "secs": secs, "style": style,
            "abc": score.read_text(encoding="utf-8") if score.exists() else ""}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(160deg,#1b1035 0%,#2b1055 45%,#120a2e 100%); color:#f5f3ff; }
    h1,h2,h3,p,label,.stCaption { color:#f5f3ff !important; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input { background:#e8e8e8 !important; color:#111 !important; border-radius:10px; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:#e8e8e8 !important; color:#111 !important; }
    .stButton > button { background:linear-gradient(90deg,#7c3aed,#a855f7); color:#fff; border:none;
                         border-radius:10px; font-weight:600; }
    .stButton > button:hover { background:linear-gradient(90deg,#6d28d9,#9333ea); color:#fff; }
    div[data-testid="stExpander"] { background:rgba(255,255,255,.04); border-radius:10px;
                                    border:1px solid rgba(255,255,255,.12); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎤 YuE2-3B — Lyrics & Style → Song")
st.caption("Local generation with m-a-p/YuE2-3B. Weights are CC BY-NC 4.0 (non-commercial use).")

for k, val in {"history": [], "style_input": "", "lyrics_input": "", "abc_text": "",
               "seed_mode": "Random", "seed_value": 42}.items():
    ss.setdefault(k, val)
if "abc_pending" in ss:  # set by the "load score" button; must be applied before the widget exists
    ss["abc_text"] = ss.pop("abc_pending")

# ---- sidebar (information only, nothing here blocks generation) ------------- #
with st.sidebar:
    st.header("Setup")
    with_legacy = st.checkbox("Use YuE2-Vae-legacy as an option", value=False,
                              help="Benchmark decoder (higher benchmark musicality). YuE2-Vae is the default "
                                   "and gives better perceptual audio quality.")
    st.markdown("**📍 Weights found**")
    for repo in [MAIN_REPO, VAE_REPO] + ([VAE_LEGACY_REPO] if with_legacy else []):
        loc = find_model(repo)
        st.caption(f"{repo.split('/')[-1]}: " + (f"`{loc}`" if loc else "not found on disk — will try the repo id"))

    if torch is not None and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        st.caption(f"🎮 {torch.cuda.get_device_name(0)} · {vram:.0f} GB VRAM")
        if vram < 23:
            st.warning("The model card targets a 24 GB GPU (peak ≈ 11–14 GB).")
    else:
        st.caption("🎮 No CUDA GPU visible to torch (generation will likely fail).")

    if st.button("Unload model / free VRAM"):
        unload_pipeline()
        st.rerun()

# ---- generate --------------------------------------------------------------- #
st.subheader("Generate")

vae_repo = VAE_REPO
if with_legacy:
    vae_repo = st.radio("Decoder (VAE)", [VAE_REPO, VAE_LEGACY_REPO], horizontal=True)

st.caption("Style presets")
cols = st.columns(len(STYLE_PRESETS))
for col, (label, tags) in zip(cols, STYLE_PRESETS.items()):
    if col.button(label):
        ss["style_input"] = tags

if st.button("📥 Load the official example (今晚不眠) — shows the expected lyrics format"):
    snap = find_model(MAIN_REPO)
    try:
        demo = json.loads((snap / "examples" / "tonight-awake.json").read_text(encoding="utf-8"))
        ss["style_input"], ss["lyrics_input"] = demo["style"], demo["lyrics"]
        ss["seed_mode"], ss["seed_value"] = "Fixed", int(demo.get("seed", 42))
    except Exception as exc:
        st.warning(f"Couldn't load the example: {exc}")

style_input = st.text_input("Style prompt", key="style_input",
                            placeholder="Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums")
lyrics_input = st.text_area("Lyrics", key="lyrics_input", height=220,
                            placeholder="Paste lyrics organised into sections. Use the official example above to see the format.")

with st.expander("🎼 Score (ABC) — for covers and editing", expanded=bool(ss["abc_text"])):
    st.caption("Optional. Cover: paste a melody ABC (no chord symbols) and choose the *melody* planning mode. "
               "Edit: generate once, load the returned score here, change it, and generate again with the same seed.")
    up = st.file_uploader("…or upload an .abc file", type=["abc", "txt"])
    if up is not None and st.button("Use uploaded file"):
        ss["abc_pending"] = up.getvalue().decode("utf-8", errors="replace")
        st.rerun()
    abc_input = st.text_area("ABC score", key="abc_text", height=160)

a, b = st.columns(2)
cot = a.selectbox("Planning mode (cot)", list(COT_HELP), format_func=COT_HELP.get, index=0)
fmt = b.radio("Output format", ["flac", "mp3", "wav"], horizontal=True,
              help="FLAC is what YuE2 writes natively; MP3/WAV are converted with ffmpeg afterwards.")

with st.expander("⚙️ Advanced"):
    use_cfg = st.checkbox("Custom text guidance (cfg_scale)", value=False,
                          help="Defaults: 1.0 for full/melody, 1.01 for off. The model card suggests trying 1.2.")
    cfg = st.slider("cfg_scale", 1.0, 2.0, 1.2, 0.05, disabled=not use_cfg)
    st.radio("Seed", ["Random", "Fixed"], horizontal=True, key="seed_mode")
    st.number_input("Seed value", 0, 2**53 - 1, step=1, key="seed_value", disabled=ss["seed_mode"] == "Random")

st.caption("Song length follows your lyrics/structure. Stage progress and tokens/s print in the terminal.")

if st.button("🎧 Generate song", type="primary"):
    if not style_input.strip() or not lyrics_input.strip():
        st.warning("Enter both a style prompt and lyrics.")
    else:
        seed = random.randint(0, 2**31 - 1) if ss["seed_mode"] == "Random" else int(ss["seed_value"])
        try:
            with st.spinner("Generating… this takes about a minute for a 3–4 minute song on a 4090."):
                res = generate(style_input, lyrics_input, abc_input, cot,
                               cfg if use_cfg else None, seed, fmt, vae_repo)
            ss["last"] = res
            ss["history"].insert(0, res)
        except Exception as exc:
            st.error("Generation failed. See the traceback below.")
            if "out of memory" in str(exc).lower():
                st.warning("Out of GPU memory — close other GPU programs or use shorter lyrics.")
            st.exception(exc)

last = ss.get("last")
if last:
    st.success(f"{last['msg']}  ⏱ {last['secs']:.0f}s • seed {last['seed']}")
    st.audio(last["path"])
    with open(last["path"], "rb") as f:
        st.download_button("⬇️ Download", f, file_name=Path(last["path"]).name)
    if last["abc"]:
        with st.expander("Generated score (ABC)"):
            st.code(last["abc"], language=None)
            if st.button("✏️ Load this score into the editor"):
                ss["abc_pending"] = last["abc"]
                st.rerun()

if len(ss["history"]) > 1:
    st.subheader("📜 Earlier tracks this session")
    for i, item in enumerate(ss["history"][1:], start=2):
        st.write(f"**{i}.** `{item['style'][:80]}` — seed {item['seed']}")
        st.audio(item["path"])
