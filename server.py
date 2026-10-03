"""OmniVoice Studio — local web UI for OmniVoice TTS (voice clone / voice design / auto voice). Developed by Kenzy.

Run:  .venv/Scripts/python.exe server.py                 -> http://127.0.0.1:8765
      .venv/Scripts/python.exe server.py --no-preload --port 9000

Layout:  data/voices/<id>/{meta.json, ref.wav, styles/<sid>.wav, prompt-<model>-<style>.pt}  reference voices (+ styles)
         data/outputs/<id>.{wav,json}          generated takes (history); json carries spoken_text, qa, alignment
         data/projects/<id>/project.json       script-mode projects; export/ holds per-line wav+mp3+json, full.wav, timeline.json
         data/pronunciation.json               pronunciation rules applied before synthesis
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from num2words import num2words
except ImportError:  # optional: numbers are then read as the model sees them
    num2words = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("studio")

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
VOICES_DIR = DATA / "voices"
OUT_DIR = DATA / "outputs"
PROJECTS_DIR = DATA / "projects"
PRON_FILE = DATA / "pronunciation.json"
for _d in (VOICES_DIR, OUT_DIR, PROJECTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SR = 24000
MAX_REF_SECONDS = 20.0     # OmniVoice warns above 20 s and recommends 3–10 s references
TARGET_REF_SECONDS = 10.0  # long uploads are cut at a pause near this length
MIN_REF_SECONDS = 1.0
MAX_WORDS_PER_SECOND = 6.0  # a typed transcript denser than this cannot match the audio
NOISY_SNR_DB = 25.0         # below this the reference likely carries music / room noise

MODELS = {
    "base": {"repo": "k2-fsa/OmniVoice", "label": "OmniVoice gốc",
             "note": "Apache-2.0 · dùng thương mại được · khuyên dùng"},
    "vi": {"repo": "splendor1811/omnivoice-vietnamese", "label": "Fine-tune tiếng Việt (thử nghiệm)",
           "note": "Khi đo hay nuốt chữ · dữ liệu CC-BY-NC-SA, chỉ dùng phi thương mại"},
}
LANGUAGES = [
    ("vi", "Tiếng Việt"), ("en", "English"), ("ja", "日本語"), ("zh", "中文"), ("ko", "한국어"),
    ("fr", "Français"), ("de", "Deutsch"), ("es", "Español"), ("th", "ไทย"), ("id", "Bahasa Indonesia"),
    ("auto", "Để model tự đoán"),
]
COLORS = ["#FFD60A", "#FF4D8D", "#4CC9F0", "#06D6A0", "#9B5DE5", "#FF8C42"]
DEFAULT_RULES = [  # the recap channel's usual "say" spellings
    {"from": "Enel", "to": "Ê-nen"}, {"from": "vcl", "to": "vê-cờ-lờ"}, {"from": "đcm", "to": "đê-cờ-mờ"},
    {"from": "vl", "to": "vê-lờ"}, {"from": "đm", "to": "đê-mờ"}, {"from": "meme", "to": "mim"},
    {"from": "cute", "to": "kiu"}, {"from": "ok", "to": "ô-kê"},
]


# ----------------------------------------------------------------------------- text: pronunciation + numbers

def load_rules() -> list[dict]:
    if PRON_FILE.exists():
        try:
            return json.loads(PRON_FILE.read_text(encoding="utf-8"))
        except Exception:
            log.warning("pronunciation.json unreadable, using defaults")
    save_rules(DEFAULT_RULES)
    return [dict(r) for r in DEFAULT_RULES]


def save_rules(rules: list[dict]) -> None:
    PRON_FILE.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+|\d+)([.,]\d+)?(\s*%)?(?![\w])")


def _number_to_words(m: re.Match) -> str:
    whole, frac, pct = m.group(1), m.group(2), m.group(3)
    try:
        words = num2words(int(re.sub(r"[.,]", "", whole)), lang="vi")
        if frac:
            digits = frac[1:]
            words += " phẩy " + (num2words(int(digits), lang="vi") if len(digits) <= 2
                                 else " ".join(num2words(int(d), lang="vi") for d in digits))
    except Exception:
        return m.group(0)
    return words + (" phần trăm" if pct else "")


def spoken_form(text: str, rules: list[dict], read_numbers: bool = True) -> str:
    """What the model is asked to read: *emphasis* markers dropped, pronunciation rules applied, digits spelled out."""
    s = re.sub(r"\*+", "", text)
    for r in rules:
        if not r.get("enabled", True) or not r.get("from"):
            continue
        pat = r["from"] if r.get("regex") else re.escape(r["from"])
        if r.get("whole", True):
            pat = rf"(?<![\w]){pat}(?![\w])"
        try:
            s = re.sub(pat, r.get("to", ""), s, flags=re.IGNORECASE)
        except re.error:
            continue
    if read_numbers and num2words:
        s = NUM_RE.sub(_number_to_words, s)
    return re.sub(r"[ \t]+", " ", s).strip()


# ----------------------------------------------------------------------------- text: QA against Whisper

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").replace("đ", "d")


def norm_tokens(text: str) -> list[tuple[str, int]]:
    """(syllable, index of the whitespace word it came from); hyphenated spellings split into syllables and digits
    are spelled out ("100" → một trăm) so a number compares equal however either side wrote it."""
    out = []
    for i, w in enumerate(text.split()):
        if num2words:
            w = NUM_RE.sub(_number_to_words, w)
        for part in re.split(r"[-–/\s]+", w):
            t = re.sub(r"[^\w]", "", unicodedata.normalize("NFC", part.lower()))
            if t:
                out.append((t, i))
    return out


def phon_key(t: str) -> str:
    """Tolerant key for northern-Vietnamese homophones and Whisper spellings (sét/xét, chả/trả, ổng/ông)."""
    t = _strip_accents(t)
    for a, b in (("ngh", "ng"), ("gh", "g"), ("tr", "ch"), ("gi", "d"), ("r", "d"), ("x", "s"), ("qu", "kw"), ("k", "c"), ("q", "c")):
        if t.startswith(a):
            t = b + t[len(a):]
            break
    return t


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def qa_compare(spoken: str, heard: str) -> dict:
    """Word-level verdict: ok / near (homophone or spelling variant) / wrong / missing, plus extra words heard."""
    E, H = norm_tokens(spoken), norm_tokens(heard)  # digits are spelled out on both sides inside norm_tokens
    A, B = [t for t, _ in E], [t for t, _ in H]
    status, extra = ["missing"] * len(A), []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                status[k] = "ok"
        elif tag == "replace":
            ea, hb = A[i1:i2], B[j1:j2]
            # one letter off in the tolerant key (Ê-nen/Inen, run/Jun) is a Whisper spelling, not a misread → "near"
            close = lambda x, y: edit_distance(x, y) <= (1 if min(len(x), len(y)) <= 4 else 2)
            if len(ea) == len(hb):
                for k in range(len(ea)):
                    status[i1 + k] = "near" if close(phon_key(ea[k]), phon_key(hb[k])) else "wrong"
            elif close("".join(map(phon_key, ea)), "".join(map(phon_key, hb))):
                for k in range(i1, i2):
                    status[k] = "near"
            else:
                for k in range(i1, i2):
                    status[k] = "wrong"
        elif tag == "insert":
            extra.extend(B[j1:j2])
    rank = {"ok": 0, "near": 1, "wrong": 2, "missing": 3}
    words = spoken.split()
    word_status = ["ok"] * len(words)
    for (_, i), s in zip(E, status):
        if rank[s] > rank[word_status[i]]:
            word_status[i] = s
    a, b = " ".join(A), " ".join(B)
    return {"heard": heard.strip(),
            "errors": sum(s in ("wrong", "missing") for s in word_status),
            "near": sum(s == "near" for s in word_status),
            "cer": round(edit_distance(a, b) / max(1, len(a)), 3),
            "wer": round(edit_distance(A, B) / max(1, len(A)), 3),
            "words": [{"w": w, "s": s} for w, s in zip(words, word_status)], "extra": extra}


def align_words(spoken: str, chunks: list[dict], duration: float) -> dict:
    """Whisper word timestamps mapped onto the known text (ElevenLabs character-alignment format, like
    recap-shorts/tools/align_words.py), with unmatched words interpolated between known neighbours."""
    heard = []
    for c in chunks:
        w = c["text"].strip()
        if not w:
            continue
        t0, t1 = c["timestamp"]
        heard.append((w, float(t0 or 0), float(t1) if t1 is not None else float(t0 or 0) + 0.2))
    toks = spoken.split()
    norm1 = lambda w: re.sub(r"[^\w]", "", unicodedata.normalize("NFC", w.lower()))
    A, B = [norm1(t) for t in toks], [norm1(h[0]) for h in heard]
    times: list = [None] * len(toks)
    matched = 0
    for blk in difflib.SequenceMatcher(None, A, B, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = (heard[blk.b + k][1], heard[blk.b + k][2])
            matched += 1
    i = 0
    while i < len(toks):
        if times[i] is None:
            j = i
            while j < len(toks) and times[j] is None:
                j += 1
            lo = times[i - 1][1] if i > 0 else 0.0
            hi = times[j][0] if j < len(toks) else duration
            step = max(hi - lo, 0.01) / (j - i)
            for k in range(i, j):
                times[k] = (lo + (k - i) * step, lo + (k - i + 1) * step)
            i = j
        else:
            i += 1
    chars, st, en = [], [], []
    for k, (tok, (a, b)) in enumerate(zip(toks, times)):
        if k:
            chars.append(" "); st.append(a); en.append(a)
        n = len(tok)
        for c_i, ch in enumerate(tok):
            chars.append(ch); st.append(round(a + (b - a) * c_i / n, 4)); en.append(round(a + (b - a) * (c_i + 1) / n, 4))
    return {"words": [{"w": t, "start": round(a, 3), "end": round(b, 3)} for t, (a, b) in zip(toks, times)],
            "alignment": {"characters": chars, "character_start_times_seconds": st, "character_end_times_seconds": en},
            "matched": f"{matched}/{len(toks)}", "duration": round(duration, 3)}


# ----------------------------------------------------------------------------- audio helpers

def ffmpeg_to_wav(src: Path, dst: Path, max_seconds: float | None = None) -> float:
    """Convert any audio/video file to 24 kHz mono 16-bit WAV. Returns the duration in seconds."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise HTTPException(400, f"ffmpeg không đọc được file: {proc.stderr.strip()[:300]}")
    return round(sf.info(str(dst)).duration, 2)


def _frame_db(x: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    hop = int(sr * hop_s)
    n = len(x) // hop
    if n == 0:
        return np.array([-80.0])
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-6)


def find_pause_cut(wav: Path, lo: float = 4.0, hi: float = 15.0, target: float = TARGET_REF_SECONDS,
                   min_pause: float = 0.25) -> Optional[float]:
    """Time (s) inside a pause between `lo` and `hi` seconds, closest to `target`; None if there is no pause."""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    db = _frame_db(x, sr)
    quiet = db < max(db.max() - 35, -50)
    pauses, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.01 >= min_pause:
                pauses.append(((start + i) / 2 * 0.01, (i - start) * 0.01))
            start = None
    candidates = [(abs(t - target) - 0.5 * length, t) for t, length in pauses if lo <= t <= hi]
    return min(candidates)[1] if candidates else None


def trim_and_measure(wav: Path) -> tuple[float, float]:
    """Trim leading/trailing silence in place (0.1 s / 0.15 s padding). Returns (duration, SNR estimate in dB)."""
    x, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    db = _frame_db(x, sr)
    floor, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    snr = speech - floor
    hop = int(sr * 0.01)
    idx = np.where(db > max(floor + 12, speech - 35))[0]
    if len(idx):
        a = max(0, idx[0] * hop - int(0.1 * sr))
        b = min(len(x), (idx[-1] + 1) * hop + int(0.15 * sr))
        if b - a < len(x):
            x = x[a:b]
            sf.write(str(wav), x, sr, subtype="PCM_16")
    return round(len(x) / sr, 2), round(snr, 1)


def prepare_reference(src: Path, dst: Path) -> tuple[float, Optional[str], bool, float]:
    """Upload -> 24 kHz mono reference WAV: long files are cut at a pause (5–15 s, near 10 s), edges trimmed.
    Returns (duration, note for the UI or None, was_cut, snr_db)."""
    full = dst.with_name(f"full-{uuid.uuid4().hex[:6]}.wav")
    notes, cut = [], False
    try:
        total = ffmpeg_to_wav(src, full)
        if total <= MAX_REF_SECONDS:
            full.replace(dst)
        else:
            at = find_pause_cut(full) or min(TARGET_REF_SECONDS + 2, MAX_REF_SECONDS)
            ffmpeg_to_wav(full, dst, at)
            cut = True
            notes.append(f"File dài {total:.0f} s nên đã cắt lấy {at:.1f} s đầu (đến chỗ ngắt hơi gần nhất)")
    finally:
        full.unlink(missing_ok=True)
    duration, snr = trim_and_measure(dst)
    if snr < NOISY_SNR_DB:
        notes.append(f"Mẫu có nền ồn hoặc nhạc (SNR ~{snr:.0f} dB) — giọng clone có thể lẫn tạp âm, nên dùng đoạn sạch hơn")
    return duration, (" · ".join(notes) or None), cut, snr


MODEL_SPEED_MIN, MODEL_SPEED_MAX = 0.75, 1.2  # relative factor the model copes with (stutters below, drops words above)
PACE_MIN, PACE_MAX, PACE_DEFAULT = 3.0, 4.6, 3.8  # words/s the model reads cleanly; it copies the reference's pace,
# so a fast reference (Học Giả clip: 5.5 w/s) made it drop whole phrases at speed 1.0 — the pace cap is what prevents that


def ref_pace(style: Optional[dict]) -> Optional[float]:
    """Speaking pace of a reference clip in words/s (what the model will imitate)."""
    if not style:
        return None
    words, dur = len((style.get("ref_text") or "").split()), float(style.get("duration") or 0)
    return round(words / dur, 2) if words and dur > 0 else None


def plan_speed(user_speed: float, tempo: float, mode: str, pace: Optional[float]) -> tuple[float, float]:
    """Split the requested speed into (model speed factor, post time-stretch) so the model never has to read outside
    PACE_MIN–PACE_MAX words/s; the stretch makes up the difference, so the result is still `user_speed` × natural pace."""
    if mode == "model":
        return user_speed, tempo
    if mode == "stretch":
        return 1.0, tempo * user_speed
    natural = pace or PACE_DEFAULT
    desired = natural * user_speed
    if PACE_MIN * 0.97 <= desired <= PACE_MAX * 1.03:  # close enough: let the model do it all, no stretch pass
        model_pace = desired
    else:
        model_pace = min(max(desired, PACE_MIN), PACE_MAX)
    model_speed = min(max(model_pace / natural, MODEL_SPEED_MIN), MODEL_SPEED_MAX)
    return round(model_speed, 3), round(tempo * user_speed / model_speed, 3)


NOISY_FLOOR_DB = -74.0  # clean OmniVoice takes sit at -79…-81 dBFS between words; the "rè" take the user heard was -72.9


def audio_quality(audio: np.ndarray, sr: int = SR) -> dict:
    """Noise floor between words (10th percentile of 10 ms RMS frames) and speech level, in dBFS."""
    hop = int(sr * 0.01)
    n = len(audio) // hop
    if n < 10:
        return {"floor_db": None, "speech_db": None, "snr_db": None, "noisy": False}
    db = 20 * np.log10(np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-6)
    floor, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    return {"floor_db": round(floor, 1), "speech_db": round(speech, 1), "snr_db": round(speech - floor, 1), "noisy": floor > NOISY_FLOOR_DB}


def time_stretch(wav: Path, factor: float) -> float:
    """Change pace without touching pitch or words (rubberband, fallback atempo). factor > 1 = faster. Returns duration."""
    tmp = wav.with_name(wav.stem + "-stretch.wav")
    for flt in (f"rubberband=tempo={factor:.4f}:pitch=1:transients=crisp", f"atempo={factor:.4f}"):
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", flt, "-ar", str(SR), "-ac", "1",
                            "-sample_fmt", "s16", str(tmp)], capture_output=True, text=True)
        if p.returncode == 0 and tmp.exists():
            tmp.replace(wav)
            return round(sf.info(str(wav)).duration, 2)
    tmp.unlink(missing_ok=True)
    raise HTTPException(500, "ffmpeg không kéo giãn được audio")


def transcript_mismatch(text: str, duration: float) -> Optional[str]:
    words = len(text.split())
    if duration > 0 and words / duration > MAX_WORDS_PER_SECOND:
        return f"Lời bạn gõ ({words} từ) dài hơn nhiều so với {duration:.1f} s audio"
    return None


# ----------------------------------------------------------------------------- storage helpers

def safe_id(value: str) -> str:
    if not value or any(ch in value for ch in "/\\") or ".." in value:
        raise HTTPException(404, "Không tìm thấy")
    return value


def read_voice(vdir: Path) -> Optional[dict]:
    try:
        v = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    if "styles" not in v:  # v1 layout: one reference per voice
        v["styles"] = [{"id": "default", "label": "Mặc định", "file": "ref.wav", "ref_text": v.pop("ref_text", ""),
                        "duration": v.pop("duration", 0), "transcribed": v.pop("transcribed", False)}]
        v.pop("note", None)
        (vdir / "meta.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    return v


def list_voices() -> list[dict]:
    voices = [v for v in (read_voice(d) for d in VOICES_DIR.iterdir() if d.is_dir()) if v]
    voices.sort(key=lambda v: (bool(v.get("preset")), -float(v.get("created", 0))))  # user voices first, newest on top
    for v in voices:
        for s in v["styles"]:
            s["pace"] = ref_pace(s)
    return voices


def get_voice(voice_id: str) -> dict:
    v = read_voice(VOICES_DIR / safe_id(voice_id))
    if not v:
        raise HTTPException(404, "Không tìm thấy giọng")
    return v


def get_style(voice: dict, style_id: Optional[str]) -> dict:
    if not style_id:
        return voice["styles"][0]
    for s in voice["styles"]:
        if s["id"] == style_id:
            return s
    raise HTTPException(404, "Không tìm thấy sắc thái")


def save_voice(meta: dict) -> None:
    (VOICES_DIR / meta["id"] / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_output(out_id: str) -> dict:
    p = OUT_DIR / f"{safe_id(out_id)}.json"
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy bản ghi")
    return json.loads(p.read_text(encoding="utf-8"))


def save_output(meta: dict) -> None:
    (OUT_DIR / f"{meta['id']}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def project_path(pid: str) -> Path:
    return PROJECTS_DIR / safe_id(pid) / "project.json"


def get_project(pid: str) -> dict:
    p = project_path(pid)
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy dự án")
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- engine

class Engine:
    """Holds one OmniVoice model (+ its Whisper) in VRAM. Heavy calls run in a worker thread under `lock`."""

    def __init__(self) -> None:
        self.model = None
        self.key: Optional[str] = None
        self.lock = threading.Lock()
        self.stage: Optional[str] = None
        self.stage_detail: Optional[str] = None
        self.stage_since = 0.0
        self.load_seconds: Optional[float] = None
        self.last_rtf: Optional[float] = None
        self.error: Optional[str] = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def set_stage(self, stage: Optional[str], detail: Optional[str] = None) -> None:
        self.stage, self.stage_detail, self.stage_since = stage, detail, time.time()

    def ensure(self, key: str):
        if key not in MODELS:
            raise HTTPException(400, f"Model không hợp lệ: {key}")
        if self.model is not None and self.key == key:
            return self.model
        from omnivoice import OmniVoice  # slow import, keep it lazy

        self.set_stage("loading")
        if self.model is not None:
            log.info("Unloading %s", self.key)
            self.model, self.key = None, None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        t = time.time()
        log.info("Loading %s on %s", MODELS[key]["repo"], self.device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = OmniVoice.from_pretrained(MODELS[key]["repo"], device_map=self.device, dtype=dtype)
        self.key = key
        self.load_seconds = round(time.time() - t, 1)
        log.info("Loaded in %.1fs", self.load_seconds)
        return self.model

    @property
    def asr_loaded(self) -> bool:
        return self.model is not None and getattr(self.model, "_asr_pipe", None) is not None

    def asr(self, path: Path, language: str = "vi", words: bool = False):
        """Whisper (large-v3-turbo, loaded by OmniVoice) on a file: text, or the raw pipeline result with word chunks."""
        model = self.ensure(self.key or "base")
        if model._asr_pipe is None:
            self.set_stage("loading_asr")
            model.load_asr_model()
        gk = {"task": "transcribe"}
        if language and language != "auto":
            gk["language"] = language
        if words:
            return model._asr_pipe(str(path), return_timestamps="word", generate_kwargs=gk)
        return model._asr_pipe(str(path), generate_kwargs=gk)["text"].strip()

    def transcribe(self, wav: Path, language: str = "vi") -> str:
        self.set_stage("transcribing")
        return self.asr(wav, language)

    def clone_prompt(self, voice: dict, style: dict, key: str):
        from omnivoice import VoiceClonePrompt

        model = self.ensure(key)
        vdir = VOICES_DIR / voice["id"]
        cache = vdir / f"prompt-{key}-{style['id']}.pt"
        if cache.exists():
            try:
                return VoiceClonePrompt.load(str(cache))
            except Exception:
                cache.unlink(missing_ok=True)
        self.set_stage("encoding_ref")
        prompt = model.create_voice_clone_prompt(ref_audio=str(vdir / style["file"]), ref_text=style.get("ref_text") or None)
        prompt.save(str(cache))
        return prompt

    def generate(self, req: "GenRequest", voice: Optional[dict], style: Optional[dict], rules: list[dict]) -> dict:
        model = self.ensure(req.model)
        spoken = spoken_form(req.text, rules, req.read_numbers)
        if not spoken:
            raise HTTPException(400, "Sau khi áp từ điển, không còn gì để đọc")
        kwargs = {}
        if req.mode == "clone":
            kwargs["voice_clone_prompt"] = self.clone_prompt(voice, style, req.model)
        elif req.mode == "design":
            kwargs["instruct"] = req.instruct
        cuda = self.device.startswith("cuda")
        base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        want_qa = req.qa or req.takes > 1
        # speed: the model's own speed is a duration constraint tied to the reference's pace, so keep the pace it has
        # to produce in a safe band and time-stretch the rest
        pace = ref_pace(style) if req.mode == "clone" else None
        model_speed, stretch = plan_speed(req.speed, req.tempo, req.speed_mode, pace)
        candidates = []
        try:
            for k in range(req.takes):
                seed = base_seed if k == 0 else random.randint(0, 2**31 - 1)
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                self.set_stage("generating", f"bản {k + 1}/{req.takes}" if req.takes > 1 else None)
                t = time.time()
                audio = model.generate(
                    text=spoken,
                    language=None if req.language == "auto" else req.language,
                    speed=model_speed, num_step=req.num_step, guidance_scale=req.guidance_scale, denoise=req.denoise,
                    **kwargs,
                )[0]
                gen_s = time.time() - t
                audio = np.asarray(audio, dtype=np.float32)
                peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                if peak > 0.99:
                    audio = audio * (0.99 / peak)
                tmp = OUT_DIR / f"tmp-{uuid.uuid4().hex[:8]}.wav"
                sf.write(str(tmp), audio, SR)
                cand = {"seed": seed, "duration": round(len(audio) / SR, 2), "gen_seconds": round(gen_s, 2),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None, "path": tmp,
                        "quality": audio_quality(audio)}
                if want_qa:
                    self.set_stage("qa", f"bản {k + 1}/{req.takes}" if req.takes > 1 else None)
                    cand["qa"] = qa_compare(spoken, self.asr(tmp, req.language))
                candidates.append(cand)
                if want_qa and cand["qa"]["errors"] == 0 and not cand["quality"]["noisy"]:
                    break  # clean read, clean floor: no need for more takes
            # fewest misread words first, then a clean noise floor, then CER
            best = min(candidates, key=lambda c: ((c["qa"]["errors"], c["quality"]["noisy"], c["qa"]["cer"]) if c.get("qa")
                                                  else (0, c["quality"]["noisy"], 0)))
            out_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            best["path"].replace(OUT_DIR / f"{out_id}.wav")
        finally:
            for c in candidates:
                c["path"].unlink(missing_ok=True)
        dur = best["duration"]
        if abs(stretch - 1.0) > 0.01:
            self.set_stage("stretching")
            dur = time_stretch(OUT_DIR / f"{out_id}.wav", stretch)
        self.last_rtf = best["gen_seconds"] / dur if dur else None
        meta = {
            "id": out_id, "created": time.time(), "text": req.text, "spoken_text": spoken, "mode": req.mode,
            "voice_id": voice["id"] if voice else None,
            "voice_name": (voice["name"] + (f" · {style['label']}" if style and style["id"] != voice["styles"][0]["id"] else "")) if voice else None,
            "style_id": style["id"] if style else None,
            "instruct": req.instruct if req.mode == "design" else None,
            "language": req.language, "speed": req.speed, "model_speed": round(model_speed, 3), "tempo": round(stretch, 3),
            "ref_pace": pace, "model_pace": round((pace or PACE_DEFAULT) * model_speed, 2),
            "num_step": req.num_step, "seed": best["seed"],
            "guidance_scale": req.guidance_scale, "denoise": req.denoise, "model": req.model,
            "duration": dur, "gen_seconds": best["gen_seconds"], "rtf": round(best["gen_seconds"] / dur, 3) if dur else None,
            "peak_vram_gb": best["peak_vram_gb"], "qa": best.get("qa"), "quality": best["quality"],
            "takes": [{"seed": c["seed"], "duration": c["duration"], "errors": c["qa"]["errors"] if c.get("qa") else None,
                       "cer": c["qa"]["cer"] if c.get("qa") else None, "floor_db": c["quality"]["floor_db"], "noisy": c["quality"]["noisy"],
                       "chosen": c is best} for c in candidates],
            "project_id": req.project_id, "line": req.line, "url": f"/outputs/{out_id}.wav",
        }
        save_output(meta)
        log.info("Generated %s: %.1fs audio in %.1fs (RTF %.3f), %d take(s), errors=%s", out_id, dur, best["gen_seconds"],
                 meta["rtf"] or 0, len(candidates), meta["qa"]["errors"] if meta["qa"] else "-")
        return meta

    def qa_output(self, meta: dict) -> dict:
        self.set_stage("qa")
        meta["qa"] = qa_compare(meta.get("spoken_text") or meta["text"], self.asr(OUT_DIR / f"{meta['id']}.wav", meta.get("language", "vi")))
        save_output(meta)
        return meta

    def align_output(self, meta: dict) -> dict:
        self.set_stage("aligning")
        wav = OUT_DIR / f"{meta['id']}.wav"
        res = self.asr(wav, meta.get("language", "vi"), words=True)
        meta["alignment"] = align_words(meta.get("spoken_text") or meta["text"], res.get("chunks", []), sf.info(str(wav)).duration)
        save_output(meta)
        return meta

    def export_project(self, proj: dict, gap: float, align: bool) -> dict:
        pid = proj["id"]
        exp = PROJECTS_DIR / pid / "export"
        if exp.exists():
            shutil.rmtree(exp)
        exp.mkdir(parents=True)
        pieces, timeline, missing, t = [], [], [], 0.0
        silence = np.zeros(int(gap * SR), dtype=np.float32)
        for ln in proj.get("lines", []):
            n = int(ln.get("n", len(timeline) + 1))
            label = re.sub(r"[^\w-]", "", str(ln.get("label") or f"{n:02d}")) or f"{n:02d}"
            out_id = ln.get("output_id")
            if not out_id or not (OUT_DIR / f"{out_id}.wav").exists():
                missing.append(label)
                continue
            meta = get_output(out_id)
            if align and not meta.get("alignment"):
                self.set_stage("aligning", label)
                meta = self.align_output(meta)
            self.set_stage("exporting", label)
            x, sr = sf.read(str(OUT_DIR / f"{out_id}.wav"), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            sf.write(str(exp / f"{label}.wav"), x, sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(exp / f"{label}.wav"), "-codec:a", "libmp3lame", "-q:a", "2",
                            str(exp / f"{label}.mp3")], capture_output=True)
            dur = len(x) / sr
            (exp / f"{label}.json").write_text(json.dumps({
                "id": label, "text": meta.get("spoken_text") or meta["text"], "display_text": meta["text"], "provider": "omnivoice",
                "voice": meta.get("voice_name"), "seed": meta["seed"], "duration": round(dur, 3),
                "alignment": (meta.get("alignment") or {}).get("alignment"), "matched": (meta.get("alignment") or {}).get("matched"),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            if pieces:
                pieces.append(silence)
                t += gap
            pieces.append(x)
            timeline.append({"n": n, "label": label, "text": meta["text"], "spoken_text": meta.get("spoken_text"), "start": round(t, 3),
                             "end": round(t + dur, 3), "duration": round(dur, 3), "output_id": out_id, "seed": meta["seed"],
                             "errors": (meta.get("qa") or {}).get("errors"),
                             "words": [{"w": w["w"], "start": round(t + w["start"], 3), "end": round(t + w["end"], 3)}
                                       for w in (meta.get("alignment") or {}).get("words", [])]})
            t += dur
        if not pieces:
            raise HTTPException(400, "Chưa có dòng nào được tạo để xuất")
        full = np.concatenate(pieces)
        sf.write(str(exp / "full.wav"), full, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(exp / "full.wav"), "-codec:a", "libmp3lame", "-q:a", "2",
                        str(exp / "full.mp3")], capture_output=True)
        (exp / "timeline.json").write_text(json.dumps({"project": pid, "name": proj.get("name"), "gap": gap, "duration": round(t, 3),
                                                       "lines": timeline, "missing": missing}, ensure_ascii=False, indent=1), encoding="utf-8")
        with zipfile.ZipFile(exp / "export.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(exp.iterdir()):
                if p.name != "export.zip":
                    z.write(p, p.name)
        return {"url": f"/projects/{pid}/export/export.zip", "dir": str(exp), "duration": round(t, 2),
                "lines": len(timeline), "missing": missing, "full_url": f"/projects/{pid}/export/full.wav"}


ENGINE = Engine()


async def run_locked(fn, *args):
    """Run a heavy engine call in a worker thread; 409 if another one is in flight."""
    if not ENGINE.lock.acquire(blocking=False):
        raise HTTPException(409, "Máy đang bận việc khác, đợi xíu nhé.")
    try:
        return await asyncio.to_thread(fn, *args)
    except HTTPException:
        raise
    except Exception as e:  # surface the real error to the UI
        log.exception("worker failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        ENGINE.set_stage(None)
        ENGINE.lock.release()


# ----------------------------------------------------------------------------- API

app = FastAPI(title="OmniVoice Studio", description="Developed by Kenzy", docs_url="/api/docs", redoc_url=None)


class GenRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    mode: str = Field("clone", pattern="^(clone|design|auto)$")
    voice_id: Optional[str] = None
    style_id: Optional[str] = None
    instruct: Optional[str] = None
    language: str = "vi"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    num_step: int = Field(32, ge=8, le=64)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    guidance_scale: float = Field(2.0, ge=1.0, le=5.0)
    denoise: bool = True
    model: str = "base"
    read_numbers: bool = True
    tempo: float = Field(1.0, ge=0.5, le=2.0)   # extra time-stretch after synthesis (keeps pitch and words)
    speed_mode: str = Field("auto", pattern="^(auto|model|stretch)$")
    takes: int = Field(1, ge=1, le=5)
    qa: bool = True
    project_id: Optional[str] = None
    line: Optional[int] = None


class LoadRequest(BaseModel):
    model: str


class VoiceUpdate(BaseModel):
    name: Optional[str] = None


class StyleUpdate(BaseModel):
    label: Optional[str] = None
    ref_text: Optional[str] = None


class FromOutput(BaseModel):
    output_id: str
    name: Optional[str] = None
    voice_id: Optional[str] = None
    label: Optional[str] = None


class RulesBody(BaseModel):
    rules: list[dict]


class PreviewBody(BaseModel):
    text: str
    read_numbers: bool = True


class ExportBody(BaseModel):
    gap: float = Field(0.3, ge=0, le=3)
    align: bool = True


@app.get("/api/status")
def api_status():
    gpu = total = used = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        free, tot = torch.cuda.mem_get_info(0)
        total, used = round(tot / 2**30, 2), round((tot - free) / 2**30, 2)
    return {
        "device": ENGINE.device, "gpu": gpu, "vram_total_gb": total, "vram_used_gb": used,
        "model": ENGINE.key, "models": MODELS, "languages": LANGUAGES, "asr_loaded": ENGINE.asr_loaded,
        "busy": ENGINE.lock.locked(), "stage": ENGINE.stage, "stage_detail": ENGINE.stage_detail,
        "stage_seconds": round(time.time() - ENGINE.stage_since, 1) if ENGINE.stage else 0,
        "load_seconds": ENGINE.load_seconds, "last_rtf": ENGINE.last_rtf, "error": ENGINE.error,
        "num2words": num2words is not None,
    }


@app.post("/api/model/load")
async def api_load(req: LoadRequest):
    await run_locked(ENGINE.ensure, req.model)
    return {"model": ENGINE.key, "load_seconds": ENGINE.load_seconds}


# ---- pronunciation

@app.get("/api/pronunciation")
def api_rules():
    return {"rules": load_rules(), "defaults": DEFAULT_RULES}


@app.put("/api/pronunciation")
def api_save_rules(body: RulesBody):
    rules = [{"from": str(r.get("from", "")).strip(), "to": str(r.get("to", "")).strip(), "enabled": bool(r.get("enabled", True)),
              "whole": bool(r.get("whole", True)), "regex": bool(r.get("regex", False))} for r in body.rules if str(r.get("from", "")).strip()]
    save_rules(rules)
    return {"rules": rules}


@app.post("/api/pronunciation/preview")
def api_preview(body: PreviewBody):
    return {"spoken": spoken_form(body.text, load_rules(), body.read_numbers)}


# ---- generation, QA, alignment

@app.post("/api/generate")
async def api_generate(req: GenRequest):
    req.text = req.text.strip()
    if not req.text:
        raise HTTPException(400, "Kịch bản đang trống")
    voice = style = None
    if req.mode == "clone":
        if not req.voice_id:
            raise HTTPException(400, "Chọn một giọng để nhân bản đã")
        voice = get_voice(req.voice_id)
        style = get_style(voice, req.style_id)
    elif req.mode == "design":
        req.instruct = (req.instruct or "").strip()
        if not req.instruct:
            raise HTTPException(400, "Chọn ít nhất một thuộc tính giọng")
    return await run_locked(ENGINE.generate, req, voice, style, load_rules())


@app.post("/api/qa/{out_id}")
async def api_qa(out_id: str):
    return await run_locked(ENGINE.qa_output, get_output(out_id))


@app.post("/api/align/{out_id}")
async def api_align(out_id: str):
    return await run_locked(ENGINE.align_output, get_output(out_id))


# ---- voices and styles

async def _make_style(vdir: Path, file_src: Path, label: str, ref_text: str, language: str, style_id: str,
                      known_text_matches: bool = False) -> tuple[dict, Optional[str]]:
    """Shared by every way a reference enters the app: prepare the audio, decide whether the typed transcript can be
    trusted, otherwise let Whisper transcribe exactly what was kept."""
    dst = vdir / ("ref.wav" if style_id == "default" else f"styles/{style_id}.wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration, note, cut, snr = prepare_reference(file_src, dst)
    if duration < MIN_REF_SECONDS:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "Đoạn mẫu quá ngắn, cần ít nhất 1 giây")
    text, transcribed = (ref_text or "").strip(), False
    reason = None
    if text and not cut and not known_text_matches:
        reason = transcript_mismatch(text, duration)
    elif text and cut:
        reason = "đã cắt file"
    if not text or reason:
        text = await run_locked(ENGINE.transcribe, dst, language)
        transcribed = True
        if reason and reason != "đã cắt file":
            note = (note + " · " if note else "") + reason
        if reason:
            note = (note or "") + " → Whisper đã chép lại lời cho đúng đoạn được giữ"
    style = {"id": style_id, "label": label.strip() or "Mặc định", "file": str(dst.relative_to(vdir)).replace("\\", "/"),
             "ref_text": text, "duration": duration, "transcribed": transcribed, "snr_db": snr}
    return style, note


@app.get("/api/voices")
def api_voices():
    return list_voices()


@app.post("/api/voices")
async def api_add_voice(name: str = Form(...), ref_text: str = Form(""), label: str = Form("Mặc định"),
                        language: str = Form("vi"), file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Thiếu tên giọng")
    vid = "v-" + uuid.uuid4().hex[:10]
    vdir = VOICES_DIR / vid
    vdir.mkdir()
    tmp = vdir / f"upload{Path(file.filename or 'audio').suffix.lower()[:8] or '.bin'}"
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        style, note = await _make_style(vdir, tmp, label, ref_text, language, "default")
    except Exception:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)
    meta = {"id": vid, "name": name, "color": random.choice(COLORS), "created": time.time(), "source": "upload",
            "preset": False, "styles": [style]}
    save_voice(meta)
    return {**meta, "note": note}


@app.post("/api/voices/from-output")
async def api_voice_from_output(req: FromOutput):
    meta = get_output(req.output_id)
    src = OUT_DIR / f"{meta['id']}.wav"
    if req.voice_id:  # new style on an existing voice
        voice = get_voice(req.voice_id)
        vdir = VOICES_DIR / voice["id"]
        sid = "s-" + uuid.uuid4().hex[:6]
        style, note = await _make_style(vdir, src, req.label or f"Sắc thái {len(voice['styles']) + 1}",
                                        meta.get("spoken_text") or meta["text"], meta.get("language", "vi"), sid, known_text_matches=True)
        voice["styles"].append(style)
        save_voice(voice)
        return {**voice, "note": note}
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "Thiếu tên giọng")
    vid = "v-" + uuid.uuid4().hex[:10]
    vdir = VOICES_DIR / vid
    vdir.mkdir()
    try:
        style, note = await _make_style(vdir, src, req.label or "Mặc định", meta.get("spoken_text") or meta["text"],
                                        meta.get("language", "vi"), "default", known_text_matches=True)
    except Exception:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    voice = {"id": vid, "name": name, "color": random.choice(COLORS), "created": time.time(), "source": f"output:{meta['id']}",
             "preset": False, "styles": [style]}
    save_voice(voice)
    return {**voice, "note": note}


@app.post("/api/voices/{voice_id}/styles")
async def api_add_style(voice_id: str, label: str = Form(...), ref_text: str = Form(""), language: str = Form("vi"),
                        file: UploadFile = File(...)):
    voice = get_voice(voice_id)
    vdir = VOICES_DIR / voice["id"]
    sid = "s-" + uuid.uuid4().hex[:6]
    tmp = vdir / f"upload{Path(file.filename or 'audio').suffix.lower()[:8] or '.bin'}"
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        style, note = await _make_style(vdir, tmp, label, ref_text, language, sid)
    finally:
        tmp.unlink(missing_ok=True)
    voice["styles"].append(style)
    save_voice(voice)
    return {**voice, "note": note}


@app.patch("/api/voices/{voice_id}")
def api_update_voice(voice_id: str, req: VoiceUpdate):
    v = get_voice(voice_id)
    if req.name is not None and req.name.strip():
        v["name"] = req.name.strip()
    save_voice(v)
    return v


@app.patch("/api/voices/{voice_id}/styles/{style_id}")
def api_update_style(voice_id: str, style_id: str, req: StyleUpdate):
    v = get_voice(voice_id)
    s = get_style(v, style_id)
    if req.label is not None and req.label.strip():
        s["label"] = req.label.strip()
    if req.ref_text is not None and req.ref_text.strip() and req.ref_text.strip() != s.get("ref_text"):
        reason = transcript_mismatch(req.ref_text.strip(), s.get("duration", 0))
        if reason:
            raise HTTPException(400, reason + ". Chỉ gõ đúng những gì nghe thấy trong đoạn mẫu.")
        s["ref_text"], s["transcribed"] = req.ref_text.strip(), False
        for p in (VOICES_DIR / voice_id).glob(f"prompt-*-{style_id}.pt"):  # cached prompts embed the old text
            p.unlink()
    save_voice(v)
    return v


@app.delete("/api/voices/{voice_id}/styles/{style_id}")
def api_delete_style(voice_id: str, style_id: str):
    v = get_voice(voice_id)
    s = get_style(v, style_id)
    if len(v["styles"]) == 1:
        raise HTTPException(400, "Giọng cần ít nhất một sắc thái — xoá cả giọng nếu không dùng nữa")
    v["styles"] = [x for x in v["styles"] if x["id"] != style_id]
    (VOICES_DIR / voice_id / s["file"]).unlink(missing_ok=True)
    for p in (VOICES_DIR / voice_id).glob(f"prompt-*-{style_id}.pt"):
        p.unlink()
    save_voice(v)
    return v


@app.delete("/api/voices/{voice_id}")
def api_delete_voice(voice_id: str):
    get_voice(voice_id)
    shutil.rmtree(VOICES_DIR / voice_id)
    return {"ok": True}


@app.get("/api/voices/{voice_id}/audio")
def api_voice_audio(voice_id: str, style: Optional[str] = None):
    v = get_voice(voice_id)
    return FileResponse(VOICES_DIR / voice_id / get_style(v, style)["file"], media_type="audio/wav")


# ---- history

@app.get("/api/history")
def api_history(limit: int = 200, project: Optional[str] = None):
    items = []
    for p in sorted(OUT_DIR.glob("*.json"), reverse=True):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if project and m.get("project_id") != project:
            continue
        items.append(m)
        if len(items) >= limit:
            break
    return items


@app.delete("/api/history/{out_id}")
def api_delete_output(out_id: str):
    get_output(out_id)
    for ext in (".json", ".wav"):
        (OUT_DIR / f"{out_id}{ext}").unlink(missing_ok=True)
    return {"ok": True}


# ---- projects (script mode)

@app.get("/api/projects")
def api_projects():
    items = []
    for d in PROJECTS_DIR.iterdir():
        if (d / "project.json").exists():
            try:
                p = json.loads((d / "project.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            lines = p.get("lines", [])
            items.append({"id": p["id"], "name": p.get("name") or p["id"], "updated": p.get("updated", 0),
                          "lines": len(lines), "done": sum(1 for ln in lines if ln.get("output_id"))})
    items.sort(key=lambda x: -x["updated"])
    return items


@app.get("/api/projects/{pid}")
def api_project(pid: str):
    return get_project(pid)


@app.put("/api/projects/{pid}")
def api_save_project(pid: str, body: dict):
    safe_id(pid)
    if not isinstance(body.get("lines"), list):
        raise HTTPException(400, "Dự án cần danh sách dòng")
    body["id"] = pid
    body.setdefault("created", time.time())
    body["updated"] = time.time()
    p = project_path(pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    return body


@app.delete("/api/projects/{pid}")
def api_delete_project(pid: str):
    get_project(pid)
    shutil.rmtree(PROJECTS_DIR / pid)
    return {"ok": True}


@app.post("/api/projects/{pid}/export")
async def api_export_project(pid: str, body: ExportBody):
    return await run_locked(ENGINE.export_project, get_project(pid), body.gap, body.align)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# ----------------------------------------------------------------------------- main

def preload(key: str, with_asr: bool) -> None:
    with ENGINE.lock:
        try:
            ENGINE.ensure(key)
            if with_asr:
                ENGINE.set_stage("loading_asr")
                ENGINE.model.load_asr_model()
        except Exception as e:
            ENGINE.error = f"{type(e).__name__}: {e}"
            log.exception("preload failed")
        finally:
            ENGINE.set_stage(None)


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="OmniVoice Studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="base", choices=list(MODELS))
    ap.add_argument("--no-preload", action="store_true", help="don't load the model until the first request")
    ap.add_argument("--no-asr", action="store_true", help="don't preload Whisper (it loads on first QA/transcription)")
    args = ap.parse_args()
    if not args.no_preload:
        threading.Thread(target=preload, args=(args.model, not args.no_asr), daemon=True).start()
    log.info("OmniVoice Studio -> http://%s:%d  (device %s)", args.host, args.port, ENGINE.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
