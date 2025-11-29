# utils.py
import sys
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import re
import os
import logging
import time
import base64
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urljoin
import asyncio

import httpx
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader

# Use sync_playwright inside a thread to avoid asyncio subprocess limitations on Windows
from playwright.sync_api import sync_playwright

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(ch)

REQUEST_TIMEOUT = 60.0
BROWSER_NAV_TIMEOUT_MS = 60_000
DEFAULT_DOWNLOAD_DIR = "downloads"


def clean_url_fragment(fragment: str) -> str:
    if fragment is None:
        return ""
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return fragment.strip()


def _render_page_sync(url: str, wait_for: float = 2.0) -> Tuple[str, str]:
    logger.info("_render_page_sync: %s", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=BROWSER_NAV_TIMEOUT_MS, wait_until="networkidle")
            if wait_for and wait_for > 0:
                page.wait_for_timeout(int(wait_for * 1000))

            html = page.content()
            text = ""
            if page.query_selector("#result"):
                result_html = page.inner_html("#result")
                html = f"<div id='result'>{result_html}</div>"
                try:
                    text = page.inner_text("#result")
                except Exception:
                    soup = BeautifulSoup(result_html, "html.parser")
                    text = soup.get_text("\n", strip=True)
            else:
                try:
                    text = page.inner_text("body")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text("\n", strip=True)

            try:
                context.close()
            except Exception:
                pass
            return html, text
        finally:
            try:
                browser.close()
            except Exception:
                pass


async def render_page_get_text(url: str, wait_for: float = 2.0) -> Tuple[str, str]:
    logger.info("render_page_get_text: %s", url)
    try:
        return await asyncio.to_thread(_render_page_sync, url, wait_for)
    except NotImplementedError as e:
        logger.exception("Sync-playwright thread failed: %s", e)
        # fallback: simple HTTP fetch (no JS)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(url)
                r.raise_for_status()
                html = r.text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text("\n", strip=True)
                return html, text
        except Exception as e2:
            logger.exception("Fallback HTTP fetch failed: %s", e2)
            raise


def extract_submit_info(html: str, base_url: str, page_text: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    submit_candidates: List[str] = []
    linked_files: List[str] = []

    def add_candidate(raw: str):
        if not raw:
            return
        cleaned = clean_url_fragment(raw)
        if not cleaned:
            return
        if not cleaned.startswith("http"):
            cleaned = urljoin(base_url, cleaned)
        submit_candidates.append(cleaned)

    for f in soup.find_all("form"):
        add_candidate(f.get("action", ""))

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith((".pdf", ".csv", ".xlsx", ".xls", ".mp3", ".wav", ".ogg", ".m4a", ".opus")):
            linked_files.append(href if href.startswith("http") else urljoin(base_url, href))
        add_candidate(href)

    combined = html + ("\n" + (page_text or ""))
    for m in re.finditer(r"(https?://[^\s'\"<>]+)", combined):
        add_candidate(m.group(0))

    for m in re.finditer(r"['\"]?(/[^'\"\s<>]*submit[^'\"\s<>]*)['\"]?", combined, flags=re.IGNORECASE):
        add_candidate(m.group(1))

    seen = set()
    submit_candidates_clean = []
    for c in submit_candidates:
        if c not in seen:
            seen.add(c)
            submit_candidates_clean.append(c)

    submit_url = None
    for c in submit_candidates_clean:
        low = c.lower()
        if ("submit" in low) or ("answer" in low) or ("/post" in low) or ("/response" in low):
            submit_url = c
            break
    if submit_url is None and submit_candidates_clean:
        submit_url = submit_candidates_clean[0]

    text = page_text or soup.get_text(" ", strip=True)
    hint = None
    m = re.search(r"sum of the\s+[\"']?(\w+)[\"']?\s+column", text, flags=re.IGNORECASE)
    if m:
        hint = {"operation": "sum", "column": m.group(1)}

    return {"submit_url": submit_url, "submit_candidates": submit_candidates_clean, "hint": hint, "linked_files": linked_files}


async def download_file_if_needed(url: str, session: httpx.AsyncClient, download_dir: str = DEFAULT_DOWNLOAD_DIR) -> Optional[str]:
    logger.info("download_file_if_needed: %s", url)
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    try:
        r = await session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        logger.warning("download_file_if_needed: GET failed %s: %s", url, e)
        return None

    content_type = r.headers.get("content-type", "").lower()
    parsed_name = url.split("?")[0].split("/")[-1] or f"file-{int(time.time())}"
    ext = parsed_name.split(".")[-1].lower()

    is_pdf = "pdf" in content_type or ext == "pdf"
    is_csv = "csv" in content_type or ext == "csv"
    is_excel = "excel" in content_type or ext in ("xls", "xlsx")
    is_image = content_type.startswith("image/") or ext in ("png", "jpg", "jpeg", "gif")
    is_audio = any(e in content_type for e in ("audio/", "mpeg", "wav", "ogg", "opus")) or ext in ("mp3", "wav", "ogg", "m4a", "opus")

    if not (is_pdf or is_csv or is_excel or is_image or is_audio):
        logger.info("download_file_if_needed: not a supported file type (%s, ext=%s)", content_type, ext)
        return None

    filename = f"{int(time.time())}-{parsed_name}"
    path = Path(download_dir) / filename
    path.write_bytes(r.content)
    logger.info("download_file_if_needed: saved %s", str(path))
    return str(path)


def try_simple_compute(page_html: str, page_text: str, base_url: str) -> Optional[Any]:
    logger.info("try_simple_compute: attempt basic computations")
    try:
        tables = pd.read_html(page_html)
    except Exception:
        tables = []

    for df in tables:
        cols = [str(c) for c in df.columns]
        cols_lower = [c.lower() for c in cols]
        if "value" in cols_lower:
            colname = df.columns[cols_lower.index("value")]
            nums = pd.to_numeric(df[colname], errors="coerce").dropna()
            total = nums.sum()
            if pd.isna(total):
                continue
            return int(total) if float(total).is_integer() else float(total)

    m_sum = re.search(r"sum of the\s+[\"']?(\w+)[\"']?\s+column", page_text, flags=re.IGNORECASE)
    if m_sum:
        col = m_sum.group(1).lower()
        for df in tables:
            cols_lower = [str(c).lower() for c in df.columns]
            if col in cols_lower:
                colname = df.columns[cols_lower.index(col)]
                nums = pd.to_numeric(df[colname], errors="coerce").dropna()
                total = nums.sum()
                if pd.isna(total):
                    continue
                return int(total) if float(total).is_integer() else float(total)
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", page_text)]
        if nums:
            total = sum(nums)
            return int(total) if float(total).is_integer() else float(total)

    return None


def extract_numbers_from_pdf(path: str) -> Optional[float]:
    try:
        reader = PdfReader(path)
        nums = []
        for page in reader.pages:
            text = page.extract_text() or ""
            found = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            nums.extend([float(x) for x in found])
        if not nums:
            return None
        s = sum(nums)
        return int(s) if float(s).is_integer() else float(s)
    except Exception as e:
        logger.warning("extract_numbers_from_pdf failed: %s", e)
        return None


def extract_answer_from_page(html: str, page_text: Optional[str] = None) -> Optional[Any]:
    try:
        m = re.search(r"atob\(\s*`([^`]+)`\s*\)", html)
        if not m:
            m = re.search(r"atob\(\s*['\"]([^'\"]+)['\"]\s*\)", html)
        if m:
            b64 = m.group(1)
            try:
                decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
            except Exception:
                decoded = ""
            if "{" in decoded and "}" in decoded:
                try:
                    start = decoded.find("{")
                    end = decoded.rfind("}") + 1
                    candidate = decoded[start:end]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        return parsed["answer"]
                except Exception:
                    pass
            m2 = re.search(r'"answer"\s*:\s*([0-9]+|"(?:[^"]*)")', decoded)
            if m2:
                val = m2.group(1)
                try:
                    return json.loads(val)
                except Exception:
                    return val.strip('"')
    except Exception:
        pass

    combined = html + ("\n" + (page_text or ""))
    m3 = re.search(r'"answer"\s*:\s*([0-9]+|"(?:[^"]*)")', combined)
    if m3:
        val = m3.group(1)
        try:
            return json.loads(val)
        except Exception:
            return val.strip('"')

    m4 = re.search(r"Answer\s*[:=]\s*([0-9]+)", combined, flags=re.IGNORECASE)
    if m4:
        try:
            return int(m4.group(1))
        except Exception:
            return m4.group(1)

    return None


# Audio helpers --------------------------------------------------------------

def find_audio_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("audio"):
        src = a.get("src")
        if src:
            urls.append(src if src.startswith("http") else urljoin(base_url, src))
        for s in a.find_all("source"):
            ssrc = s.get("src")
            if ssrc:
                urls.append(ssrc if ssrc.startswith("http") else urljoin(base_url, ssrc))
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(href.lower().endswith(ext) for ext in (".mp3", ".wav", ".ogg", ".m4a", ".opus")):
            urls.append(href if href.startswith("http") else urljoin(base_url, href))
    seen = set()
    clean = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


async def download_audio(url: str, session: httpx.AsyncClient, out_dir: str = DEFAULT_DOWNLOAD_DIR) -> Optional[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        r = await session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        logger.warning("download_audio: failed GET %s: %s", url, e)
        return None

    parsed_name = url.split("?")[0].split("/")[-1] or f"audio-{int(time.time())}"
    out_path = Path(out_dir) / f"{int(time.time())}-{parsed_name}"
    out_path.write_bytes(r.content)
    logger.info("download_audio: saved %s", str(out_path))
    return str(out_path)


def transcribe_with_whisper_local(audio_path: str, model: str = "small") -> Optional[str]:
    try:
        import whisper
        model_w = whisper.load_model(model)
        result = model_w.transcribe(audio_path)
        text = result.get("text", "").strip()
        return text if text else None
    except Exception as e:
        logger.warning("transcribe_with_whisper_local failed: %s", e)
        return None


def transcribe_audio_path(audio_path: str) -> Optional[str]:
    t = transcribe_with_whisper_local(audio_path, model="small")
    if t:
        return t
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8_float16")
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        return text if text else None
    except Exception as e:
        logger.warning("faster-whisper fallback failed: %s", e)
    return None


def transcribe_audio_via_openai(audio_path: str, model: str = "whisper-1", timeout: int = 120) -> Optional[str]:
    """
    Upload audio to OpenAI's transcription API (cloud). Requires OPENAI_API_KEY env var.
    Tries modern OpenAI client first, then legacy openai package.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.debug("transcribe_audio_via_openai: OPENAI_API_KEY not set")
        return None

    try:
        # Try new OpenAI client (openai >= 1.0)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            with open(audio_path, "rb") as af:
                resp = client.audio.transcriptions.create(model=model, file=af, timeout=timeout)
            # resp may be a dict-like or object; attempt to read "text"
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if text:
                return text.strip()
        except Exception as e_new:
            # fallback to legacy openai package
            import openai
            openai.api_key = api_key
            with open(audio_path, "rb") as af:
                try:
                    result = openai.Audio.transcribe(model, af, timeout=timeout)
                    text = result.get("text") if isinstance(result, dict) else getattr(result, "text", None)
                    if text:
                        return text.strip()
                except Exception:
                    af.seek(0)
                    resp2 = openai.Audio.transcriptions.create(model=model, file=af)
                    text = resp2.get("text") if isinstance(resp2, dict) else getattr(resp2, "text", None)
                    if text:
                        return text.strip()
    except Exception as e:
        logger.warning("transcribe_audio_via_openai failed: %s", e)
        return None

    return None


# Placeholder LLM solver - replace with your real LLM client
async def llm_solve_if_needed(page_text: str, downloaded_paths: Optional[List[str]] = None) -> Optional[Any]:
    logger.info("llm_solve_if_needed: called (placeholder)")
    if "true or false" in (page_text or "").lower():
        return True
    if downloaded_paths:
        for p in downloaded_paths:
            if p.lower().endswith(".pdf"):
                s = extract_numbers_from_pdf(p)
                if s is not None:
                    return s
    return None
