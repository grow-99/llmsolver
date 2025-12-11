# utils.py
import os
import re
import glob
import time
import json
import hashlib
import logging
import httpx
import pandas as pd
from typing import Optional
import asyncio

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# -------------------------
# Playwright rendering (sync wrapped into asyncio)
# -------------------------
def _render_page_sync(url: str, timeout: int = 30) -> str:
    """
    Synchronous Playwright rendering. We import inside the function to avoid
    import-time issues when main loads dotenv before worker imports Playwright.
    """
    from playwright.sync_api import sync_playwright

    logger.info(f"_render_page_sync: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=timeout * 1000)
            # Wait for network to be mostly idle or a short time to render dynamic content
            page.wait_for_load_state("networkidle", timeout=timeout * 1000)
        except Exception:
            # fallback: give it a bit more time
            try:
                time.sleep(1)
            except Exception:
                pass
        content = page.content()
        browser.close()
        return content


async def render_page_get_text(url: str, timeout: int = 30) -> str:
    """
    Async wrapper to call the sync Playwright renderer in a thread.
    Returns the raw HTML content (string).
    """
    logger.info(f"render_page_get_text: {url}")
    try:
        return await asyncio.to_thread(_render_page_sync, url, timeout)
    except Exception as e:
        logger.exception("render_page_get_text failed")
        raise


# -------------------------
# File download helpers
# -------------------------
def _safe_filename_from_url(url: str) -> str:
    name = os.path.basename(url.split("?")[0])
    if not name:
        h = hashlib.sha1(url.encode()).hexdigest()[:8]
        name = f"file-{h}"
    return name


def download_file_if_needed(url: str, downloads_dir: Optional[str] = None, client: Optional[httpx.Client] = None) -> Optional[str]:
    """
    Download a file if not present. Returns local path or None on failure.
    Synchronous helper (used from worker which is async by awaiting run_in_executor).
    """
    downloads_dir = downloads_dir or DOWNLOAD_DIR
    os.makedirs(downloads_dir, exist_ok=True)
    local_name = _safe_filename_from_url(url)
    local_path = os.path.join(downloads_dir, local_name)

    if os.path.exists(local_path):
        logger.info(f"download_file_if_needed: already exists {local_path}")
        return local_path

    logger.info(f"download_file_if_needed: {url}")
    client = client or httpx.Client(timeout=60)
    try:
        r = client.get(url, follow_redirects=True, timeout=60.0)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        logger.info(f"download_file_if_needed: saved {local_path}")
        return local_path
    except Exception:
        logger.exception("download_file_if_needed failed")
        return None


async def download_file(url: str, downloads_dir: Optional[str] = None) -> Optional[str]:
    return await asyncio.to_thread(download_file_if_needed, url, downloads_dir)


def download_audio(url: str, downloads_dir: Optional[str] = None) -> Optional[str]:
    return download_file_if_needed(url, downloads_dir)


# -------------------------
# Transcription helpers (Whisper local)
# -------------------------
def transcribe_with_whisper_local(audio_path: str) -> Optional[str]:
    """
    Transcribe an audio file using the installed whisper package.
    Returns transcription string or None on error.
    """
    try:
        import whisper
    except Exception:
        logger.warning("transcribe_with_whisper_local: whisper not available")
        return None

    try:
        # choose a small-ish model that balances speed and accuracy
        model = whisper.load_model("base")
        logger.info("transcribe_with_whisper_local: loaded whisper model")
        # model.transcribe returns dict with 'text'
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        logger.info("transcribe_with_whisper_local: done")
        return text
    except Exception:
        logger.exception("transcribe_with_whisper_local failed")
        return None


# -------------------------
# Simple deterministic computation helpers
# -------------------------
def find_first_csv_file_in_downloads(downloads_dir: Optional[str] = None) -> Optional[str]:
    downloads_dir = downloads_dir or DOWNLOAD_DIR
    glob_path = os.path.join(downloads_dir, "*.csv")
    files = sorted(glob.glob(glob_path), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def extract_number_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    # find numbers (ints or decimals)
    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if not nums:
        return None
    # prefer a "cutoff" nearby if present
    m = re.search(r'cutoff[^0-9\-+.\n\r]{0,40}([-+]?\d*\.\d+|[-+]?\d+)', text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    # otherwise return last number
    try:
        return float(nums[-1])
    except:
        return None


import re
import pandas as pd
from typing import Optional

_NUMBER_CLEAN_RE = re.compile(r"[^\d\.\-+]")  # remove everything except digits, dot, sign

def _clean_num_string(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # remove thousands separators and currency symbols, but keep minus and dot
    cleaned = _NUMBER_CLEAN_RE.sub("", s)
    # if multiple dots (rare), keep first two parts
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = ".".join([parts[0], "".join(parts[1:])])
    try:
        return float(cleaned)
    except Exception:
        return None

def compute_answer_from_csv(csv_path: str, cutoff: Optional[float] = None) -> Optional[float]:
    """
    Strict logic:
    - Read CSV via pandas.
    - Use the *first data column* (drop any leading index-like column if it's unnamed and contains 0,1,2,...).
    - Convert each cell to float using _clean_num_string.
    - If cutoff provided, sum values >= cutoff. If cutoff is None, sum all numeric values.
    - Return float (or int if whole number).
    """
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # fallback: try very simple parsing (first column)
        try:
            vals = []
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) <= 1:
                return None
            # take rows after header
            for row in lines[1:]:
                first = row.split(",")[0]
                n = _clean_num_string(first)
                if n is not None:
                    vals.append(n)
            if not vals:
                return None
            if cutoff is None:
                total = sum(vals)
            else:
                total = sum(v for v in vals if v >= cutoff)
            # return int if whole
            if abs(total - round(total)) < 1e-9:
                return int(round(total))
            return float(total)
        except Exception:
            return None

    if df.empty:
        return None

    # drop fully-empty columns
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        return None

    # Heuristic: if first column is an index-like 0..n, skip it
    first_col_name = df.columns[0]
    first_col = df[first_col_name].astype(str)
    # check if index-like: many values are sequential integers
    sample = first_col.dropna().head(20).tolist()
    is_index_like = False
    try:
        ints = [int(float(_clean_num_string(x) or 0)) for x in sample]
        # if many are 0..n pattern
        if all(abs(a - b) <= 1 for a, b in zip(ints, ints[1:])):
            is_index_like = True
    except Exception:
        is_index_like = False

    if is_index_like and df.shape[1] > 1:
        # use second column as data
        data_series = df.iloc[:, 1]
    else:
        data_series = df.iloc[:, 0]

    # coerce series to numeric using clean function
    numeric_vals = []
    for v in data_series.astype(str).tolist():
        n = _clean_num_string(v)
        if n is not None:
            numeric_vals.append(n)

    if not numeric_vals:
        return None

    # find cutoff if not provided (look for column names or other columns)
    if cutoff is None:
        # look for column named 'cutoff' or 'threshold'
        for col in df.columns:
            if str(col).strip().lower() in ("cutoff", "threshold"):
                try:
                    maybe = _clean_num_string(str(df[col].dropna().iloc[0]))
                    if maybe is not None:
                        cutoff = maybe
                        break
                except Exception:
                    pass

    # compute final sum
    if cutoff is None:
        total = sum(numeric_vals)
    else:
        total = sum(v for v in numeric_vals if v >= float(cutoff))

    # return integer if it's whole
    if abs(total - round(total)) < 1e-9:
        return int(round(total))
    return float(total)



def try_simple_compute(page_html: str, downloads_dir: Optional[str] = None) -> Optional[float]:
    """
    Attempt to compute common answers from the HTML page:
    - read any tables and sum a "value" column
    - if CSV link present, download and compute on that
    Returns a number or None.
    """
    downloads_dir = downloads_dir or DOWNLOAD_DIR

    # 1) try reading HTML tables for a column named 'value' or numeric columns
    try:
        # pandas warns about literal html in future; wrap in StringIO if needed
        from io import StringIO
        tables = pd.read_html(StringIO(page_html))
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if "value" in cols:
                try:
                    s = pd.to_numeric(t["value"], errors="coerce").dropna().sum()
                    return float(s)
                except Exception:
                    pass
            # fallback: sum first numeric column
            for c in t.columns:
                ser = pd.to_numeric(t[c], errors="coerce").dropna()
                if len(ser) > 0:
                    return float(ser.sum())
    except Exception:
        pass

    # 2) look for CSV links in HTML
    m = re.search(r'href=["\']([^"\']+\.csv[^"\']*)["\']', page_html, flags=re.I)
    if m:
        csv_url = m.group(1)
        # relative URL? return None and let worker handle download
        return None

    # 3) nothing found
    return None


# -------------------------
# Placeholder for LLM assistance (safe)
# -------------------------
def llm_solve_if_needed(prompt_text: str, files: Optional[list] = None) -> Optional[str]:
    """
    Placeholder: try lightweight heuristics first (try_simple_compute), then
    optionally call an LLM if you wire OpenAI calls separately.
    This function returns a candidate string answer (not always the final numeric answer).
    """
    logger.info("llm_solve_if_needed: called (placeholder)")
    # simple heuristic: if prompt contains 'sum' and numbers, compute
    maybe = extract_number_from_text(prompt_text)
    if maybe is not None:
        return str(maybe)
    return None
