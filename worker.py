# worker.py
import os
import time
import logging
import asyncio
from typing import Optional
from urllib.parse import urlparse

import httpx

from utils import (
    render_page_get_text,
    download_file,
    transcribe_with_whisper_local,
    try_simple_compute,
    find_first_csv_file_in_downloads,
    extract_number_from_text,
    compute_answer_from_csv,
)

logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)

# Global time limit (seconds) from initial POST — must be < 180
OVERALL_TIMEOUT = 170


async def _post_json(url: str, payload: dict, timeout: int = 30) -> Optional[dict]:
    """
    Simple POST JSON helper. We assume `url` is a proper absolute URL.
    """
    logger.info(f"_post_json: POST {url!r} with keys {list(payload.keys())}")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, follow_redirects=True)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"_http_error": False, "raw_text": resp.text}
    except httpx.HTTPStatusError as e:
        try:
            body = e.response.text
        except Exception:
            body = None
        logger.warning(f"_post_json http status error {e.response.status_code}: {body}")
        return {"_http_error": True, "status_code": e.response.status_code, "body": body}
    except Exception:
        logger.exception("_post_json failed")
        return {"_http_error": True, "message": "post_failed"}


async def handle_quiz_request(email: str, secret: str, start_url: str):
    """
    Main worker entrypoint. Called as a background task by main.py.
    Follows the quiz sequence until there is no next URL or timeout.
    """
    logger.info(f"Worker START: email={email!r} url={start_url!r}")
    logger.info(f"Solving URL: {start_url}")
    deadline = time.time() + OVERALL_TIMEOUT
    current_url = start_url

    try:
        while current_url and time.time() < deadline:
            logger.info(f"Solving URL: {current_url}")

            # 1) Render page and get HTML
            try:
                page_html = await render_page_get_text(current_url, timeout=30)
            except Exception:
                logger.exception("render_page_get_text failed")
                break

            # 2) quick deterministic compute from HTML (tables etc.)
            numeric_answer = try_simple_compute(page_html)
            candidate_secret = None

            # 3) find and download CSV if linked
            import re
            csv_links = []
            for m in re.finditer(r'href=["\']([^"\']+\.(?:csv|txt|tsv))["\']', page_html, flags=re.I):
                href = m.group(1)
                if href.startswith("http"):
                    csv_links.append(href)
            downloaded_csv = None
            if csv_links:
                csv_url = csv_links[0]
                downloaded_csv = await download_file(csv_url)
                # ---------------------------------------------------------
# CSV-FIRST: compute numeric answer from CSV sum
                numeric_answer = None
                try:
                    numeric_answer = compute_answer_from_csv(downloaded_csv)
                    logger.info(f"compute_answer_from_csv returned: {numeric_answer} (from {downloaded_csv})")
                except Exception:
                    logger.exception("compute_answer_from_csv failed")
# ---------------------------------------------------------

            logger.info(f"downloaded CSV: {downloaded_csv}")

            # 4) find audio links
            audio_links = []
            for m in re.finditer(r'src=["\']([^"\']+\.(?:opus|mp3|wav|m4a))["\']', page_html, flags=re.I):
                src = m.group(1)
                if src.startswith("http"):
                    audio_links.append(src)

            transcription_text = None
            if audio_links:
                audio_url = audio_links[0]
                logger.info(f"Found audio links: {audio_links}")
                audio_path = await download_file(audio_url)
                if audio_path:
                    logger.info(f"Downloaded audio to {audio_path}, transcribing...")
                    transcription_text = await asyncio.to_thread(
                        transcribe_with_whisper_local, audio_path
                    )
                    logger.info(f"Local transcription result: {transcription_text}")

            # 5) If transcription looks like an instruction, use CSV + cutoff to compute answer
            if transcription_text:
                lower = (transcription_text or "").lower()
                if any(k in lower for k in ("csv", "cutoff", "column", "sum", "download")):
                    if not downloaded_csv:
                        downloaded_csv = find_first_csv_file_in_downloads()
                        logger.info(f"found recent csv in downloads: {downloaded_csv}")
                    cutoff = extract_number_from_text(transcription_text)
                    if downloaded_csv:
                        numeric_answer = compute_answer_from_csv(downloaded_csv, cutoff=cutoff)
                        logger.info(f"compute_answer_from_csv returned: {numeric_answer}")
                    else:
                        logger.info("No CSV available to compute from transcription instruction")
                else:
                    candidate_secret = transcription_text.strip()

            # 6) If still no numeric answer but CSV exists, try compute from CSV only
            if numeric_answer is None and downloaded_csv:
                numeric_answer = compute_answer_from_csv(downloaded_csv)
                logger.info(f"compute_answer_from_csv (no explicit cutoff) returned: {numeric_answer}")

            # 7) Fallback: try to extract a useful number from HTML if needed
            if numeric_answer is None:
                numeric_answer = extract_number_from_text(page_html)
                if numeric_answer is not None:
                    logger.info(f"extract_number_from_text returned: {numeric_answer}")

            # 8) Build payload
            # --- normalize numeric answers before building payload ---
                if numeric_answer is not None and isinstance(numeric_answer, float) and numeric_answer.is_integer():
                    numeric_answer = int(numeric_answer)
# ----------------------------------------------------------

                payload = {"email": email, "secret": secret, "url": current_url}
            if numeric_answer is not None:
                payload["answer"] = numeric_answer
            elif candidate_secret:
                payload["answer"] = candidate_secret
            else:
                payload["answer"] = "no-answer-found"

            # 9) Determine submit URL — for demo, always same domain as current_url
            try:
                p = urlparse(current_url)
                submit_url = f"{p.scheme}://{p.netloc}/submit"
                logger.info(f"Using submit_url {submit_url!r} (host={p.netloc!r})")
            except Exception:
                submit_url = None

            if not submit_url:
                logger.warning("No submit URL could be determined; aborting this page")
                break

            # <-- ADDITIONAL LOGGING: show exact payload and answer type before posting
            logger.info(f"Submitting payload (answer type={type(payload.get('answer'))}): {payload}")

            # 10) POST the payload
            resp = await _post_json(submit_url, payload, timeout=30)
            logger.info(f"Submit response: {resp}")

            # 11) Interpret response
            next_url = None
            if isinstance(resp, dict):
                if resp.get("correct") is True:
                    next_url = resp.get("url")
                elif resp.get("correct") is False:
                    reason = resp.get("reason")
                    next_url = resp.get("url")
                    delay = resp.get("delay", None)
                    logger.info(f"Server reported incorrect answer: {reason}; next url: {next_url}; delay: {delay}")
                    if delay:
                        await asyncio.sleep(float(delay))
                elif resp.get("_http_error"):
                    logger.warning(f"Submit HTTP error struct: {resp}")
                    # can't do much more here
                    break
                else:
                    next_url = resp.get("url") or None
            else:
                logger.warning(f"Unexpected submit response type: {type(resp)}; raw: {resp}")

            if not next_url:
                logger.info(f"Finished processing {start_url}")
                break

            current_url = next_url
    except Exception:
        logger.exception("handle_quiz_request failed")
    finally:
        logger.info(f"Finished processing {start_url}")
