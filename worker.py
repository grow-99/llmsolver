# worker.py
import sys
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import os
import time
import logging
import asyncio
from dotenv import load_dotenv

import httpx

from utils import (
    render_page_get_text,
    extract_submit_info,
    download_file_if_needed,
    try_simple_compute,
    llm_solve_if_needed,
    extract_answer_from_page,
    find_audio_links,
    download_audio,
    transcribe_audio_path,
    transcribe_audio_via_openai,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

OVERALL_TIMEOUT = 170  # seconds


async def handle_quiz_request(email: str, secret: str, url: str):
    start_time = time.time()
    cur_url = url
    current_secret = secret
    session = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    try:
        while cur_url and (time.time() - start_time) < OVERALL_TIMEOUT:
            logger.info("Solving URL: %s", cur_url)

            # Render page
            try:
                page_html, page_text = await render_page_get_text(cur_url, wait_for=5.0)
            except Exception as e:
                logger.exception("render_page_get_text failed for %s: %s", cur_url, e)
                break

            # Extract submit info
            try:
                submit_info = extract_submit_info(page_html, cur_url, page_text=page_text)
            except Exception as e:
                logger.exception("extract_submit_info failed: %s", e)
                break

            if not submit_info:
                logger.warning("Submit info not found. Stopping.")
                break

            submit_url = submit_info.get("submit_url")
            if not submit_url:
                logger.warning("No submit_url found. submit_candidates=%s", submit_info.get("submit_candidates"))
                break

            # Download linked files
            linked_files = submit_info.get("linked_files", [])
            downloaded = []
            for f_url in linked_files:
                try:
                    p = await download_file_if_needed(f_url, session=session)
                    if p:
                        downloaded.append(p)
                except Exception as e:
                    logger.exception("Failed to download %s: %s", f_url, e)

            # Heuristics
            try:
                answer = try_simple_compute(page_html, page_text, cur_url)
            except Exception as e:
                logger.exception("try_simple_compute raised: %s", e)
                answer = None

            # LLM fallback
            if answer is None:
                try:
                    answer = await llm_solve_if_needed(page_text, downloaded)
                except Exception as e:
                    logger.exception("LLM solver failed: %s", e)
                    answer = None

            # Explicit extraction
            candidate_secret = None
            if answer is None:
                try:
                    extracted = extract_answer_from_page(page_html, page_text)
                    if extracted is not None:
                        answer = extracted
                        logger.info("Extracted answer from page: %s", str(answer))
                except Exception as e:
                    logger.exception("extract_answer_from_page failed: %s", e)

            # AUDIO: detect and download audio; prefer cloud transcription if key present
            audio_transcript = None
            try:
                audio_links = find_audio_links(page_html, cur_url)
                if audio_links:
                    logger.info("Found audio links: %s", audio_links)
                    for al in audio_links:
                        try:
                            audio_path = await download_audio(al, session=session)
                            if audio_path:
                                logger.info("Downloaded audio to %s, transcribing...", audio_path)
                                # prefer cloud transcription if OPENAI_API_KEY present
                                txt = None
                                if os.getenv("OPENAI_API_KEY"):
                                    try:
                                        txt = await asyncio.to_thread(transcribe_audio_via_openai, audio_path)
                                        logger.info("OpenAI transcription (cloud) result: %s", txt)
                                    except Exception as e:
                                        logger.exception("OpenAI transcription failed: %s", e)
                                # fallback to local transcription
                                if not txt:
                                    try:
                                        txt = await asyncio.to_thread(transcribe_audio_path, audio_path)
                                        logger.info("Local transcription (fallback) result: %s", txt)
                                    except Exception as e:
                                        logger.exception("Local transcription failed: %s", e)
                                if txt:
                                    audio_transcript = txt
                                    break
                        except Exception as e:
                            logger.exception("Audio download/transcribe failed for %s: %s", al, e)
            except Exception:
                pass

            # If transcription found and no answer, take it as candidate secret
            if not answer and audio_transcript:
                candidate_secret = audio_transcript.strip()
                logger.info("Using audio transcription as candidate_secret: %s", candidate_secret)

            # If extracted answer is non-empty string, it may be a candidate secret
            if isinstance(answer, str) and answer.strip():
                candidate_secret = answer.strip()

            if answer is None:
                answer = ""

            payload = {"email": email, "secret": current_secret, "url": cur_url, "answer": answer}
            logger.info("Submitting to %s with payload keys: %s", submit_url, list(payload.keys()))

            # Submit
            try:
                resp = await session.post(submit_url, json=payload, timeout=httpx.Timeout(60.0))
            except Exception as e:
                logger.exception("Submit POST failed: %s", e)
                break

            if resp.status_code not in (200, 201):
                logger.warning("Submit returned status %s; body: %s", resp.status_code, getattr(resp, "text", "<no body>"))
                break

            try:
                body = resp.json()
            except Exception as e:
                logger.exception("Invalid JSON response from submit endpoint: %s", e)
                break

            logger.info("Submit response: %s", body)

            # Handle secret mismatch: follow grader-provided next URL if present (honor delay), else one-time retry with candidate_secret
            if body.get("correct") is False and isinstance(body.get("reason"), str) and "Secret mismatch" in body.get("reason"):
                next_url = body.get("url")
                delay = body.get("delay") or 0
                if next_url:
                    try:
                        delay_val = float(delay)
                    except Exception:
                        delay_val = 0
                    logger.info("Server reported Secret mismatch and provided next URL. Waiting %s seconds then following %s", delay_val, next_url)
                    if delay_val > 0:
                        await asyncio.sleep(delay_val)
                    cur_url = next_url
                    continue

                # No next URL -> retry once with candidate_secret if useful
                cand = None
                if candidate_secret:
                    cand = " ".join(candidate_secret.split()).strip()
                if cand and cand != current_secret:
                    logger.info("Secret mismatch and no next URL. Retrying submit once with candidate secret.")
                    current_secret = cand
                    payload_retry = {"email": email, "secret": current_secret, "url": cur_url, "answer": answer}
                    try:
                        resp2 = await session.post(submit_url, json=payload_retry, timeout=httpx.Timeout(60.0))
                    except Exception as e:
                        logger.exception("Retry submit POST failed: %s", e)
                        break

                    if resp2.status_code not in (200, 201):
                        logger.warning("Retry submit returned status %s; body: %s", resp2.status_code, getattr(resp2, "text", "<no body>"))
                        break

                    try:
                        body2 = resp2.json()
                    except Exception as e:
                        logger.exception("Invalid JSON on retry response: %s", e)
                        break

                    logger.info("Retry submit response: %s", body2)
                    body = body2
                else:
                    logger.warning("Secret mismatch: no usable candidate secret to retry and no next URL. Stopping.")
                    break

            # If correct and next URL, follow it
            if body.get("correct") is True and body.get("url"):
                cur_url = body["url"]
                await asyncio.sleep(0.1)
                continue
            else:
                break

    finally:
        try:
            await session.aclose()
        except Exception:
            pass
        logger.info("Finished processing %s", url)
