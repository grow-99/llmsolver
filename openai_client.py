# openai_client.py
import os
import json
import logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("openai_client")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM calls disabled.")

async def llm_solve_if_needed(page_text: str, downloaded_files: list):
    if not OPENAI_API_KEY:
        return None
    import openai
    openai.api_key = OPENAI_API_KEY
    prompt = f"""
You are a helpful assistant. Read the page content below and determine the quiz answer.
Page content:
\"\"\"{page_text[:3000]}\"\"\"
Downloaded files: {downloaded_files}
Return a JSON object like {{ "answer": 12345 }} or {{ "answer": null, "explain":"..." }}.
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a careful data assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0
        )
        text = resp["choices"][0]["message"]["content"]
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0)).get("answer")
        n = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if n:
            return float(n.group(0))
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
    return None
