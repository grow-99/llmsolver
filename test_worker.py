import asyncio
from worker import handle_quiz_request

# run the worker manually (same values FastAPI uses)
asyncio.run(handle_quiz_request(
    "24f2002804@ds.study.iitm.ac.in",
    "tsop93@69&t",
    "https://tds-llm-analysis.s-anand.net/demo"
))
