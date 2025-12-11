import asyncio, logging, sys, traceback
from worker import handle_quiz_request

# configure logging to stdout so worker logs appear here
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

print('TEST_RUN STARTING - will run handle_quiz_request now', flush=True)

async def main():
    try:
        await handle_quiz_request(
            "24f2002804@ds.study.iitm.ac.in",
            "tsop93@69&t",
            "https://tds-llm-analysis.s-anand.net/demo"
        )
        print('TEST_RUN COMPLETED (no exception)', flush=True)
    except Exception as e:
        print('TEST_RUN EXCEPTION:', flush=True)
        traceback.print_exc()
        raise

if __name__ == '__main__':
    asyncio.run(main())
