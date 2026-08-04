"""
scripts/ci_fail_triage.py
-------------------------
Triages pytest failures in CI by analyzing the error logs 
with the PolyMentor Groq pipeline and generating a helpful PR comment.
"""

import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import PolyMentorPipeline

async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ci_fail_triage.py <path_to_pytest_output>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: Log file not found -> {log_path}")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        # Read the last 150 lines to avoid massive prompt context limits,
        # but capture the most important traceback info at the end.
        lines = f.readlines()
        log_content = "".join(lines[-150:])

    if not log_content.strip():
        print("Empty log file.")
        sys.exit(0)

    pipeline = PolyMentorPipeline()
    if not pipeline.api_key:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    message = (
        "The following is the output from a failed `pytest` run in our CI pipeline. "
        "Please analyze the traceback, explain exactly what caused the tests to fail, "
        "and provide actionable suggestions or a code fix to resolve the failure.\n\n"
        f"```text\n{log_content}\n```"
    )

    response = await pipeline.chat(
        message=message,
        code="",
        language="python",
        level="intermediate"
    )

    if response.status == "ok":
        print(f"## 🚨 CI Test Failure Triage\n")
        print(f"I noticed your tests are failing. Here's my analysis of the error logs:\n")
        print(f"### 📝 Explanation:")
        print(response.answer)
        
        if response.fixed_code:
            print(f"\n### ✨ Suggested Fix:")
            print(f"```python\n{response.fixed_code}\n```")
            
        if response.suspected_bugs:
            print("\n### ⚠️ Suspected Root Causes:")
            for bug in response.suspected_bugs:
                print(f"- {bug}")
    else:
        print("## 🚨 CI Test Failure\n")
        print("Tests failed, but I encountered an error trying to analyze the logs.")
        print(f"> {response.status}: {response.answer}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
