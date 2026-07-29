#!/usr/bin/env python3
"""
analyze_pr.py
-------------
Analyzes a Pull Request diff using PolyMentor's Groq pipeline and generates a markdown comment.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.pipeline import PolyMentorPipeline

async def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_pr.py <path_to_diff>")
        sys.exit(1)

    diff_path = Path(sys.argv[1])
    if not diff_path.exists():
        print(f"Diff file not found: {diff_path}")
        sys.exit(1)

    diff_content = diff_path.read_text(encoding="utf-8")
    if not diff_content.strip():
        Path("pr_comment.md").write_text("No changes found in the diff.")
        sys.exit(0)
        
    # Optional: Truncate diff if it's too massive for the model context
    # Groq's llama-3.3-70b-versatile has a very large context, but just in case
    max_chars = 30000 
    if len(diff_content) > max_chars:
        diff_content = diff_content[:max_chars] + "\n... [Diff truncated due to size limit]"

    print(f"Analyzing PR diff ({len(diff_content)} chars)...")
    
    pipeline = PolyMentorPipeline.from_groq()
    
    question = (
        "You are an expert code reviewer. Please review this pull request diff. "
        "Identify any bugs, security vulnerabilities, or performance issues. "
        "Suggest improvements and provide a brief summary of the changes."
    )
    
    result = await pipeline.analyze(
        code=diff_content,
        language="diff",
        level="advanced",
        question=question
    )

    if result.status != "ok":
        Path("pr_comment.md").write_text(f"PolyMentor encountered an error: {result.status}\n{result.answer}")
        sys.exit(0)

    comment_lines = []
    comment_lines.append("## 🤖 PolyMentor PR Review\n")
    comment_lines.append(result.answer)
    
    if result.suspected_bugs:
        comment_lines.append("\n### 🐛 Suspected Bugs")
        for bug in result.suspected_bugs:
            comment_lines.append(f"- {bug}")
            
    if result.lesson:
        comment_lines.append(f"\n### 💡 Key Takeaway\n{result.lesson}")
        
    if result.next_steps:
        comment_lines.append("\n### 🎯 Suggested Action Items")
        for step in result.next_steps:
            comment_lines.append(f"- {step}")
            
    if result.fixed_code:
        comment_lines.append(f"\n### ✨ Suggested Fix\n```diff\n{result.fixed_code}\n```")
        
    comment_lines.append(f"\n---\n*Analyzed with {result.model} in {result.elapsed_ms:.0f}ms*")
    
    out_path = Path("pr_comment.md")
    out_path.write_text("\n".join(comment_lines), encoding="utf-8")
    print(f"Successfully wrote PR comment to {out_path.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
