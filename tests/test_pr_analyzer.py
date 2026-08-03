"""
tests/test_pr_analyzer.py
-------------------------
Unit tests for the PolyMentor automated PR diff reviewer and static analysis grounding layer.
"""

import pytest
from pathlib import Path
from scripts.analyze_pr import parse_pr_diff, analyze_hunks_static, DiffHunk, HunkAnalysisResult


SAMPLE_DIFF = """diff --git a/src/app.py b/src/app.py
index 1234567..89abcdef 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,5 @@
 def calculate():
-    return 0
+    x = [1, 2, 3]
+    return x[0]
diff --git a/src/index.js b/src/index.js
new file mode 100644
index 0000000..abcdef1
--- /dev/null
+++ b/src/index.js
@@ -0,0 +1,2 @@
+console.log("Welcome to PolyMentor");
"""

BUGGY_DIFF = """diff --git a/src/faulty.py b/src/faulty.py
index 111..222 100644
--- a/src/faulty.py
+++ b/src/faulty.py
@@ -1,3 +1,4 @@
 def handle_data():
+    eval("unsafe_command")
"""


class TestPRDiffParser:
    def test_parse_empty_diff(self):
        hunks = parse_pr_diff("")
        assert len(hunks) == 0

    def test_parse_multi_file_diff(self):
        hunks = parse_pr_diff(SAMPLE_DIFF)
        assert len(hunks) == 2
        
        py_hunk = hunks[0]
        assert py_hunk.file_path == "src/app.py"
        assert py_hunk.language == "python"
        assert "x = [1, 2, 3]" in py_hunk.added_code

        js_hunk = hunks[1]
        assert js_hunk.file_path == "src/index.js"
        assert js_hunk.language == "javascript"
        assert "console.log" in js_hunk.added_code


class TestPRStaticAnalysis:
    def test_analyze_clean_hunks(self):
        hunks = parse_pr_diff(SAMPLE_DIFF)
        results = analyze_hunks_static(hunks)
        assert len(results) == 2
        for hr in results:
            assert hr.has_bugs is False
            assert hr.quality_score >= 80

    def test_analyze_buggy_hunks(self):
        hunks = parse_pr_diff(BUGGY_DIFF)
        results = analyze_hunks_static(hunks)
        assert len(results) == 1
        faulty = results[0]
        assert faulty.has_bugs is True
        assert faulty.quality_score < 100
        errors = faulty.static_summary.get("errors", [])
        assert any("eval" in str(e) for e in errors)


class MockMentorResponse:
    def __init__(self, status="ok", answer="Looks good!", suspected_bugs=None, lesson="Keep learning", next_steps=None, fixed_code=None):
        self.status = status
        self.answer = answer
        self.suspected_bugs = suspected_bugs or []
        self.lesson = lesson
        self.next_steps = next_steps or []
        self.fixed_code = fixed_code
        self.model = "mock-llama-70b"
        self.elapsed_ms = 45.0
        self.tokens_used = 120


class MockPipeline:
    def __init__(self, response=None):
        self.response = response or MockMentorResponse()

    async def analyze(self, code, language, level, question=None):
        return self.response


from scripts.analyze_pr import truncate_diff_budget, extract_pr_repo_context, generate_pr_comment_md, run_pr_review
import json

class TestPRReviewPipeline:
    def test_truncate_diff_budget(self):
        hunks = parse_pr_diff(SAMPLE_DIFF)
        results = analyze_hunks_static(hunks)
        
        # Large budget
        packed, truncated, dropped = truncate_diff_budget(results, max_chars=10000)
        assert truncated is False
        assert len(dropped) == 0
        assert "src/app.py" in packed and "src/index.js" in packed
        
        # Tiny budget
        packed_tiny, truncated_tiny, dropped_tiny = truncate_diff_budget(results, max_chars=150)
        assert truncated_tiny is True
        assert len(dropped_tiny) > 0

    def test_generate_pr_comment_md_clean(self):
        hunks = parse_pr_diff(SAMPLE_DIFF)
        results = analyze_hunks_static(hunks)
        mock_resp = MockMentorResponse(answer="Architecture looks sound.")
        md = generate_pr_comment_md(mock_resp, results, [])
        assert "🟢 **Low Risk / Clean**" in md
        assert "File Quality Scorecard" in md
        assert "Architecture looks sound." in md

    def test_generate_pr_comment_md_buggy(self):
        hunks = parse_pr_diff(BUGGY_DIFF)
        results = analyze_hunks_static(hunks)
        mock_resp = MockMentorResponse(suspected_bugs=["Unsafe eval usage detected."])
        md = generate_pr_comment_md(mock_resp, results, [])
        assert "Critical Risk" in md or "Medium Risk" in md
        assert "Deterministic Static Analysis Findings" in md
        assert "Unsafe eval usage detected." in md

    @pytest.mark.asyncio
    async def test_run_pr_review_quality_gates(self, tmp_path):
        diff_file = tmp_path / "pr_diff.txt"
        diff_file.write_text(BUGGY_DIFF, encoding="utf-8")
        out_md = tmp_path / "pr_comment.md"
        out_json = tmp_path / "summary.json"
        
        mock_pipeline = MockPipeline()
        
        # With fail_on_bugs=True
        exit_code, summary = await run_pr_review(
            diff_file=str(diff_file),
            output_md=str(out_md),
            json_summary=str(out_json),
            fail_on_bugs=True,
            pipeline=mock_pipeline
        )
        assert exit_code == 2
        assert summary["total_static_errors"] > 0
        assert out_md.exists()
        assert out_json.exists()
        
        saved_json = json.loads(out_json.read_text(encoding="utf-8"))
        assert saved_json["status"] == "ok"
        assert saved_json["total_static_errors"] > 0

