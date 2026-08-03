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
