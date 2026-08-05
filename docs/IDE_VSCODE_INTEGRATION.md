# PolyMentor IDE & VS Code Integration Guide

PolyMentor provides a modular, deterministic static analysis and grounded AI mentoring bridge specifically designed for Language Server Protocol (LSP) adapters, VS Code extensions, and IDE linters.

---

## 🚀 Quick Setup (VS Code Workspace)

To integrate PolyMentor directly into your local Visual Studio Code workspace without installing custom extension packages, adopt the preconfigured templates located in `ide_presets/vscode/`:

1. Copy `ide_presets/vscode/settings.json` to your local `.vscode/settings.json`.
2. Copy `ide_presets/vscode/tasks.json` to your local `.vscode/tasks.json`.
3. Press **Ctrl+Shift+B** (or run `Tasks: Run Build Task` from the command palette) to run real-time static analysis on your active editor file or across your entire workspace!

---

## 🛠️ `scripts/ide_bridge.py` API Reference

The primary interface for IDE clients is `scripts/ide_bridge.py`. It communicates over standard CLI arguments and standard input/output streams using JSON payloads formatted according to industry-standard LSP specification.

### 1. Single-File Diagnostics (LSP Format)
Analyze an on-disk source file and produce 0-indexed diagnostic ranges suitable for VS Code editor squiggly underlines:

```bash
python scripts/ide_bridge.py --file src/example.py --format lsp
```

### 2. Live Buffer Stream via Standard Input (Zero-Disk Write)
For real-time as-you-type linting inside an IDE editor, pipe the raw buffer via `stdin` and provide the programming language identifier:

```bash
cat << 'EOF' | python scripts/ide_bridge.py --stdin-lang javascript --format lsp
function calculate(x, y) {
    if (y = 0) { console.log("Assignment inside condition!"); }
    return x / y;
}
EOF
```

### 3. Workspace Batch Scanning with Quality Score Gates
Scan an entire repository folder, aggregate quality scores, and assert a continuous integration quality gate:

```bash
python scripts/ide_bridge.py --dir src/ --format lsp --min-score 80.0
```
*Note: If the average file quality score falls below `--min-score`, the bridge terminates with exit code `2` to halt automated CI or push checks.*

---

## 📦 LSP Diagnostic JSON Structure

The JSON response emitted by `ide_bridge.py` follows standard diagnostic mapping:

```json
{
  "uri": "src/example.py",
  "language": "python",
  "quality_score": 92.5,
  "total_issues": 1,
  "diagnostics": [
    {
      "range": {
        "start": { "line": 14, "character": 4 },
        "end": { "line": 14, "character": 84 }
      },
      "severity": 2,
      "code": "security_issue",
      "source": "polymentor-analyzer",
      "message": "Unsafe subprocess execution detected | 💡 Refactor Advice: Pass shell=False or use validated lists",
      "data": {
        "code_snippet": "subprocess.run(cmd, shell=True)",
        "polymentor_rule": "security_issue"
      }
    }
  ]
}
```

### Severity Mappings
| PolyMentor Severity | LSP Integer Code | LSP Diagnostic Severity |
| :--- | :---: | :--- |
| **CRITICAL / ERROR** | `1` | `DiagnosticSeverity.Error` |
| **HIGH / WARNING** | `2` | `DiagnosticSeverity.Warning` |
| **MEDIUM / INFO** | `3` | `DiagnosticSeverity.Information` |
| **LOW / HINT** | `4` | `DiagnosticSeverity.Hint` |
