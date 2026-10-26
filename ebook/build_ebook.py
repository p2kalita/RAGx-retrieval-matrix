"""
NyayaRAG ebook builder.

Builds a single self-contained HTML ebook from the five project stages
(notebook_1 ... notebook_5). Each .py script becomes one section made of its
docstring explanation and its code. Outputs, figures and tables come from the
executed notebooks in notebooks/ and from saved artefacts in evaluation/,
models/, benchmark/ and data/. Error outputs are dropped.

Usage:
    python ebook/build_ebook.py
"""

import ast
import base64
import html
import io
import json
import re
from pathlib import Path

import markdown
import nbformat
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "ebook" / "NyayaRAG_ebook.html"

MATCH_THRESHOLD = 0.2
MAX_OUTPUT_LINES = 60
CSV_PREVIEW_ROWS = 20
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"

CHAPTERS = [
    {
        "dir": "notebook_1",
        "notebook": "notebooks/01_NYAYaRAG_Data_Preparation_and_Indexing.ipynb",
        "title": "Data Preparation, Corpus Analysis, Chunking and Retrieval Indexing",
        "extra_md": [],
    },
    {
        "dir": "notebook_2",
        "notebook": "notebooks/02_NYAYaRAG_Model_Evaluation_and_Statistical_Analysis.ipynb",
        "title": "Research Questions, Statistical Plan and Sample Size Determination",
        "extra_md": ["notebook_2/STATISTICAL_EVALUATION_GUIDE.md"],
    },
    {
        "dir": "notebook_3",
        "notebook": "notebooks/03_NyayaRAG_Pilot_Questions_Generatated_simulated.ipynb",
        "title": "LLM-Judge Evaluation of Pilot Benchmark Questions",
        "extra_md": ["notebook_3/notebook_3_literatureReview.md"],
    },
    {
        "dir": "notebook_4",
        "notebook": "notebooks/04_NyayaRAG_Pilot_Evaluation_and_Pipeline_Validation.ipynb",
        "title": "Pilot Retrieval Evaluation and Statistical Analysis (RQ1 and RQ2)",
        "extra_md": [],
    },
    {
        "dir": "notebook_5",
        "notebook": "notebooks/05_NyayaRAG_Generation_and_Evaluation_Qwen.ipynb",
        "title": "Evidence-Grounded Generation and Safe Abstention (RQ3 and RQ4)",
        "extra_md": [],
    },
]

# Script (chapter dir / filename prefix) -> repository artefacts it produces or uses.
ARTIFACTS = {
    ("notebook_1", "04"): ["data/processed/extraction_log.csv"],
    ("notebook_1", "10"): ["models/e5/e5_config.json"],
    ("notebook_1", "11"): ["models/bge/bge_config.json"],
    ("notebook_1", "12"): ["models/inlegalbert/inlegalbert_config.json"],
    ("notebook_1", "13"): ["models/retrieval_index_metadata.json"],
    ("notebook_3", "01"): ["evaluation/nyayarag_reconstructed_questions.csv"],
    ("notebook_3", "05"): ["evaluation/nyayarag_llm_judge_evaluation.csv"],
    ("notebook_3", "06"): ["evaluation/nyayarag_metric_means.png"],
    ("notebook_3", "07"): ["evaluation/nyayarag_questions_for_manual_review.csv"],
    ("notebook_3", "08"): ["evaluation/nyayarag_candidate_comparison.csv"],
    ("notebook_4", "01"): ["benchmark/final/pilot_questions.csv"],
    ("notebook_4", "02"): ["evaluation/pilot/retrieval/tfidf_pilot_retrieval.csv"],
    ("notebook_4", "03"): ["evaluation/pilot/annotation/tfidf_pilot_annotations.csv"],
    ("notebook_4", "04"): ["evaluation/pilot/retrieval/bm25_pilot_retrieval.csv",
                           "evaluation/pilot/annotation/bm25_pilot_annotations.csv"],
    ("notebook_4", "05"): ["evaluation/pilot/retrieval/e5_pilot_retrieval.csv",
                           "evaluation/pilot/annotation/e5_pilot_annotations.csv"],
    ("notebook_4", "06"): ["evaluation/pilot/retrieval/bge_pilot_retrieval.csv",
                           "evaluation/pilot/annotation/bge_pilot_annotations.csv"],
    ("notebook_4", "07"): ["evaluation/pilot/retrieval/inlegalbert_pilot_retrieval.csv",
                           "evaluation/pilot/annotation/inlegalbert_pilot_annotations.csv"],
    ("notebook_4", "08"): ["evaluation/pilot/retrieval/hybrid_pilot_retrieval.csv",
                           "evaluation/pilot/annotation/hybrid_pilot_annotations.csv"],
    ("notebook_5", "05"): ["evaluation/pilot/generation_gemini/gemini35_pilot_generation_results.csv"],
    ("notebook_5", "06"): ["evaluation/pilot/generation_gemini/gemini35_pilot_generation_review.csv"],
    ("notebook_5", "07"): ["evaluation/pilot/generation_gemini/gemini35_rq3_manual_annotation.csv"],
    ("notebook_5", "09"): ["evaluation/pilot/generation_gemini/pilot_final_results.csv"],
}

STATS = {}


# =============================================================================
# Text helpers
# =============================================================================

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def md_to_html(text):
    return markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5"
    )


BULLET_RE = re.compile(r"^([*\-•]|\d+[.)])\s+")
NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+([A-Z].{0,90})$")
TOP_HEADING_RE = re.compile(r"^\d+\.\s+[A-Z][\w&/()-]*(?:\s+[\w&/()-]+){0,6}$")
SECTION_HEADING_RE = re.compile(r"^(Section|Part|Step)\s+[\w.]+\s*[:—–-].{0,120}$")


def normalise_docstring(doc):
    """Turn a plain-text docstring into Markdown."""
    out = []
    in_table = False
    prev_kind = None  # "bullet", "text", "blank", "heading"
    indent_stack = []
    raw_lines = doc.splitlines()
    for pos, raw in enumerate(raw_lines):
        line = raw.rstrip()
        next_blank = pos + 1 >= len(raw_lines) or not raw_lines[pos + 1].strip()
        after_item = pos > 0 and bool(BULLET_RE.match(raw_lines[pos - 1].strip()))
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if re.fullmatch(r"[=\-_~*]{5,}", stripped):
            continue
        if not stripped:
            out.append("")
            prev_kind = "blank"
            in_table = False
            continue
        if stripped.startswith("|"):
            if not in_table and out and out[-1] != "":
                out.append("")
            out.append(stripped)
            in_table = True
            prev_kind = "table"
            continue
        if stripped.startswith("#"):
            out.extend(["", "#" + stripped, ""])
            prev_kind = "heading"
            continue

        m = NUM_HEADING_RE.match(stripped)
        if indent == 0 and (m or SECTION_HEADING_RE.match(stripped) or (TOP_HEADING_RE.match(stripped) and next_blank and not after_item)):
            level = "####" if m and m.group(1).count(".") >= 2 else "###"
            out.extend(["", f"{level} {stripped}", ""])
            prev_kind = "heading"
            continue

        if BULLET_RE.match(stripped):
            marker = BULLET_RE.match(stripped).group(1)
            body = stripped[len(BULLET_RE.match(stripped).group(0)):]
            if marker[0].isdigit():
                item = f"{marker[:-1]}. {body}"
            else:
                item = f"- {body}"
            if prev_kind not in ("bullet", None):
                indent_stack = []
                if out and out[-1] != "":
                    out.append("")
            # Nesting depth follows the relative indentation of the bullets.
            while indent_stack and indent_stack[-1] > indent:
                indent_stack.pop()
            if not indent_stack or indent_stack[-1] < indent:
                indent_stack.append(indent)
            out.append("    " * (len(indent_stack) - 1) + item)
            prev_kind = "bullet"
            continue

        if prev_kind == "bullet" and indent > 0:
            out[-1] += " " + stripped
            continue
        if prev_kind == "text" and out:
            out[-1] += " " + stripped
            continue

        if stripped.endswith(":") and len(stripped) < 60:
            if prev_kind not in ("blank", None):
                out.append("")
            out.extend([f"**{stripped}**", ""])
            prev_kind = "blank"
            continue

        if prev_kind == "bullet":
            out.append("")
        out.append(stripped)
        prev_kind = "text"
    return "\n".join(out)


def split_script(path):
    """Return (docstring, code_without_docstring)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree, clean=True) or ""
    code = src
    if doc and tree.body and isinstance(tree.body[0], ast.Expr):
        end = tree.body[0].end_lineno
        code = "\n".join(src.splitlines()[end:])
    return doc, code.strip("\n")


def strip_doc_title(doc):
    """Separate a leading title line (after banners) from the rest of the docstring."""
    lines = [l for l in doc.splitlines()]
    while lines and (not lines[0].strip() or re.fullmatch(r"[=\-_~*]{5,}", lines[0].strip())):
        lines.pop(0)
    if not lines:
        return "", ""
    title = lines[0].strip()
    rest = lines[1:]
    while rest and re.fullmatch(r"[=\-_~*]{5,}", rest[0].strip() or "-----"):
        if not rest[0].strip():
            rest.pop(0)
            continue
        rest.pop(0)
    return title, "\n".join(rest)


def pretty_title(filename):
    stem = re.sub(r"^\d+_", "", Path(filename).stem)
    return stem.replace("_", " ").title().replace("Rq", "RQ").replace("Llm", "LLM") \
        .replace("Tfidf", "TF-IDF").replace("Bm25", "BM25").replace("Bge", "BGE") \
        .replace("E5", "E5").replace("Rrf", "RRF").replace("Pdf", "PDF") \
        .replace("Inlegalbert", "InLegalBERT").replace("Eda", "EDA")


# =============================================================================
# Notebook matching
# =============================================================================

def code_lines(text):
    lines = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!") or s.startswith("%"):
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) > 3:
            lines.add(s)
    return lines


def overlap(cell_lines, script_lines):
    if not cell_lines:
        return 0.0
    # Containment of the cell in the script: scripts are much larger than cells.
    return len(cell_lines & script_lines) / len(cell_lines)


def as_text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


# =============================================================================
# Output rendering
# =============================================================================

class Counter:
    def __init__(self, chapter_no):
        self.chapter_no = chapter_no
        self.figures = 0
        self.tables = 0

    def fig(self):
        self.figures += 1
        return f"Figure {self.chapter_no}.{self.figures}"

    def tab(self):
        self.tables += 1
        return f"Table {self.chapter_no}.{self.tables}"


def truncate(text):
    lines = text.rstrip().splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES] + [f"… output truncated ({len(lines) - MAX_OUTPUT_LINES} more lines)"]
    return "\n".join(lines)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sanitise_html(fragment):
    fragment = re.sub(r"<script\b.*?</script>", "", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<style\b.*?</style>", "", fragment, flags=re.S | re.I)
    fragment = re.sub(r"\son\w+=\"[^\"]*\"", "", fragment)
    return fragment


def render_outputs(outputs, counter, chapter_stats):
    parts = []
    for out in outputs:
        otype = out.get("output_type")
        if otype == "error":
            chapter_stats["errors_dropped"] += 1
            continue
        if otype == "stream":
            if out.get("name") == "stderr":
                chapter_stats["stderr_dropped"] += 1
                continue
            text = ANSI_RE.sub("", as_text(out.get("text")))
            if "Traceback (most recent call last)" in text:
                chapter_stats["errors_dropped"] += 1
                continue
            if text.strip():
                parts.append(f'<pre class="output">{html.escape(truncate(text))}</pre>')
            continue
        data = out.get("data", {})
        if "application/vnd.plotly.v1+json" in data:
            spec = data["application/vnd.plotly.v1+json"]
            div_id = f"plotly-{counter.chapter_no}-{counter.figures + 1}"
            caption = counter.fig()
            payload = json.dumps({"data": spec.get("data", []), "layout": spec.get("layout", {})})
            payload = payload.replace("</", "<\\/")
            parts.append(
                f'<figure class="plot"><div id="{div_id}" class="plotly-div"></div>'
                f'<script type="application/json" data-plotly="{div_id}">{payload}</script>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
            chapter_stats["plotly"] += 1
            continue
        image = None
        for mime in ("image/png", "image/jpeg"):
            if mime in data:
                image = (mime, as_text(data[mime]).replace("\n", ""))
                break
        if image:
            caption = counter.fig()
            parts.append(
                f'<figure><img alt="{caption}" src="data:{image[0]};base64,{image[1]}">'
                f"<figcaption>{caption}</figcaption></figure>"
            )
            chapter_stats["figures"] += 1
            continue
        if "image/svg+xml" in data:
            caption = counter.fig()
            parts.append(f"<figure>{sanitise_html(as_text(data['image/svg+xml']))}<figcaption>{caption}</figcaption></figure>")
            chapter_stats["figures"] += 1
            continue
        if "text/html" in data:
            fragment = sanitise_html(as_text(data["text/html"]))
            if "<table" in fragment:
                caption = counter.tab()
                parts.append(f'<div class="table-wrap"><div class="caption">{caption}</div>{fragment}</div>')
                chapter_stats["tables"] += 1
            elif fragment.strip():
                parts.append(f'<div class="html-output">{fragment}</div>')
            continue
        if "text/markdown" in data:
            parts.append(f'<div class="html-output">{md_to_html(as_text(data["text/markdown"]))}</div>')
            continue
        if "text/plain" in data:
            text = ANSI_RE.sub("", as_text(data["text/plain"]))
            if text.strip() and not text.startswith("<IPython") and "Widget" not in text:
                parts.append(f'<pre class="output">{html.escape(truncate(text))}</pre>')
    return "\n".join(parts)


def render_artifact(rel_path, counter, chapter_stats):
    path = ROOT / rel_path
    if not path.exists():
        return ""
    label = f'<div class="source">Source: <code>{html.escape(rel_path)}</code></div>'
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode()
        caption = counter.fig()
        chapter_stats["figures"] += 1
        return (f'<figure><img alt="{caption}" src="data:{mime};base64,{b64}">'
                f"<figcaption>{caption} — {html.escape(path.name)}</figcaption>{label}</figure>")
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")
        preview = df.head(CSV_PREVIEW_ROWS).copy()
        for col in preview.select_dtypes(include="object"):
            preview[col] = preview[col].astype(str).map(lambda v: v if len(v) <= 220 else v[:220] + "…")
        caption = counter.tab()
        chapter_stats["tables"] += 1
        note = f"{len(df):,} rows × {len(df.columns)} columns"
        if len(df) > CSV_PREVIEW_ROWS:
            note += f" (first {CSV_PREVIEW_ROWS} shown)"
        return (f'<div class="table-wrap"><div class="caption">{caption} — {html.escape(path.name)} '
                f'<span class="muted">· {note}</span></div>'
                f'{preview.to_html(index=False, classes="dataframe", border=0, na_rep="")}{label}</div>')
    if path.suffix.lower() == ".json":
        text = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
        return (f'<details class="artifact"><summary>{html.escape(path.name)}</summary>'
                f'<pre class="output">{html.escape(truncate(text))}</pre>{label}</details>')
    return ""


# =============================================================================
# Chapter assembly
# =============================================================================

def build_chapter(no, spec, formatter):
    chapter_dir = ROOT / spec["dir"]
    scripts = sorted(chapter_dir.glob("*.py"), key=lambda p: (not p.name[:2].isdigit(), p.name))
    counter = Counter(no)
    chapter_stats = {"sections": 0, "figures": 0, "tables": 0, "plotly": 0, "errors_dropped": 0,
                     "stderr_dropped": 0, "cells_matched": 0, "cells_followed": 0, "cells_unmatched": 0}

    script_info = []
    for path in scripts:
        doc, code = split_script(path)
        script_info.append({"path": path, "doc": doc, "code": code,
                            "lines": code_lines(code), "cells": []})

    # Assign executed notebook cells (with outputs) to the best-matching script.
    unmatched = []
    last_matched = None
    nb_path = ROOT / spec["notebook"]
    if nb_path.exists():
        nb = nbformat.read(nb_path, as_version=4)
        cells = nb.cells
        for idx, cell in enumerate(cells):
            if cell.cell_type != "code" or not cell.get("outputs"):
                continue
            useful = [o for o in cell.outputs if o.get("output_type") != "error"]
            if not useful:
                chapter_stats["errors_dropped"] += len(cell.outputs)
                continue
            cl = code_lines(cell.source)
            scores = [overlap(cl, s["lines"]) for s in script_info]
            best = max(range(len(scores)), key=scores.__getitem__) if scores else None
            notes = []
            j = idx + 1
            while j < len(cells) and cells[j].cell_type == "markdown":
                notes.append(cells[j].source)
                j += 1
            entry = {"index": idx, "outputs": cell.outputs, "notes": notes}
            if best is not None and scores[best] >= MATCH_THRESHOLD:
                script_info[best]["cells"].append(entry)
                last_matched = best
                chapter_stats["cells_matched"] += 1
            elif last_matched is not None:
                # Display-only cells follow the code that produced them in notebook order.
                script_info[last_matched]["cells"].append(entry)
                chapter_stats["cells_followed"] += 1
            else:
                unmatched.append(entry)
                chapter_stats["cells_unmatched"] += 1

    chapter_id = f"ch{no}"
    toc = []
    body = [f'<section class="chapter" id="{chapter_id}">',
            f'<div class="chapter-label">Chapter {no}</div><h1>{html.escape(spec["title"])}</h1>',
            f'<p class="chapter-source">Source: <code>{spec["dir"]}/</code> · executed notebook '
            f'<code>{html.escape(spec["notebook"])}</code></p>']

    for sec_no, info in enumerate(script_info, start=1):
        path = info["path"]
        sec_title = pretty_title(path.name)
        sec_id = f"{chapter_id}-{slug(path.stem)}"
        toc.append((sec_id, f"{no}.{sec_no} {sec_title}"))
        chapter_stats["sections"] += 1

        doc_title, doc_rest = strip_doc_title(info["doc"])
        body.append(f'<section class="script" id="{sec_id}">')
        body.append(f'<h2><span class="secno">{no}.{sec_no}</span> {html.escape(sec_title)}</h2>')
        body.append(f'<div class="script-file"><code>{spec["dir"]}/{path.name}</code></div>')
        if doc_title:
            body.append(f'<p class="doc-title">{html.escape(doc_title)}</p>')
        if doc_rest.strip():
            body.append(f'<div class="explanation">{md_to_html(normalise_docstring(doc_rest))}</div>')

        n_lines = info["code"].count("\n") + 1
        body.append(f'<details class="code"><summary>Code · {n_lines} lines</summary>'
                    f'{highlight(info["code"], PythonLexer(), formatter)}</details>')

        result_parts = []
        for cell in info["cells"]:
            rendered = render_outputs(cell["outputs"], counter, chapter_stats)
            if rendered:
                result_parts.append(rendered)
            if cell["notes"]:
                notes_html = md_to_html("\n\n".join(cell["notes"]))
                result_parts.append(f'<div class="nb-notes">{notes_html}</div>')
        for rel in ARTIFACTS.get((spec["dir"], path.name[:2]), []):
            rendered = render_artifact(rel, counter, chapter_stats)
            if rendered:
                result_parts.append(rendered)
        if result_parts:
            body.append('<div class="results"><h3 class="results-title">Results and outputs</h3>')
            body.extend(result_parts)
            body.append("</div>")
        body.append("</section>")

    if unmatched:
        sec_no = len(script_info) + 1
        sec_id = f"{chapter_id}-additional-outputs"
        toc.append((sec_id, f"{no}.{sec_no} Additional Notebook Outputs"))
        body.append(f'<section class="script" id="{sec_id}">')
        body.append(f'<h2><span class="secno">{no}.{sec_no}</span> Additional Notebook Outputs</h2>')
        body.append('<p class="muted">Outputs from the executed notebook that do not map directly '
                    'onto a single script above, in notebook order.</p><div class="results">')
        for cell in unmatched:
            rendered = render_outputs(cell["outputs"], counter, chapter_stats)
            if rendered:
                body.append(rendered)
            if cell["notes"]:
                body.append(f'<div class="nb-notes">{md_to_html(chr(10).join(cell["notes"]))}</div>')
        body.append("</div></section>")

    for rel in spec["extra_md"]:
        md_path = ROOT / rel
        if not md_path.exists():
            continue
        sec_no = len(toc) + 1
        sec_title = pretty_title(md_path.name)
        sec_id = f"{chapter_id}-{slug(md_path.stem)}"
        toc.append((sec_id, f"{no}.{sec_no} {sec_title}"))
        body.append(f'<section class="script" id="{sec_id}">')
        body.append(f'<h2><span class="secno">{no}.{sec_no}</span> {html.escape(sec_title)}</h2>')
        body.append(f'<div class="script-file"><code>{rel}</code></div>')
        text = md_path.read_text(encoding="utf-8")
        text = re.sub(r"^\s*={5,}\s*$", "", text, flags=re.M)  # drop ASCII banner rules
        text = re.sub(r"^(#{1,2}) ", r"\1## ", text, flags=re.M)  # demote headings below h2
        body.append(f'<div class="explanation">{md_to_html(text)}</div></section>')

    body.append("</section>")
    STATS[spec["dir"]] = chapter_stats
    return chapter_id, toc, "\n".join(body)


# =============================================================================
# Page template
# =============================================================================

CSS = """
:root {
  --bg: #fbfaf7; --fg: #1f2328; --muted: #6b7076; --accent: #8a3b12; --rule: #e3dfd6;
  --panel: #f3f0e8; --code-bg: #f6f4ef; --sidebar: #f1ede4; --table-head: #ece7dc;
  --link: #8a3b12;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #17181b; --fg: #e4e2dd; --muted: #9a9ea5; --accent: #e59866; --rule: #2e3036;
    --panel: #1f2125; --code-bg: #1d1f23; --sidebar: #1b1c20; --table-head: #262830;
    --link: #e59866;
  }
}
:root[data-theme="dark"] {
  --bg: #17181b; --fg: #e4e2dd; --muted: #9a9ea5; --accent: #e59866; --rule: #2e3036;
  --panel: #1f2125; --code-bg: #1d1f23; --sidebar: #1b1c20; --table-head: #262830;
  --link: #e59866;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 Georgia, "Iowan Old Style", "Palatino Linotype", serif; }
a { color: var(--link); }
code, pre { font-family: "JetBrains Mono", Consolas, "SFMono-Regular", Menlo, monospace; font-size: 0.82em; }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); }
nav.toc { position: sticky; top: 0; height: 100vh; overflow-y: auto; background: var(--sidebar);
  border-right: 1px solid var(--rule); padding: 24px 18px; font: 14px/1.45 system-ui, sans-serif; }
nav.toc .brand { font-weight: 700; font-size: 18px; margin-bottom: 4px; }
nav.toc ol { list-style: none; padding: 0; margin: 0; }
nav.toc > ol > li { margin: 14px 0 4px; }
nav.toc > ol > li > a { font-weight: 600; color: var(--fg); text-decoration: none; }
nav.toc ol ol { margin: 4px 0 0 8px; }
nav.toc ol ol a { color: var(--muted); text-decoration: none; display: block; padding: 2px 0; }
nav.toc a:hover { color: var(--accent); }
nav.toc .theme { margin-top: 20px; }
nav.toc button { font: inherit; background: var(--panel); color: var(--fg); border: 1px solid var(--rule);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; }
main { padding: 48px clamp(16px, 5vw, 72px) 120px; max-width: 1080px; }
.cover { min-height: 80vh; display: flex; flex-direction: column; justify-content: center;
  border-bottom: 1px solid var(--rule); margin-bottom: 48px; }
.cover .kicker { font: 600 13px system-ui, sans-serif; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); }
.cover h1 { font-size: clamp(40px, 7vw, 72px); margin: 12px 0 8px; line-height: 1.05; }
.cover .subtitle { font-size: 22px; color: var(--muted); max-width: 720px; }
.cover .meta { margin-top: 32px; font: 14px system-ui, sans-serif; color: var(--muted); }
.cover ol { font-size: 18px; }
.chapter { padding-top: 32px; margin-top: 48px; border-top: 3px double var(--rule); }
.chapter-label { font: 600 13px system-ui, sans-serif; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); }
.chapter h1 { font-size: 38px; line-height: 1.15; margin: 6px 0 8px; }
.chapter-source, .script-file, .source, .muted { color: var(--muted); font-size: 14px; }
.script { margin-top: 56px; }
.script h2 { font-size: 26px; margin-bottom: 2px; }
.secno { color: var(--accent); margin-right: 6px; }
.doc-title { font-style: italic; color: var(--muted); margin-top: 8px; }
.explanation h3 { font-size: 20px; margin-top: 28px; }
.explanation h4 { font-size: 17px; }
details.code { margin: 20px 0; border: 1px solid var(--rule); border-radius: 8px; background: var(--code-bg); }
details.code > summary, details.artifact > summary { cursor: pointer; padding: 8px 14px;
  font: 600 13px system-ui, sans-serif; color: var(--muted); }
details.code .highlight { overflow-x: auto; margin: 0; border-top: 1px solid var(--rule); }
details.code pre { margin: 0; padding: 12px 14px; line-height: 1.5; }
.highlight table { border: 0; margin: 0; }
.highlight td { border: 0; padding: 0; background: transparent; }
.highlight .linenos { color: var(--muted); user-select: none; padding-right: 12px; text-align: right; }
.results { margin-top: 20px; border-left: 3px solid var(--accent); padding-left: 18px; }
.results-title { font: 600 13px system-ui, sans-serif; letter-spacing: .1em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 12px; }
pre.output { background: var(--panel); padding: 12px 14px; border-radius: 6px; overflow-x: auto;
  white-space: pre; line-height: 1.45; }
figure { margin: 24px 0; text-align: center; }
figure img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; border: 1px solid var(--rule); }
figcaption, .caption { font: 600 13px system-ui, sans-serif; color: var(--muted); margin: 8px 0; text-align: left; }
.table-wrap { overflow-x: auto; margin: 18px 0; }
table { border-collapse: collapse; font: 13px/1.4 system-ui, sans-serif; margin: 12px 0; }
th, td { border: 1px solid var(--rule); padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: var(--table-head); }
.explanation table { font-size: 14px; }
.nb-notes { background: var(--panel); border-radius: 8px; padding: 4px 18px; margin: 14px 0; font-size: 16px; }
.nb-notes h1, .nb-notes h2 { font-size: 20px; }
.nb-notes h3, .nb-notes h4 { font-size: 17px; }
.html-output { overflow-x: auto; }
.explanation p, .explanation li, .nb-notes p, .nb-notes li { overflow-wrap: anywhere; }
.explanation pre, .nb-notes pre { overflow-x: auto; background: var(--code-bg); padding: 12px 14px; border-radius: 6px; }
.explanation pre.mermaid, .nb-notes pre.mermaid { background: #fff; text-align: center; }
.plotly-div { width: 100%; min-height: 420px; }
details.artifact { margin: 14px 0; border: 1px solid var(--rule); border-radius: 8px; }
details.artifact .output, details.artifact .source { margin: 0 14px 12px; }
@media (max-width: 900px) {
  .layout { display: block; }
  nav.toc { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--rule); }
}
@media print {
  nav.toc { display: none; }
  .layout { display: block; }
  .chapter { page-break-before: always; }
  details.code { display: block; }
  figure, table { page-break-inside: avoid; }
}
"""

JS = """
(function () {
  var btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('nyayarag-theme', root.dataset.theme); } catch (e) {}
  });
  try { var t = localStorage.getItem('nyayarag-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  window.addEventListener('beforeprint', function () {
    document.querySelectorAll('details').forEach(function (d) { d.open = true; });
  });
  if (window.mermaid) {
    document.querySelectorAll('code.language-mermaid').forEach(function (c) {
      var pre = c.parentElement; pre.className = 'mermaid'; pre.textContent = c.textContent;
    });
    mermaid.initialize({startOnLoad: false, theme: 'neutral'});
    mermaid.run({querySelector: 'pre.mermaid'});
  }
  if (window.Plotly) {
    document.querySelectorAll('script[data-plotly]').forEach(function (s) {
      try {
        var spec = JSON.parse(s.textContent);
        Plotly.newPlot(s.dataset.plotly, spec.data, spec.layout, {responsive: true, displaylogo: false});
      } catch (e) { console.warn('Plotly render failed', e); }
    });
  }
})();
"""


def build():
    formatter = HtmlFormatter(linenos="table", cssclass="highlight", style="friendly")
    dark_formatter = HtmlFormatter(cssclass="highlight", style="monokai")
    pyg_css = formatter.get_style_defs(".highlight")
    dark_css = dark_formatter.get_style_defs(":root[data-theme=\"dark\"] .highlight")
    dark_auto_css = dark_formatter.get_style_defs(":root:not([data-theme=\"light\"]) .highlight")
    # Keep our own code background in both themes.
    pyg_css += "\n.highlight, .highlight pre { background: transparent !important; }"

    toc_html, chapters_html = [], []
    for no, spec in enumerate(CHAPTERS, start=1):
        chapter_id, toc, chapter_html = build_chapter(no, spec, formatter)
        chapters_html.append(chapter_html)
        items = "".join(f'<li><a href="#{sid}">{html.escape(t)}</a></li>' for sid, t in toc)
        toc_html.append(f'<li><a href="#{chapter_id}">{no}. {html.escape(spec["title"])}</a><ol>{items}</ol></li>')

    has_plotly = any(s["plotly"] for s in STATS.values())
    plotly_tag = f'<script src="{PLOTLY_CDN}"></script>' if has_plotly else ""
    if "language-mermaid" in "".join(chapters_html):
        plotly_tag += f'<script src="{MERMAID_CDN}"></script>'
    chapter_list = "".join(f"<li>{html.escape(c['title'])}</li>" for c in CHAPTERS)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NyayaRAG Research Ebook</title>
<style>{CSS}
{pyg_css}
@media (prefers-color-scheme: dark) {{ {dark_auto_css} }}
{dark_css}
.highlight, .highlight pre {{ background: transparent !important; }}
</style>
{plotly_tag}
</head>
<body>
<div class="layout">
<nav class="toc">
  <div class="brand">NyayaRAG</div>
  <div class="muted">Research ebook</div>
  <ol>{''.join(toc_html)}</ol>
  <div class="theme"><button id="theme-toggle" type="button">Toggle light / dark</button></div>
</nav>
<main>
<header class="cover">
  <div class="kicker">Research ebook</div>
  <h1>NyayaRAG</h1>
  <p class="subtitle">A retrieval-augmented legal research assistant for Indian Supreme Court judgments:
  data preparation, statistical design, benchmark evaluation, retrieval experiments and evidence-grounded generation.</p>
  <ol>{chapter_list}</ol>
  <div class="meta">Compiled from the project scripts in <code>notebook_1/</code> to <code>notebook_5/</code>,
  with outputs and figures from the executed notebooks and saved evaluation artefacts.</div>
</header>
{''.join(chapters_html)}
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(page, encoding="utf-8")

    print(f"{'chapter':<12}{'sections':>9}{'figures':>9}{'plotly':>8}{'tables':>8}"
          f"{'matched':>9}{'followed':>10}{'unmatched':>11}{'errors':>8}{'stderr':>8}")
    for name, s in STATS.items():
        print(f"{name:<12}{s['sections']:>9}{s['figures']:>9}{s['plotly']:>8}{s['tables']:>8}"
              f"{s['cells_matched']:>9}{s['cells_followed']:>10}{s['cells_unmatched']:>11}{s['errors_dropped']:>8}{s['stderr_dropped']:>8}")
    print(f"\nWrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
