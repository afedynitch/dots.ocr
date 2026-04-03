"""
DotsOCR HTTP API Server
========================

A FastAPI-based HTTP API that wraps the DotsOCR model for PDF/image OCR.
It manages a vLLM backend server automatically (launches if not running).

Endpoints:
    POST /ocr          - Upload a PDF or image file for OCR processing
    GET  /health       - Check API server and vLLM backend health
    GET  /prompts      - List available prompt modes and their descriptions
    POST /vllm/start   - Manually start the vLLM backend
    POST /vllm/stop    - Manually stop the vLLM backend

Start via: ./start_server.sh
Stop via:  ./stop_server.sh
"""

from __future__ import annotations

import base64
import os
import json
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_GPU = os.getenv("VLLM_GPU", "0")
MODEL_PATH = os.getenv("MODEL_PATH", "./weights/DotsOCR_1_5")
MODEL_NAME = os.getenv("MODEL_NAME", "model")
API_PORT = int(os.getenv("API_PORT", "8300"))

VLLM_PROCESS: subprocess.Popen | None = None

# ---------------------------------------------------------------------------
# vLLM management helpers
# ---------------------------------------------------------------------------

def _vllm_health_url() -> str:
    return f"http://{VLLM_HOST}:{VLLM_PORT}/health"


def is_vllm_running() -> bool:
    """Check if the vLLM server is responsive."""
    try:
        r = httpx.get(_vllm_health_url(), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def start_vllm() -> dict:
    """Launch the vLLM server as a subprocess if not already running."""
    global VLLM_PROCESS

    if is_vllm_running():
        return {"status": "already_running"}

    model_abs = str(Path(MODEL_PATH).resolve())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VLLM_GPU
    # Ensure the model directory is on PYTHONPATH for custom modelling code
    env["PYTHONPATH"] = str(Path(model_abs).parent) + ":" + env.get("PYTHONPATH", "")

    # Resolve vllm binary — use the one from the same venv as the current Python
    vllm_bin = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")

    cmd = [
        vllm_bin, "serve", model_abs,
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--chat-template-content-format", "string",
        "--served-model-name", MODEL_NAME,
        "--trust-remote-code",
        "--port", str(VLLM_PORT),
    ]

    VLLM_PROCESS = subprocess.Popen(cmd, env=env)

    # Wait for the server to become healthy (up to 5 min)
    for _ in range(300):
        if is_vllm_running():
            return {"status": "started", "pid": VLLM_PROCESS.pid}
        time.sleep(1)

    return {"status": "timeout", "message": "vLLM did not become healthy within 5 minutes"}


def stop_vllm() -> dict:
    """Stop the managed vLLM subprocess."""
    global VLLM_PROCESS
    if VLLM_PROCESS is not None:
        VLLM_PROCESS.terminate()
        try:
            VLLM_PROCESS.wait(timeout=30)
        except subprocess.TimeoutExpired:
            VLLM_PROCESS.kill()
        pid = VLLM_PROCESS.pid
        VLLM_PROCESS = None
        return {"status": "stopped", "pid": pid}
    return {"status": "not_managed", "message": "No vLLM process managed by this server"}


# ---------------------------------------------------------------------------
# Citation postprocessing
# ---------------------------------------------------------------------------

def is_reference_block(text: str) -> bool:
    """Detect if a Text cell contains a reference list (not inline citations)."""
    if not text:
        return False
    line_refs = list(re.finditer(r'^\s*\[(\d+)\]', text, re.MULTILINE))
    if len(line_refs) < 2:
        return False
    if line_refs[0].start() > 20:
        return False
    nums = [int(m.group(1)) for m in line_refs]
    return nums == sorted(nums) and nums[-1] - nums[0] >= 1


def parse_reference_block(text: str, parent_bbox: list) -> list[dict]:
    """Split a reference block into individual Citation cells."""
    parts = re.split(r'(?m)(?=^\s*\[\d+\]\s)', text)
    citations = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'\[(\d+)\]\s*(.*)', part, re.DOTALL)
        if m:
            number = int(m.group(1))
            ref_text = m.group(2).strip()
            ref_text = re.sub(r'\n(?!\n)', ' ', ref_text)
            ref_text = re.sub(r'  +', ' ', ref_text)
            citations.append({
                "bbox": parent_bbox,
                "category": "Citation",
                "number": number,
                "text": ref_text,
            })
    return citations


def split_reference_cells(pages_data: list[dict]) -> None:
    """Replace Text cells containing reference lists with individual Citation cells."""
    for page in pages_data:
        layout = page.get("layout")
        if not isinstance(layout, list):
            continue
        new_cells = []
        for cell in layout:
            if (cell.get("category") == "Text"
                    and is_reference_block(cell.get("text", ""))):
                new_cells.extend(
                    parse_reference_block(cell["text"], cell.get("bbox", [0, 0, 0, 0]))
                )
            else:
                new_cells.append(cell)
        page["layout"] = new_cells


def _mask_math_regions(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace $...$ and $$...$$ regions with placeholders to protect them."""
    placeholders = []
    counter = [0]

    def _replace(m):
        tag = f"\x00MATH{counter[0]}\x00"
        counter[0] += 1
        placeholders.append((tag, m.group(0)))
        return tag

    # $$...$$ first (greedy over $...$), then $...$
    text = re.sub(r'\$\$.*?\$\$', _replace, text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+?\$', _replace, text)
    return text, placeholders


def _unmask_math_regions(text: str, placeholders: list[tuple[str, str]]) -> str:
    """Restore math regions from placeholders."""
    for tag, original in placeholders:
        text = text.replace(tag, original)
    return text


def _parse_citation_inner(inner: str) -> list[int] | None:
    """Parse the inside of a citation bracket into a list of reference numbers."""
    inner = inner.strip()
    # Single number
    if inner.isdigit():
        return [int(inner)]
    # Range: "4-12" or "4–12"
    range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', inner)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start < end and (end - start) <= 50:
            return list(range(start, end + 1))
    # Comma/semicolon separated: "2, 3" or "14, 15"
    if re.match(r'^[\d\s,;]+$', inner):
        nums = [int(x) for x in re.split(r'[,;]\s*', inner) if x.strip().isdigit()]
        if nums:
            return nums
    # Mixed: "2, 4-6" — split on comma first, then handle ranges
    parts = re.split(r'[,;]\s*', inner)
    nums = []
    for part in parts:
        part = part.strip()
        if part.isdigit():
            nums.append(int(part))
        else:
            rm = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', part)
            if rm:
                s, e = int(rm.group(1)), int(rm.group(2))
                if s < e and (e - s) <= 50:
                    nums.extend(range(s, e + 1))
    return nums if nums else None


# Pattern for inline citations: [N], [N, M], [N-M], [N–M], and combinations
_INLINE_CITE_RE = re.compile(
    r'\[(\d+(?:\s*[,;]\s*\d+)*(?:\s*[-–]\s*\d+)?)\]'
)


def find_reference_list_start(md: str) -> int:
    """Find the character position where the reference list begins."""
    line_start_refs = list(re.finditer(r'^(\[\d+\])\s', md, re.MULTILINE))
    if len(line_start_refs) < 3:
        return len(md)
    for m in line_start_refs:
        if m.group(1) == '[1]':
            ref1_pos = m.start()
            has_ref2 = any(
                m2.group(1) == '[2]' and m2.start() - ref1_pos < 2000
                for m2 in line_start_refs
            )
            if has_ref2:
                return ref1_pos
    return len(md)


def fix_references_block(md: str) -> str:
    """Reformat the reference list so each [N] entry is its own paragraph."""
    ref_start_idx = find_reference_list_start(md)
    if ref_start_idx >= len(md):
        return md

    pre_references = md[:ref_start_idx]
    references_block = md[ref_start_idx:]

    parts = re.split(r'(?=\[\d+\]\s)', references_block)
    formatted_refs = []
    for part in parts:
        part = part.strip()
        if part:
            part = re.sub(r'\n(?!\n)', ' ', part)
            part = re.sub(r'  +', ' ', part)
            formatted_refs.append(part)

    if not formatted_refs:
        return md

    if not pre_references.endswith('\n\n'):
        pre_references = pre_references.rstrip('\n') + '\n\n'

    return pre_references + '\n\n'.join(formatted_refs)


def convert_inline_citations(md: str, ref_start_pos: int) -> str:
    """Convert inline citations in body text to pandoc [@N] format."""
    body = md[:ref_start_pos]
    refs = md[ref_start_pos:]

    body, placeholders = _mask_math_regions(body)

    def _expand(match):
        nums = _parse_citation_inner(match.group(1))
        if not nums:
            return match.group(0)
        return '[' + '; '.join(f'@{n}' for n in nums) + ']'

    body = _INLINE_CITE_RE.sub(_expand, body)
    body = _unmask_math_regions(body, placeholders)
    return body + refs


# ---------------------------------------------------------------------------
# Author-year citation style (ApJ, MNRAS, etc.)
# ---------------------------------------------------------------------------

# Lowercase surname prefixes that should be joined to the next word
_SURNAME_PREFIXES = {'van', 'von', 'de', 'del', 'della', 'di', 'le', 'la', 'el',
                     'al', 'bin', 'ibn', 'mac', 'mc', 'den', 'der', 'ter', 'ten'}

# Pattern matching an ApJ-style reference line
_AUTHORYEAR_REF_RE = re.compile(
    r'^[A-ZÀ-ÖØ-Þa-zß-öø-ÿ\'][A-Za-zÀ-ÖØ-Þß-öø-ÿ\' -]+,\s+.*\b(1[89]\d{2}|20[0-3]\d)\b',
    re.MULTILINE,
)


def is_authoryear_reference_block(text: str) -> bool:
    """Detect if a Text cell contains an author-year reference list."""
    if not text:
        return False
    matches = _AUTHORYEAR_REF_RE.findall(text)
    return len(matches) >= 5


def _extract_first_author(line: str) -> str | None:
    """Extract the first author surname from an ApJ reference line.

    Handles:
      - Simple: ``Aab, A.,`` → ``Aab``
      - Multi-word: ``van Velzen, S.,`` → ``vanVelzen``
      - Apostrophe: ``Rouillé d'Orfeuil, B.,`` → ``Rouilledorfeuil`` (simplified)
      - Collaboration: ``Pierre Auger Collaboration, Aab, ...`` → ``PierreAugerCollaboration``
    """
    # Collaboration names: ends with "Collaboration" before the first real author
    collab_m = re.match(r'^(.+?Collaboration)\s*,', line)
    if collab_m:
        name = collab_m.group(1)
        # CamelCase the words, strip spaces
        return re.sub(r'[^A-Za-z0-9]', '', name)

    # Normal author: everything before the first ", Initial"
    m = re.match(r'^([^,]+),\s+[A-Z]', line)
    if not m:
        return None
    surname = m.group(1).strip()
    # CamelCase multi-word surnames: "van Velzen" → "vanVelzen"
    parts = surname.split()
    if len(parts) > 1:
        result = []
        for i, p in enumerate(parts):
            clean = re.sub(r"[^A-Za-zÀ-ÖØ-Þß-öø-ÿ]", '', p)
            if i == 0 and p.lower() in _SURNAME_PREFIXES:
                result.append(clean.lower())
            elif i > 0 and p.lower() in _SURNAME_PREFIXES:
                result.append(clean.lower())
            else:
                result.append(clean)
        return ''.join(result)
    # Strip non-alpha (apostrophes, hyphens kept as-is in key)
    return re.sub(r"[^A-Za-zÀ-ÖØ-Þß-öø-ÿ]", '', surname)


def _extract_year(line: str) -> str | None:
    """Extract the publication year from an ApJ reference line."""
    # Year appears after the author list, typically preceded by a space
    m = re.search(r'\b(1[89]\d{2}|20[0-3]\d)([a-z])?\b', line)
    if m:
        return m.group(1) + (m.group(2) or '')
    return None


def parse_authoryear_reference_block(text: str, parent_bbox: list,
                                     start_number: int = 1) -> list[dict]:
    """Split an author-year reference block into individual Citation cells."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    citations = []
    raw_refs = []  # (first_author, year, full_text)

    for line in lines:
        author = _extract_first_author(line)
        year = _extract_year(line)
        if author and year:
            raw_refs.append((author, year, line))

    # Generate citation keys with disambiguation
    key_counts: dict[str, int] = {}
    for author, year, _ in raw_refs:
        base = f"{author}{year}"
        key_counts[base] = key_counts.get(base, 0) + 1

    # Assign keys — add a/b/c suffix only when there are duplicates
    key_seen: dict[str, int] = {}
    for i, (author, year, full_text) in enumerate(raw_refs):
        base = f"{author}{year}"
        if key_counts[base] > 1:
            idx = key_seen.get(base, 0)
            key_seen[base] = idx + 1
            key = f"{base}{chr(ord('a') + idx)}"
        else:
            key = base
        citations.append({
            "bbox": parent_bbox,
            "category": "Citation",
            "number": start_number + i,
            "key": key,
            "text": full_text,
        })

    return citations


def split_authoryear_reference_cells(pages_data: list[dict]) -> bool:
    """Replace Text cells containing author-year reference lists with Citation cells.

    Returns True if any author-year references were found.
    """
    found = False
    number = 1
    for page in pages_data:
        layout = page.get("layout")
        if not isinstance(layout, list):
            continue
        new_cells = []
        for cell in layout:
            if (cell.get("category") == "Text"
                    and is_authoryear_reference_block(cell.get("text", ""))):
                cites = parse_authoryear_reference_block(
                    cell["text"], cell.get("bbox", [0, 0, 0, 0]),
                    start_number=number,
                )
                new_cells.extend(cites)
                number += len(cites)
                found = True
            else:
                new_cells.append(cell)
        page["layout"] = new_cells
    return found


def build_citation_key_map(pages_data: list[dict]) -> dict[tuple[str, str], str]:
    """Build a lookup from (author_surname, year) → citation key.

    The author_surname is stored in *lowercase* for case-insensitive matching.
    """
    key_map: dict[tuple[str, str], str] = {}
    for page in pages_data:
        layout = page.get("layout")
        if not isinstance(layout, list):
            continue
        for cell in layout:
            if not isinstance(cell, dict):
                continue
            if cell.get("category") != "Citation" or "key" not in cell:
                continue
            key = cell["key"]
            text = cell["text"]
            author = _extract_first_author(text)
            year = _extract_year(text)
            if author and year:
                # Store with lowercase author for case-insensitive lookup
                key_map[(author.lower(), year)] = key
    return key_map


def prefix_authoryear_references(md: str, key_map: dict) -> str:
    """Prefix each author-year reference line with [Key] for pandoc linking."""
    ref_start = find_authoryear_reflist_start(md)
    if ref_start >= len(md):
        return md

    pre = md[:ref_start]
    refs = md[ref_start:]

    lines = refs.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped == 'References':
            new_lines.append(stripped)
            continue
        # Try to extract author+year and look up the key
        author = _extract_first_author(stripped)
        year = _extract_year(stripped)
        if author and year:
            key = key_map.get((author.lower(), year))
            if key:
                new_lines.append(f'[{key}] {stripped}')
                continue
        new_lines.append(stripped)

    return pre + '\n\n'.join(new_lines)


def find_authoryear_reflist_start(md: str) -> int:
    """Find where the reference list begins by looking for a 'References' header."""
    # Match "References" as a section header (## References) or standalone line
    m = re.search(r'^(?:#{1,3}\s+)?References\s*$', md, re.MULTILINE)
    if m:
        return m.start()
    return len(md)


def _lookup_cite_key(author: str, year: str, key_map: dict) -> str | None:
    """Look up citation key for an author+year, case-insensitive on author."""
    # Normalize author: strip spaces, collapse to CamelCase-ish
    normalized = re.sub(r"[^A-Za-zÀ-ÖØ-Þß-öø-ÿ]", '', author)
    return key_map.get((normalized.lower(), year))


# Author name pattern for inline citations: allows lowercase prefixes (van, de, d')
# and multi-part surnames like "Rouillé d'Orfeuil", "Alves Batista"
_AUTHOR_PAT = (
    r"(?:(?:van|von|de|d'|del|della)\s+)?"
    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-Þß-öø-ÿé'-]+"
    r"(?:\s+(?:d'|de\s+)?[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-Þß-öø-ÿé'-]+)*"
)


def _convert_single_authoryear_cite(text: str, key_map: dict) -> str | None:
    """Convert a single author-year reference like 'Name et al. year' to a key.

    Returns the citation key (e.g., 'Abbasi2014') or None if not found.
    """
    text = text.strip()
    # Pattern: Author et al. year
    m = re.match(rf'({_AUTHOR_PAT})\s+et\s+al\.\s+(\d{{4}}[a-z]?)', text)
    if m:
        return _lookup_cite_key(m.group(1), m.group(2), key_map)

    # Pattern: Author1 & Author2 year
    m = re.match(rf'({_AUTHOR_PAT})\s+&\s+{_AUTHOR_PAT}\s+(\d{{4}}[a-z]?)', text)
    if m:
        return _lookup_cite_key(m.group(1), m.group(2), key_map)

    # Pattern: Author year (single author)
    m = re.match(rf'({_AUTHOR_PAT})\s+(\d{{4}}[a-z]?)\s*$', text)
    if m:
        return _lookup_cite_key(m.group(1), m.group(2), key_map)

    return None


def convert_authoryear_inline_citations(md: str, ref_start_pos: int,
                                        key_map: dict) -> str:
    """Convert author-year inline citations to pandoc format."""
    body = md[:ref_start_pos]
    refs = md[ref_start_pos:]

    body, placeholders = _mask_math_regions(body)

    # 1. Parenthetical citations: (... year ...) possibly with multiple semicolons
    #    Matches: (Aab et al. 2017; Hanlon 2019), (e.g., Farrar et al. 2015),
    #    (see Allard 2012, for a review)
    def _convert_parenthetical(m):
        inner = m.group(1)
        # Split by semicolons to handle multi-cite
        parts = re.split(r'\s*;\s*', inner)
        prefix = ''
        suffix = ''

        # Extract prefix like "e.g.," or "see"
        first = parts[0]
        prefix_m = re.match(r'^((?:e\.g\.,?|see|cf\.)\s+)', first)
        if prefix_m:
            prefix = prefix_m.group(1)
            parts[0] = first[prefix_m.end():]

        # Extract suffix from last part like ", for a review"
        last = parts[-1]
        suffix_m = re.search(r',\s+(for\s+.+)$', last)
        if suffix_m:
            suffix = ', ' + suffix_m.group(1)
            parts[-1] = last[:suffix_m.start()]

        # Convert each citation part
        keys = []
        unconverted = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            key = _convert_single_authoryear_cite(part, key_map)
            if key:
                keys.append(f'@{key}')
            else:
                unconverted.append(part)

        if not keys:
            return m.group(0)  # nothing converted, leave as-is

        if unconverted:
            # Mix of converted and unconverted — leave as-is to be safe
            return m.group(0)

        cite_str = '; '.join(keys)
        if prefix or suffix:
            return f'[{prefix}{cite_str}{suffix}]'
        return f'[{cite_str}]'

    # Match parenthetical citations containing author names and years
    body = re.sub(
        r'\(([^)]*?\b(?:1[89]\d{2}|20[0-3]\d)[a-z]?[^)]*)\)',
        _convert_parenthetical,
        body,
    )

    # 2. Narrative citations: Name et al. (year), Name & Name2 (year), Name (year)
    def _convert_narrative(m):
        author_part = m.group(1).strip()
        year = m.group(2)
        # Strip "et al." or "& SecondAuthor" to get first author only
        first_author = re.sub(r'\s+(et\s+al\.|&\s+.+)', '', author_part).strip()
        key = _lookup_cite_key(first_author, year, key_map)
        if key:
            return f'@{key}'
        return m.group(0)

    # Name et al. (year) — e.g., "Karachentsev et al. (2013)", "van Velzen et al. (2012)"
    body = re.sub(
        rf'({_AUTHOR_PAT}\s+et\s+al\.)\s+\((\d{{4}}[a-z]?)\)',
        _convert_narrative,
        body,
    )

    # Name1 & Name2 (year) — e.g., "Jansson & Farrar (2012)"
    body = re.sub(
        rf'({_AUTHOR_PAT}\s+&\s+{_AUTHOR_PAT})\s+\((\d{{4}}[a-z]?)\)',
        _convert_narrative,
        body,
    )

    # Name (year) — single author, e.g., "Sommers (2001)"
    body = re.sub(
        rf'({_AUTHOR_PAT})\s+\((\d{{4}}[a-z]?)\)',
        _convert_narrative,
        body,
    )

    body = _unmask_math_regions(body, placeholders)
    return body + refs


_PAGE_SENTINEL = '\n\n<!-- PAGE_BREAK -->\n\n'


def _detect_citation_style(pages_data: list[dict]) -> str:
    """Detect whether the document uses numeric [N] or author-year citations."""
    for page in pages_data:
        layout = page.get("layout")
        if not isinstance(layout, list):
            continue
        for cell in layout:
            if not isinstance(cell, dict):
                continue
            if cell.get("category") == "Text":
                text = cell.get("text", "")
                if is_reference_block(text):
                    return "numeric"
                if is_authoryear_reference_block(text):
                    return "authoryear"
    return "unknown"


def postprocess_citations(pages_data: list[dict]) -> None:
    """Top-level citation postprocessor for both JSON layout and markdown."""
    style = _detect_citation_style(pages_data)

    if style == "numeric":
        # JSON: split reference blocks into Citation cells
        split_reference_cells(pages_data)

        # Markdown: fix references block per-page first, then convert inline citations
        for md_key in ('markdown', 'markdown_nohf'):
            for p in pages_data:
                md = p.get(md_key) or ''
                p[md_key] = fix_references_block(md)

            parts = [p.get(md_key) or '' for p in pages_data]
            merged = _PAGE_SENTINEL.join(parts)
            ref_start = find_reference_list_start(merged)
            merged = convert_inline_citations(merged, ref_start)

            result_parts = merged.split(_PAGE_SENTINEL)
            if len(result_parts) == len(pages_data):
                for p, new_md in zip(pages_data, result_parts):
                    p[md_key] = new_md

    elif style == "authoryear":
        # JSON: split reference blocks into Citation cells
        split_authoryear_reference_cells(pages_data)

        # Build lookup map from (author, year) → citation key
        key_map = build_citation_key_map(pages_data)

        # Markdown: convert inline citations
        for md_key in ('markdown', 'markdown_nohf'):
            parts = [p.get(md_key) or '' for p in pages_data]
            merged = _PAGE_SENTINEL.join(parts)

            ref_start = find_authoryear_reflist_start(merged)
            merged = convert_authoryear_inline_citations(merged, ref_start, key_map)
            merged = prefix_authoryear_references(merged, key_map)

            result_parts = merged.split(_PAGE_SENTINEL)
            if len(result_parts) == len(pages_data):
                for p, new_md in zip(pages_data, result_parts):
                    p[md_key] = new_md


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

PROMPT_DESCRIPTIONS = {
    "prompt_layout_all_en": "Full document layout + OCR → JSON with bboxes, categories, and text (default)",
    "prompt_layout_only_en": "Layout detection only (no text extraction)",
    "prompt_ocr": "Plain text extraction (no layout info)",
    "prompt_grounding_ocr": "OCR within a given bounding box",
    "prompt_web_parsing": "Parse webpage screenshot layout into JSON",
    "prompt_scene_spotting": "Detect and recognize scene text",
    "prompt_image_to_svg": "Generate SVG code to reconstruct the image",
    "prompt_general": "General free-form prompt",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, launch vLLM in a background thread (non-blocking)."""
    def _start():
        result = start_vllm()
        print(f"vLLM startup: {result}")
    threading.Thread(target=_start, daemon=True).start()
    yield
    stop_vllm()


app = FastAPI(
    title="DotsOCR API",
    version="1.0.0",
    description=(
        "HTTP API for document OCR powered by DotsOCR. "
        "Upload PDF or image files and receive structured layout + text results as JSON. "
        "The server automatically manages a vLLM backend for model inference."
    ),
    lifespan=lifespan,
)


# ---- Response models --------------------------------------------------------

class HealthResponse(BaseModel):
    api: str = Field(description="API server status", examples=["ok"])
    vllm: str = Field(description="vLLM backend status: 'ok' or 'unavailable'")
    vllm_url: str = Field(description="vLLM health endpoint URL")


class VllmActionResponse(BaseModel):
    status: str = Field(description="Action result")
    pid: int | None = Field(default=None, description="Process ID if applicable")
    message: str | None = Field(default=None, description="Additional info")


class PageResult(BaseModel):
    page_no: int = Field(description="0-based page number")
    layout: list | str | None = Field(default=None, description="Parsed layout cells (list of dicts with bbox, category, text) or raw text")
    markdown: str | None = Field(default=None, description="Markdown rendering of the page content")
    markdown_nohf: str | None = Field(default=None, description="Markdown rendering without page headers/footers")
    image_base64: str | None = Field(default=None, description="Base64-encoded layout image (JPEG), included when include_images=true")


class OcrResponse(BaseModel):
    filename: str = Field(description="Original uploaded filename")
    num_pages: int = Field(description="Number of pages processed")
    prompt_mode: str = Field(description="Prompt mode used for OCR")
    pages: list[PageResult] = Field(description="Per-page OCR results")


# ---- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Health check",
         description="Returns the health status of both the API server and the vLLM backend.")
async def health():
    vllm_ok = is_vllm_running()
    return HealthResponse(
        api="ok",
        vllm="ok" if vllm_ok else "unavailable",
        vllm_url=_vllm_health_url(),
    )


@app.get("/prompts", tags=["System"],
         summary="List available prompt modes",
         description="Returns all supported prompt modes and a human-readable description of each.")
async def list_prompts():
    return PROMPT_DESCRIPTIONS


@app.post("/vllm/start", response_model=VllmActionResponse, tags=["System"],
          summary="Start vLLM backend",
          description="Manually launch the vLLM model server. No-op if already running.")
async def vllm_start():
    result = start_vllm()
    return VllmActionResponse(**result)


@app.post("/vllm/stop", response_model=VllmActionResponse, tags=["System"],
          summary="Stop vLLM backend",
          description="Stop the vLLM model server managed by this API.")
async def vllm_stop():
    result = stop_vllm()
    return VllmActionResponse(**result)


@app.post("/ocr", response_model=OcrResponse, tags=["OCR"],
          summary="OCR a PDF or image file",
          description=(
              "Upload a PDF or image file (jpg, jpeg, png) for OCR processing. "
              "Returns structured per-page results including layout bounding boxes, "
              "categories, extracted text, and a Markdown rendering. "
              "Use the `prompt_mode` parameter to choose the OCR task type "
              "(see GET /prompts for options)."
          ))
async def ocr(
    file: UploadFile = File(
        ..., description="PDF or image file to process (.pdf, .jpg, .jpeg, .png)"
    ),
    prompt_mode: str = Form(
        default="prompt_layout_all_en",
        description="OCR task prompt mode. See GET /prompts for available modes.",
    ),
    dpi: int = Form(default=200, description="DPI for PDF rasterization (default 200)"),
    num_threads: int = Form(default=16, description="Parallel threads for multi-page PDFs"),
    include_images: bool = Form(default=False, description="Include base64-encoded layout images in the response"),
):
    # Validate vLLM is up
    if not is_vllm_running():
        raise HTTPException(status_code=503, detail="vLLM backend is not available. POST /vllm/start to launch it.")

    # Validate prompt mode
    from dots_ocr.utils.prompts import dict_promptmode_to_prompt
    if prompt_mode not in dict_promptmode_to_prompt:
        raise HTTPException(status_code=400, detail=f"Unknown prompt_mode '{prompt_mode}'. GET /prompts for options.")

    # Validate file extension
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}")

    # Save upload to a temp directory
    work_dir = tempfile.mkdtemp(prefix="dotsocr_")
    input_path = os.path.join(work_dir, filename)
    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run OCR via DotsOCRParser
        from dots_ocr.parser import DotsOCRParser
        parser = DotsOCRParser(
            ip=VLLM_HOST,
            port=VLLM_PORT,
            model_name=MODEL_NAME,
            dpi=dpi,
            num_thread=num_threads,
            output_dir=work_dir,
        )
        results = parser.parse_file(input_path, output_dir=work_dir, prompt_mode=prompt_mode)

        # Build response
        pages = []
        for r in results:
            layout = None
            markdown = None

            # Read layout JSON if present
            layout_path = r.get("layout_info_path")
            if layout_path and os.path.exists(layout_path):
                with open(layout_path, "r", encoding="utf-8") as lf:
                    layout = json.load(lf)

            # Read markdown if present
            md_path = r.get("md_content_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as mf:
                    markdown = mf.read()

            # Read no-header/footer markdown if present
            markdown_nohf = None
            md_nohf_path = r.get("md_content_nohf_path")
            if md_nohf_path and os.path.exists(md_nohf_path):
                with open(md_nohf_path, "r", encoding="utf-8") as mf:
                    markdown_nohf = mf.read()

            # Optionally include base64-encoded layout image
            image_b64 = None
            if include_images:
                img_path = r.get("layout_image_path")
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as img_f:
                        image_b64 = base64.b64encode(img_f.read()).decode("ascii")

            pages.append(PageResult(
                page_no=r.get("page_no", 0),
                layout=layout,
                markdown=markdown,
                markdown_nohf=markdown_nohf,
                image_base64=image_b64,
            ))

        # Citation postprocessing (only for layout+OCR mode)
        if prompt_mode == "prompt_layout_all_en":
            pages_dicts = [p.model_dump() for p in pages]
            postprocess_citations(pages_dicts)
            pages = [PageResult(**pd) for pd in pages_dicts]

        return OcrResponse(
            filename=filename,
            num_pages=len(pages),
            prompt_mode=prompt_mode,
            pages=pages,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=API_PORT, workers=1)
