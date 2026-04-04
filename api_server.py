"""
DotsOCR HTTP API Server
========================

A FastAPI-based HTTP API that wraps the DotsOCR model for PDF/image OCR.
It manages one or more vLLM backend instances automatically — one per GPU —
and load-balances concurrent requests across them via round-robin.

Set VLLM_GPU="0,1,2,3" to run 4 instances on ports 8000-8003.

Endpoints:
    POST /ocr              - Upload a PDF or image file for OCR processing (sync)
    POST /ocr/submit       - Submit an async OCR job, returns job_id
    GET  /ocr/jobs/{id}    - Poll job progress and retrieve results
    GET  /ocr/jobs         - List all tracked jobs
    GET  /health           - Check API server and per-instance vLLM health
    GET  /prompts          - List available prompt modes and their descriptions
    POST /vllm/start       - Manually start all vLLM instances
    POST /vllm/stop        - Manually stop all vLLM instances

Start via: ./start_server.sh
Stop via:  ./stop_server.sh
"""

from __future__ import annotations

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

import logging

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("dotsocr")

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_BASE_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_GPU = os.getenv("VLLM_GPU", "0")  # comma-separated for multi-GPU, e.g. "0,1,2,3"
MODEL_PATH = os.getenv("MODEL_PATH", "./weights/DotsOCR_1_5")
MODEL_NAME = os.getenv("MODEL_NAME", "model")
API_PORT = int(os.getenv("API_PORT", "8300"))

# Parse GPU list — each GPU gets its own vLLM instance
VLLM_GPUS: list[str] = [g.strip() for g in VLLM_GPU.split(",") if g.strip()]

# Per-instance state: port assignments and processes
VLLM_PORTS: list[int] = [VLLM_BASE_PORT + i for i in range(len(VLLM_GPUS))]
VLLM_PROCESSES: list[subprocess.Popen | None] = [None] * len(VLLM_GPUS)

# Round-robin counter for load balancing
_rr_lock = threading.Lock()
_rr_counter = 0

# ---------------------------------------------------------------------------
# vLLM management helpers
# ---------------------------------------------------------------------------

def _vllm_health_url(port: int) -> str:
    return f"http://{VLLM_HOST}:{port}/health"


def _is_instance_running(port: int) -> bool:
    """Check if a single vLLM instance is responsive."""
    try:
        r = httpx.get(_vllm_health_url(port), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def is_vllm_running() -> bool:
    """Check if at least one vLLM instance is responsive."""
    return any(_is_instance_running(port) for port in VLLM_PORTS)


def get_healthy_ports() -> list[int]:
    """Return list of ports for healthy vLLM instances."""
    return [port for port in VLLM_PORTS if _is_instance_running(port)]


def next_vllm_port() -> int | None:
    """Pick the next healthy vLLM port via round-robin."""
    global _rr_counter
    healthy = get_healthy_ports()
    if not healthy:
        logger.warning("No healthy vLLM instances available")
        return None
    with _rr_lock:
        idx = _rr_counter % len(healthy)
        _rr_counter += 1
    port = healthy[idx]
    gpu = VLLM_GPUS[VLLM_PORTS.index(port)] if port in VLLM_PORTS else "?"
    logger.info("Round-robin selected port %d (GPU %s) — %d/%d healthy",
                port, gpu, len(healthy), len(VLLM_PORTS))
    return port


def _start_single_instance(gpu: str, port: int, idx: int) -> dict:
    """Launch a single vLLM instance on the given GPU and port."""
    if _is_instance_running(port):
        return {"gpu": gpu, "port": port, "status": "already_running"}

    model_abs = str(Path(MODEL_PATH).resolve())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(Path(model_abs).parent) + ":" + env.get("PYTHONPATH", "")

    vllm_bin = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")

    cmd = [
        vllm_bin, "serve", model_abs,
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--chat-template-content-format", "string",
        "--served-model-name", MODEL_NAME,
        "--trust-remote-code",
        "--port", str(port),
    ]

    VLLM_PROCESSES[idx] = subprocess.Popen(cmd, env=env)

    # Wait for healthy (up to 5 min), but bail early if the process dies
    for _ in range(300):
        if _is_instance_running(port):
            return {"gpu": gpu, "port": port, "status": "started", "pid": VLLM_PROCESSES[idx].pid}
        if VLLM_PROCESSES[idx].poll() is not None:
            rc = VLLM_PROCESSES[idx].returncode
            VLLM_PROCESSES[idx] = None
            return {"gpu": gpu, "port": port, "status": "crashed", "returncode": rc}
        time.sleep(1)

    return {"gpu": gpu, "port": port, "status": "timeout"}


VLLM_START_RETRIES = int(os.getenv("VLLM_START_RETRIES", "2"))


def start_vllm() -> dict:
    """Launch vLLM instances sequentially (one per GPU), retrying failures.

    Sequential startup avoids CUDA/NCCL initialization races and system
    resource contention that can crash instances when launched in parallel.
    """
    results = []
    for idx, (gpu, port) in enumerate(zip(VLLM_GPUS, VLLM_PORTS)):
        r = None
        for attempt in range(1, VLLM_START_RETRIES + 1):
            print(f"Starting vLLM instance {idx} on GPU {gpu}, port {port} (attempt {attempt}/{VLLM_START_RETRIES})...")
            r = _start_single_instance(gpu, port, idx)
            print(f"  → {r['status']}")
            if r["status"] in ("started", "already_running"):
                break
            # Brief pause before retry to let resources settle
            time.sleep(5)
        results.append(r)

    started = sum(1 for r in results if r["status"] in ("started", "already_running"))
    return {
        "status": "ok" if started > 0 else "failed",
        "instances": results,
        "num_gpus": len(VLLM_GPUS),
        "num_healthy": started,
    }


def stop_vllm() -> dict:
    """Stop all managed vLLM subprocesses."""
    stopped = []
    for idx, proc in enumerate(VLLM_PROCESSES):
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            stopped.append({"gpu": VLLM_GPUS[idx], "port": VLLM_PORTS[idx], "pid": proc.pid})
            VLLM_PROCESSES[idx] = None
    if stopped:
        return {"status": "stopped", "instances": stopped}
    return {"status": "not_managed", "message": "No vLLM processes managed by this server"}


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

class VllmInstanceHealth(BaseModel):
    gpu: str = Field(description="GPU device ID")
    port: int = Field(description="vLLM instance port")
    status: str = Field(description="'ok' or 'unavailable'")


class HealthResponse(BaseModel):
    api: str = Field(description="API server status", examples=["ok"])
    vllm: str = Field(description="Overall vLLM status: 'ok' if any instance is healthy")
    num_gpus: int = Field(description="Total number of configured GPU instances")
    num_healthy: int = Field(description="Number of healthy vLLM instances")
    instances: list[VllmInstanceHealth] = Field(description="Per-instance health status")


class VllmActionResponse(BaseModel):
    status: str = Field(description="Action result")
    pid: int | None = Field(default=None, description="Process ID if applicable")
    message: str | None = Field(default=None, description="Additional info")


class PageResult(BaseModel):
    page_no: int = Field(description="0-based page number")
    layout: list | str | None = Field(default=None, description="Parsed layout cells (list of dicts with bbox, category, text) or raw text")
    markdown: str | None = Field(default=None, description="Markdown rendering of the page content")
    markdown_nohf: str | None = Field(default=None, description="Markdown rendering without page headers/footers")


class OcrResponse(BaseModel):
    filename: str = Field(description="Original uploaded filename")
    num_pages: int = Field(description="Number of pages processed")
    prompt_mode: str = Field(description="Prompt mode used for OCR")
    pages: list[PageResult] = Field(description="Per-page OCR results")


class JobStatusResponse(BaseModel):
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="'pending', 'running', 'completed', or 'failed'")
    filename: str | None = Field(default=None, description="Original uploaded filename")
    gpu: str | None = Field(default=None, description="GPU handling this job")
    vllm_port: int | None = Field(default=None, description="vLLM port handling this job")
    pages_completed: int = Field(default=0, description="Number of pages processed so far")
    pages_total: int = Field(default=0, description="Total number of pages")
    progress: float = Field(default=0.0, description="Progress 0.0–1.0")
    error: str | None = Field(default=None, description="Error message if failed")
    result: OcrResponse | None = Field(default=None, description="OCR result (only when completed)")


class JobSubmitResponse(BaseModel):
    job_id: str = Field(description="Unique job identifier — poll GET /ocr/jobs/{job_id}")


# ---- Job tracking ------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Auto-cleanup: keep completed/failed jobs for 30 minutes
_JOB_TTL_SECONDS = 1800


def _cleanup_old_jobs() -> None:
    """Remove jobs older than _JOB_TTL_SECONDS."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items()
                   if j["status"] in ("completed", "failed")
                   and now - j.get("finished_at", now) > _JOB_TTL_SECONDS]
        for jid in expired:
            _jobs.pop(jid, None)


def _run_ocr_job(job_id: str, input_path: str, work_dir: str,
                 filename: str, ext: str, prompt_mode: str,
                 dpi: int, num_threads: int, vllm_port: int) -> None:
    """Execute OCR in a background thread, updating job progress."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        from dots_ocr.parser import DotsOCRParser
        from dots_ocr.utils.doc_utils import load_images_from_pdf
        from dots_ocr.utils.image_utils import PILimage_to_base64
        from PIL import Image

        def page_callback(completed: int, total: int) -> None:
            with _jobs_lock:
                _jobs[job_id]["pages_completed"] = completed
                _jobs[job_id]["pages_total"] = total
                _jobs[job_id]["progress"] = completed / total if total else 0.0

        parser = DotsOCRParser(
            ip=VLLM_HOST,
            port=vllm_port,
            model_name=MODEL_NAME,
            dpi=dpi,
            num_thread=num_threads,
            output_dir=work_dir,
        )
        results = parser.parse_file(input_path, output_dir=work_dir,
                                    prompt_mode=prompt_mode,
                                    page_callback=page_callback)

        # For single images, mark progress as complete
        with _jobs_lock:
            _jobs[job_id]["pages_completed"] = len(results)
            _jobs[job_id]["pages_total"] = len(results)
            _jobs[job_id]["progress"] = 1.0

        # Load page images for figure cropping
        if ext == ".pdf":
            page_images = load_images_from_pdf(input_path, dpi=dpi)
        else:
            page_images = [Image.open(input_path).convert("RGB")]

        # Build response (same logic as sync /ocr)
        pages = []
        for r in results:
            layout = None
            markdown = None
            layout_path = r.get("layout_info_path")
            if layout_path and os.path.exists(layout_path):
                with open(layout_path, "r", encoding="utf-8") as lf:
                    layout = json.load(lf)
            page_no = r.get("page_no", 0)
            if isinstance(layout, list) and page_no < len(page_images):
                origin_img = page_images[page_no]
                for cell in layout:
                    if isinstance(cell, dict) and cell.get("category") == "Picture":
                        try:
                            x1, y1, x2, y2 = [int(c) for c in cell["bbox"]]
                            crop = origin_img.crop((x1, y1, x2, y2))
                            cell["image_base64"] = PILimage_to_base64(crop)
                        except Exception:
                            pass
            md_path = r.get("md_content_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as mf:
                    markdown = mf.read()
            markdown_nohf = None
            md_nohf_path = r.get("md_content_nohf_path")
            if md_nohf_path and os.path.exists(md_nohf_path):
                with open(md_nohf_path, "r", encoding="utf-8") as mf:
                    markdown_nohf = mf.read()
            pages.append(PageResult(
                page_no=page_no, layout=layout,
                markdown=markdown, markdown_nohf=markdown_nohf,
            ))

        if prompt_mode == "prompt_layout_all_en":
            pages_dicts = [p.model_dump() for p in pages]
            postprocess_citations(pages_dicts)
            pages = [PageResult(**pd) for pd in pages_dicts]

        ocr_result = OcrResponse(
            filename=filename, num_pages=len(pages),
            prompt_mode=prompt_mode, pages=pages,
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = ocr_result
            _jobs[job_id]["finished_at"] = time.time()
        logger.info("Job %s completed: %d pages", job_id, len(pages))

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = time.time()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Health check",
         description="Returns the health status of both the API server and the vLLM backend.")
async def health():
    instances = []
    for gpu, port in zip(VLLM_GPUS, VLLM_PORTS):
        ok = _is_instance_running(port)
        instances.append(VllmInstanceHealth(gpu=gpu, port=port, status="ok" if ok else "unavailable"))
    num_healthy = sum(1 for i in instances if i.status == "ok")
    return HealthResponse(
        api="ok",
        vllm="ok" if num_healthy > 0 else "unavailable",
        num_gpus=len(VLLM_GPUS),
        num_healthy=num_healthy,
        instances=instances,
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
):
    # Validate vLLM is up — pick a healthy instance via round-robin
    vllm_port = next_vllm_port()
    if vllm_port is None:
        raise HTTPException(status_code=503, detail="No vLLM backend is available. POST /vllm/start to launch them.")

    gpu = VLLM_GPUS[VLLM_PORTS.index(vllm_port)] if vllm_port in VLLM_PORTS else "?"
    logger.info("OCR request: %s → GPU %s (port %d)", file.filename, gpu, vllm_port)

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

        # Run OCR via DotsOCRParser — route to the selected vLLM instance
        from dots_ocr.parser import DotsOCRParser
        from dots_ocr.utils.doc_utils import load_images_from_pdf
        from dots_ocr.utils.image_utils import PILimage_to_base64
        from PIL import Image
        parser = DotsOCRParser(
            ip=VLLM_HOST,
            port=vllm_port,
            model_name=MODEL_NAME,
            dpi=dpi,
            num_thread=num_threads,
            output_dir=work_dir,
        )
        results = parser.parse_file(input_path, output_dir=work_dir, prompt_mode=prompt_mode)

        # Load page images for figure cropping
        if ext == ".pdf":
            page_images = load_images_from_pdf(input_path, dpi=dpi)
        else:
            page_images = [Image.open(input_path).convert("RGB")]

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

            # Embed cropped figures as base64 in Picture cells
            page_no = r.get("page_no", 0)
            if isinstance(layout, list) and page_no < len(page_images):
                origin_img = page_images[page_no]
                for cell in layout:
                    if isinstance(cell, dict) and cell.get("category") == "Picture":
                        try:
                            x1, y1, x2, y2 = [int(c) for c in cell["bbox"]]
                            crop = origin_img.crop((x1, y1, x2, y2))
                            cell["image_base64"] = PILimage_to_base64(crop)
                        except Exception:
                            pass

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

            pages.append(PageResult(
                page_no=page_no,
                layout=layout,
                markdown=markdown,
                markdown_nohf=markdown_nohf,
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


@app.post("/ocr/submit", response_model=JobSubmitResponse, tags=["OCR"],
          summary="Submit an OCR job (async)",
          description=(
              "Upload a PDF or image file for asynchronous OCR processing. "
              "Returns a job_id immediately. Poll GET /ocr/jobs/{job_id} for progress and results."
          ))
async def ocr_submit(
    file: UploadFile = File(
        ..., description="PDF or image file to process (.pdf, .jpg, .jpeg, .png)"
    ),
    prompt_mode: str = Form(
        default="prompt_layout_all_en",
        description="OCR task prompt mode. See GET /prompts for available modes.",
    ),
    dpi: int = Form(default=200, description="DPI for PDF rasterization (default 200)"),
    num_threads: int = Form(default=16, description="Parallel threads for multi-page PDFs"),
):
    # Validate vLLM is up
    vllm_port = next_vllm_port()
    if vllm_port is None:
        raise HTTPException(status_code=503, detail="No vLLM backend is available. POST /vllm/start to launch them.")

    from dots_ocr.utils.prompts import dict_promptmode_to_prompt
    if prompt_mode not in dict_promptmode_to_prompt:
        raise HTTPException(status_code=400, detail=f"Unknown prompt_mode '{prompt_mode}'. GET /prompts for options.")

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".pdf", ".jpg", ".jpeg", ".png"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}")

    # Save upload
    work_dir = tempfile.mkdtemp(prefix="dotsocr_")
    input_path = os.path.join(work_dir, filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create job
    job_id = uuid.uuid4().hex[:12]
    gpu = VLLM_GPUS[VLLM_PORTS.index(vllm_port)] if vllm_port in VLLM_PORTS else None
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "filename": filename,
            "gpu": gpu,
            "vllm_port": vllm_port,
            "pages_completed": 0,
            "pages_total": 0,
            "progress": 0.0,
            "error": None,
            "result": None,
        }

    logger.info("Job %s submitted: %s → GPU %s (port %d)", job_id, filename, gpu, vllm_port)

    # Launch background thread
    threading.Thread(
        target=_run_ocr_job, daemon=True,
        args=(job_id, input_path, work_dir, filename, ext,
              prompt_mode, dpi, num_threads, vllm_port),
    ).start()

    _cleanup_old_jobs()
    return JobSubmitResponse(job_id=job_id)


@app.get("/ocr/jobs/{job_id}", response_model=JobStatusResponse, tags=["OCR"],
         summary="Get OCR job status and progress",
         description="Poll this endpoint to track progress of an async OCR job.")
async def ocr_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(job_id=job_id, **job)


@app.get("/ocr/jobs", response_model=list[JobStatusResponse], tags=["OCR"],
         summary="List all OCR jobs",
         description="Returns status of all tracked jobs (completed jobs expire after 30 minutes).")
async def ocr_jobs_list():
    _cleanup_old_jobs()
    with _jobs_lock:
        return [
            JobStatusResponse(job_id=jid, **{k: v for k, v in j.items() if k != "result"})
            for jid, j in _jobs.items()
        ]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run("api_server:app", host="0.0.0.0", port=API_PORT, workers=1)
