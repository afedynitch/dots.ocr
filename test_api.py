#!/usr/bin/env python3
"""
dots_ocr_client — Submit PDF files to a running DotsOCR API server.

Usage:
    python test_api.py [OPTIONS] FILE [FILE ...]

Examples:
    python test_api.py paper.pdf
    python test_api.py --n_threads 8 my_folder/*.pdf
    python test_api.py --api-url http://gpu01:8300 --output-dir out/ *.pdf
    python test_api.py --list-prompts          # just list available modes and exit
"""

import argparse
import base64
import glob
import json
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_health(base: str) -> bool:
    try:
        r = httpx.get(f"{base}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        print(f"API: {health['api']}, vLLM: {health['vllm']}")
        if health["vllm"] != "ok":
            print("WARNING: vLLM is not available — OCR calls will likely fail.")
            return False
        return True
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return False


def list_prompts(base: str) -> dict:
    r = httpx.get(f"{base}/prompts", timeout=10)
    r.raise_for_status()
    return r.json()


def ocr_pdf(base: str, pdf_path: str, prompt_mode: str, output_dir: str | None, no_json: bool = False) -> str:
    """Send one PDF to the OCR endpoint and save outputs. Returns a status string."""
    name = os.path.basename(pdf_path)
    stem = os.path.splitext(name)[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(pdf_path))
    out_prefix = os.path.join(out_dir, stem)

    with open(pdf_path, "rb") as f:
        r = httpx.post(
            f"{base}/ocr",
            files={"file": (name, f, "application/pdf")},
            data={"prompt_mode": prompt_mode},
            timeout=3600,
        )
    r.raise_for_status()
    result = r.json()

    pages = result.get("pages", [])
    num_pages = result.get("num_pages", len(pages))
    saved = []

    # Save JSON (unless suppressed)
    if not no_json:
        json_out = f"{out_prefix}.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        saved.append(json_out)

    # Save merged markdown (no headers/footers preferred)
    # Extract base64-embedded images into img/<stem>/ and replace with
    # relative paths like img/<stem>/page_<P>-fig_<F>.png
    md_chunks = []
    for page_idx, page in enumerate(pages, 1):
        md = page.get("markdown_nohf") or page.get("markdown") or ""
        if md.strip():
            md_chunks.append(md.strip())
    if md_chunks:
        merged_md = "\n\n".join(md_chunks) + "\n"
        img_dir = os.path.join(out_dir, "img", stem)
        fig_counter = {}  # page_idx -> count

        def _save_b64_match(m):
            b64_data = m.group("b64")
            pg = int(m.group("page"))
            fig_counter[pg] = fig_counter.get(pg, 0) + 1
            fig_name = f"page_{pg}-fig_{fig_counter[pg]}.png"
            os.makedirs(img_dir, exist_ok=True)
            with open(os.path.join(img_dir, fig_name), "wb") as img_f:
                img_f.write(base64.b64decode(b64_data))
            return f"![](img/{stem}/{fig_name})"

        # Tag each base64 image with its page number so the callback knows
        tagged_chunks = []
        for page_idx, chunk in enumerate(md_chunks, 1):
            tagged_chunks.append(
                re.sub(
                    r"!\[\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)\)",
                    lambda m, p=page_idx: f"![](data:__page_{p};base64,{m.group(1)})",
                    chunk,
                )
            )
        merged_md = "\n\n".join(tagged_chunks) + "\n"
        merged_md = re.sub(
            r"!\[\]\(data:__page_(?P<page>\d+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)\)",
            _save_b64_match,
            merged_md,
        )

        md_out = f"{out_prefix}.md"
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(merged_md)
        saved.append(md_out)

    return f"{name}: {num_pages} pages → {', '.join(saved)}"


def process_one(base: str, pdf_path: str, prompt_mode: str, output_dir: str | None, no_json: bool = False) -> tuple[str, str | None]:
    """Return (pdf_path, error_or_None)."""
    try:
        status = ocr_pdf(base, pdf_path, prompt_mode, output_dir, no_json)
        return pdf_path, None, status
    except httpx.HTTPStatusError as e:
        return pdf_path, f"HTTP {e.response.status_code}: {e.response.text}", None
    except Exception as e:
        return pdf_path, str(e), None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit PDF files to a DotsOCR API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        metavar="FILE",
        help="PDF file(s) to process. Glob patterns are expanded by the shell.",
    )
    parser.add_argument("--api-url", default="http://localhost:8300", help="Base URL of the API (default: %(default)s)")
    parser.add_argument("--prompt-mode", default="prompt_layout_all_en", help="OCR prompt mode (default: %(default)s)")
    parser.add_argument("--n_threads", type=int, default=1, metavar="N", help="Number of parallel upload threads (default: %(default)s)")
    parser.add_argument("--output-dir", metavar="DIR", help="Directory for output files (default: same dir as each PDF)")
    parser.add_argument("--no-json", action="store_true", help="Skip saving JSON output; produce only .md files")
    parser.add_argument("--list-prompts", action="store_true", help="List available prompt modes and exit")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip the health check at startup")

    args = parser.parse_args()
    base = args.api_url.rstrip("/")

    # Create output dir if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Health check
    if not args.skip_health_check:
        if not check_health(base):
            sys.exit(1)

    # List prompts
    if args.list_prompts:
        prompts = list_prompts(base)
        print("Available prompt modes:")
        for k, v in prompts.items():
            print(f"  {k}: {v}")
        return

    # Expand any glob patterns that the shell didn't expand (e.g. on Windows)
    pdf_paths = []
    for pattern in args.pdfs:
        expanded = glob.glob(pattern)
        if expanded:
            pdf_paths.extend(expanded)
        else:
            pdf_paths.append(pattern)  # let the error surface naturally

    if not pdf_paths:
        parser.error("No PDF files specified. Pass FILE arguments or use --list-prompts.")

    print(f"Processing {len(pdf_paths)} file(s) with {args.n_threads} thread(s), prompt={args.prompt_mode}\n")

    errors = []
    if args.n_threads <= 1:
        for pdf in pdf_paths:
            _, err, status = process_one(base, pdf, args.prompt_mode, args.output_dir, args.no_json)
            if err:
                print(f"  FAILED  {pdf}: {err}", file=sys.stderr)
                errors.append(pdf)
            else:
                print(f"  OK  {status}")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.n_threads) as pool:
            for pdf in pdf_paths:
                fut = pool.submit(process_one, base, pdf, args.prompt_mode, args.output_dir, args.no_json)
                futures[fut] = pdf
            for fut in as_completed(futures):
                _, err, status = fut.result()
                pdf = futures[fut]
                if err:
                    print(f"  FAILED  {pdf}: {err}", file=sys.stderr)
                    errors.append(pdf)
                else:
                    print(f"  OK  {status}")

    print(f"\nDone. {len(pdf_paths) - len(errors)}/{len(pdf_paths)} succeeded.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
