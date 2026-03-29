#!/usr/bin/env python3
"""
test_api.py — Test the DotsOCR API with the two PDF files in the project root.

Usage:
    python test_api.py [--api-url http://localhost:8100]

This script:
 1. Checks API health
 2. Lists available prompt modes
 3. Sends each PDF for OCR
 4. Prints the results summary
"""

import argparse
import json
import sys
import httpx

def main():
    parser = argparse.ArgumentParser(description="Test the DotsOCR API")
    parser.add_argument("--api-url", default="http://localhost:8300", help="Base URL of the API")
    args = parser.parse_args()
    base = args.api_url.rstrip("/")

    print(f"=== Testing DotsOCR API at {base} ===\n")

    # 1. Health check
    print("[1/4] Health check...")
    try:
        r = httpx.get(f"{base}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        print(f"  API: {health['api']}, vLLM: {health['vllm']}")
        if health["vllm"] != "ok":
            print("  WARNING: vLLM is not available. OCR calls will fail.")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # 2. List prompts
    print("\n[2/4] Available prompt modes:")
    r = httpx.get(f"{base}/prompts", timeout=10)
    prompts = r.json()
    for k, v in prompts.items():
        print(f"  - {k}: {v}")

    # 3. OCR the two test PDFs
    pdfs = ["Globus_2023_ApJ_945_12.pdf"]
    for i, pdf in enumerate(pdfs, start=3):
        print(f"\n[{i}/4] OCR: {pdf}")
        try:
            with open(pdf, "rb") as f:
                r = httpx.post(
                    f"{base}/ocr",
                    files={"file": (pdf, f, "application/pdf")},
                    data={"prompt_mode": "prompt_layout_all_en"},
                    timeout=600,
                )
            r.raise_for_status()
            result = r.json()
            print(f"  Filename: {result['filename']}")
            print(f"  Pages: {result['num_pages']}")
            for page in result["pages"]:
                layout_count = len(page["layout"]) if isinstance(page.get("layout"), list) else 0
                md_len = len(page.get("markdown") or "")
                print(f"    Page {page['page_no']}: {layout_count} layout elements, {md_len} chars markdown")

            # Save full JSON result
            out = f"test_result_{pdf.replace('.pdf', '.json')}"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Full result saved to {out}")

            # Save merged markdown (no headers/footers) 
            md_pages = []
            for page in result["pages"]:
                md = page.get("markdown_nohf") or page.get("markdown") or ""
                if md.strip():
                    md_pages.append(md.strip())
            if md_pages:
                merged_md = "\n\n".join(md_pages) + "\n"
                md_out = f"test_result_{pdf.replace('.pdf', '.md')}"
                with open(md_out, "w", encoding="utf-8") as f:
                    f.write(merged_md)
                print(f"  Merged markdown saved to {md_out}")
        except httpx.HTTPStatusError as e:
            print(f"  HTTP error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
