# DotsOCR HTTP API

A simple HTTP API for document OCR. Upload a PDF or image, get back structured layout and text as JSON.

## Quick Start

```bash
# Start (foreground)
bash start_server.sh

# Start (background, logs to api_server.log)
bash start_server.sh --background

# Stop
bash stop_server.sh
```

The API starts on port **8300** and automatically launches a vLLM model backend on port **8000**.

## Configuration

All settings are optional environment variables:

| Variable     | Default                | Description                        |
|--------------|------------------------|------------------------------------|
| `API_PORT`   | `8300`                 | Port for the HTTP API              |
| `VLLM_HOST`  | `localhost`            | vLLM backend host                  |
| `VLLM_PORT`  | `8000`                 | vLLM backend port                  |
| `VLLM_GPU`   | `0`                    | CUDA_VISIBLE_DEVICES for vLLM      |
| `MODEL_PATH` | `./weights/DotsOCR_1_5`| Path to model weights              |
| `MODEL_NAME` | `model`                | Served model name                  |

## Endpoints

Base URL: `http://localhost:8300`

Interactive docs: `http://localhost:8300/docs`

### `GET /health`

Check if the API and vLLM backend are running.

**Response:**
```json
{"api": "ok", "vllm": "ok", "vllm_url": "http://localhost:8000/health"}
```

### `GET /prompts`

List available OCR task modes.

**Response:**
```json
{
  "prompt_layout_all_en": "Full document layout + OCR → JSON with bboxes, categories, and text (default)",
  "prompt_layout_only_en": "Layout detection only (no text extraction)",
  "prompt_ocr": "Plain text extraction (no layout info)",
  "prompt_grounding_ocr": "OCR within a given bounding box",
  "prompt_web_parsing": "Parse webpage screenshot layout into JSON",
  "prompt_scene_spotting": "Detect and recognize scene text",
  "prompt_image_to_svg": "Generate SVG code to reconstruct the image",
  "prompt_general": "General free-form prompt"
}
```

### `POST /ocr`

Upload a file for OCR processing. This is the main endpoint.

**Request:** `multipart/form-data`

| Field         | Type   | Required | Default                | Description                              |
|---------------|--------|----------|------------------------|------------------------------------------|
| `file`        | file   | yes      | —                      | PDF or image (.pdf, .jpg, .jpeg, .png)   |
| `prompt_mode` | string | no       | `prompt_layout_all_en` | OCR task type (see `GET /prompts`)       |
| `dpi`         | int    | no       | `200`                  | DPI for PDF rasterization                |
| `num_threads` | int    | no       | `16`                   | Parallel threads for multi-page PDFs     |

**Example (curl):**
```bash
curl -X POST http://localhost:8300/ocr \
  -F "file=@document.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Example (Python):**
```python
import httpx

with open("document.pdf", "rb") as f:
    r = httpx.post(
        "http://localhost:8300/ocr",
        files={"file": ("document.pdf", f, "application/pdf")},
        data={"prompt_mode": "prompt_layout_all_en"},
        timeout=600,
    )
result = r.json()
```

**Response:**
```json
{
  "filename": "document.pdf",
  "num_pages": 2,
  "prompt_mode": "prompt_layout_all_en",
  "pages": [
    {
      "page_no": 0,
      "layout": [
        {
          "bbox": [58, 62, 540, 95],
          "category": "Title",
          "text": "Document Title"
        },
        {
          "bbox": [58, 110, 540, 400],
          "category": "Text",
          "text": "Body text content..."
        },
        {
          "bbox": [60, 420, 530, 580],
          "category": "Table",
          "text": "<table>...</table>"
        },
        {
          "bbox": [100, 600, 400, 700],
          "category": "Formula",
          "text": "E = mc^2"
        },
        {
          "bbox": [60, 710, 530, 850],
          "category": "Picture",
          "text": null
        }
      ],
      "markdown": "# Document Title\n\nBody text content...\n\n<table>...</table>\n\n$$\nE = mc^2\n$$"
    }
  ]
}
```

**Layout categories:** `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`.

**Text formats by category:**
- **Table** → HTML
- **Formula** → LaTeX
- **Picture** → no text (null)
- **Everything else** → Markdown

### `POST /vllm/start`

Manually start the vLLM backend. No-op if already running.

### `POST /vllm/stop`

Stop the vLLM backend managed by this API.

## For LLM Agents

**Recommended workflow:**

1. `GET /health` — confirm `vllm` is `"ok"` before sending files
2. `POST /ocr` with the file — use `prompt_layout_all_en` for full extraction
3. Read `pages[].markdown` for text content, `pages[].layout` for structured bounding boxes

**Timeouts:** OCR can take 30-300+ seconds depending on page count. Use a timeout of at least 600s.

**Error handling:**
- `503` — vLLM backend not ready. Call `POST /vllm/start` and wait, then retry.
- `400` — bad file type or unknown prompt mode.
