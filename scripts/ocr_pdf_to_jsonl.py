from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR a scanned PDF into JSONL chunks for the local knowledge base."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the scanned PDF file")
    parser.add_argument("output_path", type=Path, help="Destination JSONL file")
    parser.add_argument("--source", default="dsm5", help="Knowledge source label, for example dsm5")
    parser.add_argument("--title-prefix", default="OCR Chunk", help="Chunk title prefix")
    parser.add_argument("--id-prefix", default="ocr-chunk", help="Chunk id prefix")
    parser.add_argument("--pages-per-chunk", type=int, default=2, help="How many PDF pages to merge into one chunk")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pages_per_chunk < 1:
        raise ValueError("--pages-per-chunk must be at least 1")

    total_pages = pdf_page_count(args.pdf_path)
    end_page = args.end_page or total_pages
    page_ranges = list(iter_page_ranges(args.start_page, end_page, args.pages_per_chunk))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="psyche-ocr-") as temp_dir:
        temp_root = Path(temp_dir)
        records: list[dict[str, object]] = []
        for chunk_index, (page_start, page_end) in enumerate(page_ranges, start=1):
            image_prefix = temp_root / f"chunk-{chunk_index:04d}"
            render_pdf_pages(args.pdf_path, image_prefix, page_start, page_end)
            text = ocr_rendered_pages(sorted(temp_root.glob(f"{image_prefix.name}-*.png")))
            cleaned = clean_text(text)
            if not cleaned:
                continue

            records.append(
                {
                    "id": f"{args.id_prefix}-{chunk_index:04d}",
                    "title": f"{args.title_prefix} pp.{page_start}-{page_end}",
                    "source": args.source,
                    "content": cleaned,
                    "tags": [args.source, f"page:{page_start}-{page_end}"],
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )

    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_page_ranges(
    start_page: int,
    end_page: int,
    pages_per_chunk: int,
) -> list[tuple[int, int]]:
    return [
        (page, min(page + pages_per_chunk - 1, end_page))
        for page in range(start_page, end_page + 1, pages_per_chunk)
    ]


def render_pdf_pages(pdf_path: Path, image_prefix: Path, start_page: int, end_page: int) -> None:
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(start_page),
            "-l",
            str(end_page),
            "-png",
            str(pdf_path),
            str(image_prefix),
        ],
        check=True,
    )


def pdf_page_count(pdf_path: Path) -> int:
    output = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise RuntimeError(f"Could not determine page count for {pdf_path}")


def ocr_rendered_pages(image_paths: list[Path]) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "rapidocr_onnxruntime is required. Install it in the project venv before using this script."
        ) from exc

    ocr = RapidOCR()
    parts: list[str] = []
    for path in image_paths:
        result, _ = ocr(str(path))
        if not result:
            continue
        parts.append("\n".join(item[1] for item in result if item[1].strip()))
    return "\n".join(parts)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
