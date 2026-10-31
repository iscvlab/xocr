#!/usr/bin/env python3
"""Reproducible 3-Way PDF Pipeline & Thai Glyph Benchmark for iscvlab/xocr."""

import argparse
import os
import sys
import tempfile
import time

sys.dont_write_bytecode = True
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "skills", "xocr", "scripts"))

import ocr_runner  # noqa: E402


def run_synthetic_suite():
    """Verify Thai PUA, decomposed Sara Am, and synthetic booktabs PDF."""
    print("=" * 72)
    print("SUITE 1: DETERMINISTIC THAI GLYPH & SYNTHETIC BOOKTABS VERIFICATION")
    print("=" * 72)

    raw_sample = "สัญญาจ้าง น้ำ\uf70a\uf701 น\u0e4d\u0e49\u0e32ประปา ก\u0e4d\u0e32ลังไฟฟ้า\u200b\ufeff"
    norm_sample = ocr_runner.normalize_text(raw_sample)
    expected = "สัญญาจ้าง น้ำ\u0e48\u0e34 น้ำประปา กำลังไฟฟ้า"
    assert norm_sample == expected, f"Mismatch: {norm_sample!r} != {expected!r}"
    print(f"  ✓ Thai PUA & Sara-Am Normalization: PASS ({norm_sample})")

    import fitz

    with tempfile.TemporaryDirectory() as tmpdir:
        syn_pdf = os.path.join(tmpdir, "synthetic_booktabs.pdf")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (72, 95),
            "Table 1 | Feasibility evaluation of Unlimited-Horizon Decoding.",
            fontsize=10,
        )
        # Draw 3 booktabs horizontal rules (\toprule, \midrule, \bottomrule)
        for y in (110, 130, 190):
            page.draw_line((72, y), (480, y), width=1.0)

        rows_data = [
            (124, "Method", "Peak KV Memory", "TPS (B=16)"),
            (145, "Standard Decoding", "47.00 GB", "360"),
            (165, "Chunked Decoding", "3.50 GB", "980"),
            (184, "Unlimited-Horizon", "1.77 GB", "1315"),
        ]
        for y, c1, c2, c3 in rows_data:
            page.insert_text((80, y), c1, fontsize=10)
            page.insert_text((240, y), c2, fontsize=10)
            page.insert_text((390, y), c3, fontsize=10)
        doc.save(syn_pdf)
        doc.close()

        out_dir = os.path.join(tmpdir, "out")
        res = ocr_runner.extract_pdf(syn_pdf, out_dir, mode="auto")
        assert res["tables_extracted"] == 1, f"Expected 1 table, got {res}"
        print(
            f"  ✓ Synthetic 3-Rule Booktabs Extraction: PASS "
            f"(tables={res['tables_extracted']}, elapsed={res['elapsed_ms']} ms)"
        )


def bench_legacy_pypdf_render(pdf_path):
    """Pipeline 1: Legacy pypdf text loop + 300-DPI PNG render on all pages."""
    import fitz
    import pypdf

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        reader = pypdf.PdfReader(pdf_path)
        txt = "".join(p.extract_text() or "" for p in reader.pages)
        doc = fitz.open(pdf_path)
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            pix.save(os.path.join(tmpdir, f"page_{idx + 1:03d}.png"))
        doc.close()
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"elapsed_ms": elapsed_ms, "chars": len(txt), "tables": 0}


def bench_default_pymupdf(pdf_path):
    """Pipeline 2: Default PyMuPDF get_text() + page.find_tables()."""
    import fitz

    t0 = time.perf_counter()
    doc = fitz.open(pdf_path)
    total_chars = 0
    raw_tables = 0
    for page in doc:
        total_chars += len(page.get_text("text") or "")
        for tab in page.find_tables().tables:
            rows = tab.extract() or []
            if len(rows) >= 2 and len(rows[0]) >= 2:
                raw_tables += 1
    doc.close()
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "elapsed_ms": elapsed_ms,
        "chars": total_chars,
        "raw_table_candidates": raw_tables,
    }


def bench_xocr(pdf_path):
    """Pipeline 3: /xocr Hybrid Engine (ocr_runner.py)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = ocr_runner.extract_pdf(pdf_path, tmpdir, mode="auto")
    return {
        "elapsed_ms": res["elapsed_ms"],
        "chars": res["total_chars"],
        "tables": res["tables_extracted"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible XOCR benchmark suites."
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=[],
        help="Optional PDF path(s) for 3-Way Pipeline timing comparison.",
    )
    args = parser.parse_args()

    run_synthetic_suite()

    for pdf_path in args.pdf:
        if not os.path.isfile(pdf_path):
            continue
        print("\n" + "=" * 72)
        print(f"SUITE 2: 3-WAY PIPELINE BENCHMARK ON {os.path.basename(pdf_path)}")
        print("=" * 72)
        r_leg = bench_legacy_pypdf_render(pdf_path)
        r_def = bench_default_pymupdf(pdf_path)
        r_xocr = bench_xocr(pdf_path)
        print(f"  1. Legacy pypdf + 300-DPI Render : {r_leg}")
        print(f"  2. Default PyMuPDF find_tables() : {r_def}")
        print(f"  3. /xocr Hybrid Engine           : {r_xocr}")


if __name__ == "__main__":
    main()
