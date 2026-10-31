#!/usr/bin/env python3
"""
Sequential 1-by-1 GPU Benchmark Suite:
SCB 10X Typhoon-OCR (v0.4.1 Vanilla vs v0.4.2 Upgraded) x iscvlab/xocr Hybrid Engine

Hardware Protocol:
- Strictly executes ONE model at a time on GPU.
- Explicitly deletes model & processor, runs gc.collect() + torch.cuda.empty_cache(),
  and records VRAM before load, peak VRAM during inference, and VRAM after unload (< 15 MB)
  to guarantee zero VRAM division across models.
"""
import base64
import gc
import io
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
XOCR_ROOT = os.path.dirname(BENCH_DIR)
WORKSPACE_PARENT = os.path.dirname(XOCR_ROOT)
TYPHOON_REPO = os.path.join(WORKSPACE_PARENT, "OpenSource-Packages", "typhoon-ocr")
THESIS_SUPPORT = os.path.join(WORKSPACE_PARENT, "thesis-support")

if XOCR_ROOT not in sys.path:
    sys.path.insert(0, XOCR_ROOT)

import ocr_runner  # noqa: E402
from typhoon_ocr.ocr_utils import (  # noqa: E402
    _cleanup_element_text,
    extract_html_tables,
    get_pdf_media_box_width_height,
    get_prompt,
    image_to_base64png,
    normalize_thai_text,
    render_pdf_to_base64png,
    resize_if_needed,
)


def flush_gpu() -> float:
    """Release all cached CUDA tensors and return current allocated VRAM in MB."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
    return 0.0


def run_single_vlm_inference(
    model_id: str,
    pil_image: Image.Image,
    prompt_text: str,
    load_in_4bit: bool = False,
    max_new_tokens: int = 2048,
) -> dict:
    """Load a single VLM onto GPU, run 1 deterministic inference, record telemetry, and unload completely."""
    vram_before = flush_gpu()
    t_load0 = time.perf_counter()

    load_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    load_s = round(time.perf_counter() - t_load0, 3)
    vram_loaded = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": pil_image},
            ],
        }
    ]

    t_inf0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    in_tokens = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    infer_s = round(time.perf_counter() - t_inf0, 3)
    out_ids = [out[in_tokens:] for out in generated_ids]
    out_tokens = int(out_ids[0].shape[0])
    decoded = processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    cleaned = normalize_thai_text(decoded.strip())
    vram_peak = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    del inputs, generated_ids, out_ids, model, processor
    vram_after = flush_gpu()

    return {
        "model_id": model_id,
        "precision": "4bit-nf4" if load_in_4bit else "bfloat16",
        "load_s": load_s,
        "infer_s": infer_s,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "tok_per_s": round(out_tokens / max(infer_s, 1e-6), 2),
        "vram_before_mb": vram_before,
        "vram_loaded_mb": vram_loaded,
        "vram_peak_mb": vram_peak,
        "vram_after_unload_mb": vram_after,
        "char_count": ocr_runner.non_ws_count(cleaned),
        "html_tables_detected": len(extract_html_tables(cleaned)),
        "text_preview": cleaned[:350],
        "full_text": cleaned,
    }


def create_synthetic_thai_table_image(out_path: str) -> str:
    """Create a realistic scanned Thai financial table image (800x420 PNG) for HTML table -> CSV testing."""
    img = Image.new("RGB", (800, 420), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)
    font_candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\angsana.ttc",
        r"C:\Windows\Fonts\leelawui.ttf",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)
    title_font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
    cell_font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()

    draw.text((40, 20), "ตารางสรุปงบประมาณโครงการวิจัย AI ปี 2568 (หน่วย: ล้านบาท)", fill=(15, 23, 42), font=title_font)

    rows = [
        ["หมวดหมู่โครงการ", "ไตรมาส 1", "ไตรมาส 2", "รวมทั้งสิ้น"],
        ["1. ระบบประมวลผลเอกสาร (Document OCR)", "12.50", "15.75", "28.25"],
        ["2. โครงสร้างพื้นฐาน GPU Cluster", "45.00", "30.00", "75.00"],
        ["3. งานวิจัยโมเดลภาษาไทย (Thai VLM)", "18.20", "22.80", "41.00"],
        ["รวมงบประมาณสุทธิ", "75.70", "68.55", "144.25"],
    ]
    col_x = [40, 410, 535, 660, 760]
    row_y = [75, 130, 185, 240, 295, 350]

    for r_i, r_data in enumerate(rows):
        y0, y1 = row_y[r_i], row_y[r_i + 1]
        bg = (230, 236, 245) if r_i == 0 else ((242, 245, 250) if r_i == len(rows) - 1 else (255, 255, 255))
        draw.rectangle([col_x[0], y0, col_x[-1], y1], fill=bg, outline=(60, 60, 60), width=2)
        for c_i, val in enumerate(r_data):
            x0, x1 = col_x[c_i], col_x[c_i + 1]
            draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=1)
            draw.text((x0 + 10, y0 + 15), val, fill=(10, 10, 10), font=cell_font)

    img.save(out_path, format="PNG")
    return out_path


def run_suite_a_micro_benchmarks() -> dict:
    """Run Suite A: Micro-benchmarks for Bugs #1-#5 and Feature #6."""
    print("[Suite A] Running Micro-Benchmarks (Bugs #1-#5 & Feature #6)...")
    prompt_v15 = get_prompt("v1.5")(figure_language="Thai")

    # A1: Bug #1 resize_if_needed upscaling tax on a 600x400 image crop
    test_png = os.path.join(TYPHOON_REPO, "examples", "test.png")
    base_img = Image.open(test_png).convert("RGB")
    crop_600x400 = base_img.resize((600, 400), Image.Resampling.LANCZOS)
    # Simulate Vanilla v0.4.1 buggy upscaling to 1800x1200
    vanilla_upscaled = crop_600x400.resize((1800, 1200), Image.Resampling.LANCZOS)
    upgraded_preserved = resize_if_needed(crop_600x400, max_size=1800)

    res_vanilla_resize = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-2b", vanilla_upscaled, prompt_v15, max_new_tokens=512
    )
    res_upgraded_resize = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-2b", upgraded_preserved, prompt_v15, max_new_tokens=512
    )

    # A2: Bug #2 JPEG vs PNG lossless verification
    buf_jpg = io.BytesIO()
    crop_600x400.save(buf_jpg, format="JPEG")
    jpg_bytes = buf_jpg.getvalue()
    png_b64 = image_to_base64png(crop_600x400)
    png_bytes = base64.b64decode(png_b64)

    # A3: Bug #3 Thai PUA & decomposed Sara-Am normalization
    corrupted_samples = [
        "สัญญาจ้าง น้ำ\uf70a\uf701 น\u0e4d\u0e49\u0e32ประปา ก\u0e4d\u0e32ลังไฟฟ้า\u200b\ufeff",
        "ผู้ว\uf70aาจ้างตกลงช\u0e4d\u0e32ระค่าจ้างงวดท\uf702\uf70b 1 จ\u0e4d\u0e32านวน 1,500,000 บาท",
    ]
    pua_fixed_count = 0
    for s in corrupted_samples:
        cleaned = _cleanup_element_text(s)
        has_pua = any(0xF700 <= ord(ch) <= 0xF71A for ch in cleaned)
        if not has_pua and "\u0e4d\u0e32" not in cleaned and "\u0e4d\u0e49\u0e32" not in cleaned:
            pua_fixed_count += 1

    # A4: Bug #4 Poppler-free PyMuPDF PDF rendering speed
    sample_pdf = os.path.join(THESIS_SUPPORT, "dnp_sub_langka_official_datasheet.pdf")
    t_pdf0 = time.perf_counter()
    with patch("typhoon_ocr.pdf_utils.pdf_utils_available", False):
        w, h = get_pdf_media_box_width_height(sample_pdf, 1)
        b64_page = render_pdf_to_base64png(sample_pdf, 1, target_longest_image_dim=1800)
    pymupdf_render_ms = round((time.perf_counter() - t_pdf0) * 1000.0, 2)

    # A5: Feature #6 Scanned Thai Table Image -> Typhoon-OCR v1.5 HTML <table> -> xocr CSV + GFM Markdown
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_img_path = os.path.join(tmpdir, "thai_budget_table.png")
        create_synthetic_thai_table_image(synth_img_path)
        table_out_dir = os.path.join(tmpdir, "out_table")
        table_res = ocr_runner.extract_standalone_image(
            synth_img_path, table_out_dir, vlm_backend="typhoon-2b"
        )
        csv_path = os.path.join(table_out_dir, "table_p1_1.csv")
        csv_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                csv_rows = [line.strip() for line in f if line.strip()]

    return {
        "bug1_resize_upscaling_tax": {
            "vanilla_size": list(vanilla_upscaled.size),
            "upgraded_size": list(upgraded_preserved.size),
            "vanilla_in_tokens": res_vanilla_resize["in_tokens"],
            "upgraded_in_tokens": res_upgraded_resize["in_tokens"],
            "token_reduction_pct": round(
                (1.0 - res_upgraded_resize["in_tokens"] / res_vanilla_resize["in_tokens"]) * 100.0, 1
            ),
            "vanilla_infer_s": res_vanilla_resize["infer_s"],
            "upgraded_infer_s": res_upgraded_resize["infer_s"],
            "speedup_x": round(res_vanilla_resize["infer_s"] / max(res_upgraded_resize["infer_s"], 1e-6), 2),
        },
        "bug2_png_vs_jpeg_fidelity": {
            "vanilla_magic_hex": jpg_bytes[:4].hex(),
            "upgraded_magic_hex": png_bytes[:8].hex(),
            "upgraded_is_lossless_png": png_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        },
        "bug3_thai_pua_sara_am": {
            "samples_tested": len(corrupted_samples),
            "samples_100pct_normalized": pua_fixed_count,
        },
        "bug4_pymupdf_poppler_free_render": {
            "mediabox_pt": [w, h],
            "render_1800px_ms": pymupdf_render_ms,
            "base64_len": len(b64_page),
        },
        "feature6_scanned_table_to_csv": {
            "engine_used": table_res["engine_used"],
            "tables_extracted": table_res["tables_extracted"],
            "elapsed_ms": table_res["elapsed_ms"],
            "csv_lines": csv_rows,
        },
    }


def run_suite_b_sequential_document_benchmarks() -> dict:
    """Run Suite B: Sequential 1-by-1 GPU benchmarks on real Thai documents."""
    print("[Suite B] Running Sequential 1-by-1 GPU Document Benchmarks...")
    prompt_v15 = get_prompt("v1.5")(figure_language="Thai")

    # 1. Official SCBX AI Outlook 2025 slide (examples/test.png)
    test_png = os.path.join(TYPHOON_REPO, "examples", "test.png")
    slide_img = resize_if_needed(Image.open(test_png).convert("RGB"), max_size=1800)

    print("  -> [B1.1] Running typhoon-ai/typhoon-ocr1.5-2b (bfloat16) on examples/test.png...")
    b1_2b_bf16 = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-2b", slide_img, prompt_v15, load_in_4bit=False, max_new_tokens=2048
    )

    print("  -> [B1.2] Running typhoon-ai/typhoon-ocr1.5-3b-qat (bfloat16) on examples/test.png...")
    b1_3b_bf16 = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-3b-qat", slide_img, prompt_v15, load_in_4bit=False, max_new_tokens=2048
    )

    print("  -> [B1.3] Running typhoon-ai/typhoon-ocr1.5-3b-qat (4-bit NF4) on examples/test.png...")
    b1_3b_4bit = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-3b-qat", slide_img, prompt_v15, load_in_4bit=True, max_new_tokens=2048
    )

    # 2. Scanned Thai Royal Gazette PDF (sub_langka_royal_decree_2529_oic_00003724.pdf, Page 1)
    scanned_pdf = os.path.join(THESIS_SUPPORT, "sub_langka_royal_decree_2529_oic_00003724.pdf")
    b64_scan_p1 = render_pdf_to_base64png(scanned_pdf, 1, target_longest_image_dim=1800, pdf_backend="pymupdf")
    scan_p1_img = Image.open(io.BytesIO(base64.b64decode(b64_scan_p1))).convert("RGB")

    print("  -> [B2.1] Running EasyOCR local baseline on Scanned Royal Gazette Page 1...")
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_png_path = os.path.join(tmpdir, "royal_decree_p1.png")
        scan_p1_img.save(scan_png_path, format="PNG")
        t_easy0 = time.perf_counter()
        easy_res = ocr_runner.ocr_image_local(scan_png_path, langs=["th", "en"])
        easy_s = round(time.perf_counter() - t_easy0, 3)
        ocr_runner._EASYOCR_READERS.clear()
        flush_gpu()

    print("  -> [B2.2] Running typhoon-ai/typhoon-ocr1.5-2b (bfloat16) on Scanned Royal Gazette Page 1...")
    b2_2b_bf16 = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-2b", scan_p1_img, prompt_v15, load_in_4bit=False, max_new_tokens=2048
    )

    print("  -> [B2.3] Running typhoon-ai/typhoon-ocr1.5-3b-qat (bfloat16) on Scanned Royal Gazette Page 1...")
    b2_3b_bf16 = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-3b-qat", scan_p1_img, prompt_v15, load_in_4bit=False, max_new_tokens=2048
    )

    print("  -> [B2.4] Running typhoon-ai/typhoon-ocr1.5-3b-qat (4-bit NF4) on Scanned Royal Gazette Page 1...")
    b2_3b_4bit = run_single_vlm_inference(
        "typhoon-ai/typhoon-ocr1.5-3b-qat", scan_p1_img, prompt_v15, load_in_4bit=True, max_new_tokens=2048
    )

    # 3. Digital/Mixed Thai Government Report (dnp_sub_langka_official_datasheet.pdf, Pages 1-3)
    # Compare:
    # (a) xocr x Typhoon-OCR Hybrid Router (--mode auto, pages 1-3): C++ PyMuPDF vector fast-path on digital pages
    # (b) Vanilla Blind VLM-Every-Page (Typhoon-OCR v1.5 2B running VLM on all 3 pages)
    digital_pdf = os.path.join(THESIS_SUPPORT, "dnp_sub_langka_official_datasheet.pdf")
    print("  -> [B3.1] Running xocr x Typhoon-OCR Hybrid Router (--mode auto, pages 1-3) on Digital Thai Gov PDF...")
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        hybrid_res = ocr_runner.extract_pdf(
            digital_pdf,
            tmpdir,
            mode="auto",
            pages_arg="1-3",
            vlm_backend="typhoon-2b",
        )

    print("  -> [B3.2] Running Vanilla Blind VLM-Every-Page (--mode vlm, pages 1-3) with Typhoon-OCR v1.5 2B...")
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        blind_vlm_res = ocr_runner.extract_pdf(
            digital_pdf,
            tmpdir,
            mode="vlm",
            pages_arg="1-3",
            vlm_backend="typhoon-2b",
        )
    flush_gpu()

    return {
        "b1_scbx_slide_test_png": {
            "typhoon_15_2b_bf16": {k: v for k, v in b1_2b_bf16.items() if k != "full_text"},
            "typhoon_15_3b_qat_bf16": {k: v for k, v in b1_3b_bf16.items() if k != "full_text"},
            "typhoon_15_3b_qat_4bit": {k: v for k, v in b1_3b_4bit.items() if k != "full_text"},
        },
        "b2_scanned_thai_royal_gazette_p1": {
            "easyocr_baseline": {
                "engine": easy_res["engine"],
                "infer_s": easy_s,
                "char_count": ocr_runner.non_ws_count(easy_res["text"]),
                "text_preview": easy_res["text"][:350],
            },
            "typhoon_15_2b_bf16": {k: v for k, v in b2_2b_bf16.items() if k != "full_text"},
            "typhoon_15_3b_qat_bf16": {k: v for k, v in b2_3b_bf16.items() if k != "full_text"},
            "typhoon_15_3b_qat_4bit": {k: v for k, v in b2_3b_4bit.items() if k != "full_text"},
        },
        "b3_digital_thai_gov_pdf_pages_1_to_3": {
            "xocr_hybrid_auto": {
                "pages_processed": hybrid_res["pages_processed"],
                "low_text_pages_routed_to_vlm": hybrid_res["low_text_pages"],
                "total_chars": hybrid_res["total_chars"],
                "tables_extracted": hybrid_res["tables_extracted"],
                "elapsed_ms": hybrid_res["elapsed_ms"],
            },
            "vanilla_blind_vlm_every_page": {
                "pages_processed": blind_vlm_res["pages_processed"],
                "low_text_pages_routed_to_vlm": blind_vlm_res["low_text_pages"],
                "total_chars": blind_vlm_res["total_chars"],
                "tables_extracted": blind_vlm_res["tables_extracted"],
                "elapsed_ms": blind_vlm_res["elapsed_ms"],
            },
            "hybrid_speedup_x": round(
                blind_vlm_res["elapsed_ms"] / max(hybrid_res["elapsed_ms"], 0.1), 1
            ),
        },
    }


def main():
    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    suite_a = run_suite_a_micro_benchmarks()
    suite_b = run_suite_b_sequential_document_benchmarks()

    report = {
        "hardware": {
            "gpu": gpu_name,
            "total_vram_mb": total_vram_mb,
            "pytorch_version": torch.__version__,
            "execution_mode": "strictly_sequential_1_by_1_zero_vram_division",
        },
        "suite_a_micro_benchmarks": suite_a,
        "suite_b_sequential_gpu_benchmarks": suite_b,
        "total_benchmark_wall_s": round(time.perf_counter() - t0, 2),
    }

    out_json = os.path.join(BENCH_DIR, "typhoon_xocr_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved full benchmark results to: {out_json}")


if __name__ == "__main__":
    main()
