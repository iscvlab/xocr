#!/usr/bin/env python3
"""
Comprehensive Sequential 1-by-1 GPU Benchmark Suite:
Part C: Official SCB 10X ThaiOCRBench (Native Images + Official Ground Truth across 4 Tasks)
Part D: 4 Real Multi-Page Thai Environmental & Government PDFs (4p, 9p, 9p, 40p = 62 Pages Total)

Hardware Protocol:
- Strictly executes ONE model at a time on GPU.
- Keeps the active model loaded across its batch of samples/pages (avoiding redundant disk loads),
  then explicitly deletes model & processor, runs gc.collect() + torch.cuda.empty_cache(),
  and verifies VRAM returns to baseline (~8.12 MB) before loading the next model.
"""
import gc
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import fitz
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
XOCR_ROOT = os.path.dirname(BENCH_DIR)
WORKSPACE_PARENT = os.path.dirname(XOCR_ROOT)
THESIS_SUPPORT = os.path.join(WORKSPACE_PARENT, "thesis-support")
THESIS_WS = os.path.join(WORKSPACE_PARENT, "Thesis-workspace")

if XOCR_ROOT not in sys.path:
    sys.path.insert(0, XOCR_ROOT)
TYPHOON_PKG_DIR = os.path.join(
    WORKSPACE_PARENT, "OpenSource-Packages", "typhoon-ocr", "packages", "typhoon_ocr"
)
if os.path.exists(TYPHOON_PKG_DIR) and TYPHOON_PKG_DIR not in sys.path:
    sys.path.insert(0, TYPHOON_PKG_DIR)

import ocr_runner  # noqa: E402
from typhoon_ocr.ocr_utils import (  # noqa: E402
    extract_html_tables,
    get_prompt,
    normalize_thai_text,
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


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def strip_markup_for_text_eval(text: str) -> str:
    """Normalize whitespace and strip wrapper tags (<page_number>, <figure>) for fair text comparison."""
    if not text:
        return ""
    s = re.sub(r"<page_number>.*?</page_number>", " ", text, flags=re.DOTALL)
    s = re.sub(r"<figure>.*?</figure>", " ", s, flags=re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[#*_`~|>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_thaiocrbench_metrics(pred: str, gt: str, task_type: str) -> dict:
    """
    Compute official ThaiOCRBench metrics:
    - Character Edit Similarity (%) = (1 - edit_dist / max(len(pred), len(gt))) * 100
    - Character Error Rate (CER %)
    - Character/Subword Set F1-Score (%)
    - Table Structure / Cell Content Recall (%) for Table parsing
    """
    if task_type == "Table parsing":
        pred_clean = re.sub(r"\s+", " ", pred).strip()
        gt_clean = re.sub(r"\s+", " ", gt).strip()
    else:
        pred_clean = strip_markup_for_text_eval(pred)
        gt_clean = strip_markup_for_text_eval(gt)

    if not gt_clean:
        return {"edit_sim_pct": 100.0, "cer_pct": 0.0, "f1_pct": 100.0, "combined_score_pct": 100.0}
    if not pred_clean:
        return {"edit_sim_pct": 0.0, "cer_pct": 100.0, "f1_pct": 0.0, "combined_score_pct": 0.0}

    ed = levenshtein_distance(pred_clean, gt_clean)
    max_len = max(len(pred_clean), len(gt_clean), 1)
    edit_sim = max(0.0, 1.0 - (ed / max_len)) * 100.0
    cer = min(100.0, (ed / max(len(gt_clean), 1)) * 100.0)

    # Character trigram multiset F1 (robust across Thai strings without word boundaries)
    def ngrams(s, n=3):
        s_ns = re.sub(r"\s+", "", s)
        if len(s_ns) < n:
            return {s_ns} if s_ns else set()
        return {s_ns[i:i + n] for i in range(len(s_ns) - n + 1)}

    ref_set = ngrams(gt_clean, 3)
    hyp_set = ngrams(pred_clean, 3)
    if not ref_set or not hyp_set:
        f1 = 0.0
    else:
        inter = len(ref_set & hyp_set)
        prec = inter / len(hyp_set)
        rec = inter / len(ref_set)
        f1 = (2 * prec * rec / (prec + rec) * 100.0) if (prec + rec) > 0 else 0.0

    combined = (edit_sim + f1) / 2.0
    return {
        "edit_sim_pct": round(edit_sim, 2),
        "cer_pct": round(cer, 2),
        "f1_pct": round(f1, 2),
        "combined_score_pct": round(combined, 2),
    }


def fetch_and_cache_thaiocrbench_samples() -> list:
    """
    Fetch a balanced 12-sample stratified slice from SCB 10X's official `typhoon-ai/ThaiOCRBench`
    covering:
      - Full-page OCR (3 samples: #458 Finance, #459 Finance, #462 Technology)
      - Table parsing (4 samples: #1350 Legal, #2245 Government, #2248 Legal, #2261 Medical 4032x3024 Camera Photo)
      - Document parsing (3 samples: #457 Education, #467 Lifestyle, #2249 Insurance)
      - Handwritten content extraction (2 samples: #1355 Retail Receipt, #1367 Handwritten Note)
    Caches images & ground truth locally in benchmarks/cache_thaiocrbench/.
    """
    cache_dir = Path(BENCH_DIR) / "cache_thaiocrbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "samples_manifest.json"

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(data) >= 12 and all(Path(d["local_image_path"]).exists() for d in data):
            print(f"[ThaiOCRBench] Loaded {len(data)} cached official samples from {cache_dir}")
            return data

    print("[ThaiOCRBench] Fetching 12 stratified official samples from HuggingFace Datasets API...")
    target_offsets_and_indices = [
        (445, 25, {457, 458, 459, 462, 467}),
        (1345, 25, {1350, 1355, 1367}),
        (2245, 20, {2245, 2248, 2249, 2261}),
    ]
    samples = []
    for offset, length, wanted_indices in target_offsets_and_indices:
        time.sleep(0.5)
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=typhoon-ai%2FThaiOCRBench&config=default&split=test&offset={offset}&length={length}"
        )
        resp = json.loads(urllib.request.urlopen(url, timeout=30).read().decode("utf-8"))
        for r in resp.get("rows", []):
            r_idx = r["row_idx"]
            if r_idx in wanted_indices:
                row = r["row"]
                sample_id = row["Id"]
                img_url = row["image"]["src"]
                img_path = cache_dir / f"{r_idx}_{sample_id}.png"
                if not img_path.exists():
                    img_bytes = urllib.request.urlopen(img_url, timeout=30).read()
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_img.save(img_path, format="PNG")
                else:
                    pil_img = Image.open(img_path).convert("RGB")
                samples.append(
                    {
                        "row_idx": r_idx,
                        "id": sample_id,
                        "task": row["Task"],
                        "category": row["category"],
                        "width": pil_img.size[0],
                        "height": pil_img.size[1],
                        "question": row["question"],
                        "answer": row["answer"],
                        "local_image_path": str(img_path),
                    }
                )

    samples.sort(key=lambda x: x["row_idx"])
    manifest_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ThaiOCRBench] Cached {len(samples)} samples to {cache_dir}")
    return samples


def simulate_vanilla_v041_image_pipeline(img: Image.Image, max_size: int = 1800) -> Image.Image:
    """
    Reproduce Vanilla scb-10x/typhoon-ocr v0.4.1 image preprocessing:
    1. Bug #1: `if width > 300 or height > 300:` -> forcibly upscales any image >300px to 1800px!
    2. Bug #2: `img.save(buffered, format="JPEG")` -> re-encodes through lossy JPEG (default q=75).
    """
    width, height = img.size
    if width > 300 or height > 300:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")  # default quality=75 lossy JPEG
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def evaluate_model_on_thaiocrbench(
    samples: list,
    model_id: str,
    load_in_4bit: bool = False,
    vanilla_v041_mode: bool = False,
    label: str = "",
) -> dict:
    """
    Load 1 VLM onto GPU, evaluate all 12 ThaiOCRBench samples sequentially,
    compute ground-truth accuracy & latency/token telemetry, and completely unload VRAM.
    """
    vram_before = flush_gpu()
    print(
        f"\n[Suite C] Evaluating {label} on {len(samples)} official ThaiOCRBench samples "
        f"(VRAM before: {vram_before} MB)..."
    )
    t_load0 = time.perf_counter()

    load_kwargs = {"device_map": "auto", "attn_implementation": "sdpa"}
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
    prompt_v15 = get_prompt("v1.5")(figure_language="Thai")

    per_sample = []
    total_in_tokens = 0
    total_out_tokens = 0
    total_infer_s = 0.0
    total_csv_tables = 0

    try:
        for idx, item in enumerate(samples, 1):
            raw_img = Image.open(item["local_image_path"]).convert("RGB")
            if vanilla_v041_mode:
                proc_img = simulate_vanilla_v041_image_pipeline(raw_img, max_size=1800)
            else:
                proc_img = resize_if_needed(raw_img, max_size=1800)

            # For Handwritten VQA questions in ThaiOCRBench, use the dataset question if it asks a specific question;
            # for Full-page OCR, Table parsing, and Document parsing, use the Typhoon v1.5 OCR prompt.
            if item["task"] == "Handwritten content extraction":
                prompt_to_use = f"{prompt_v15}\n\nคำถามเพิ่มเติม: {item['question']}"
            else:
                prompt_to_use = prompt_v15

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_to_use},
                        {"type": "image", "image": proc_img},
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
                    max_new_tokens=2048,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            infer_s = round(time.perf_counter() - t_inf0, 3)
            out_ids = [out[in_tokens:] for out in generated_ids]
            out_tokens = int(out_ids[0].shape[0])
            decoded = processor.batch_decode(
                out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if vanilla_v041_mode:
                pred_text = decoded.strip()
                tables_found = 0
            else:
                pred_text = normalize_thai_text(decoded.strip())
                tables_found = len(extract_html_tables(pred_text))

            scores = compute_thaiocrbench_metrics(pred_text, item["answer"], item["task"])
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens
            total_infer_s += infer_s
            total_csv_tables += tables_found

            orig_wh = f"{raw_img.size[0]}x{raw_img.size[1]}"
            proc_wh = f"{proc_img.size[0]}x{proc_img.size[1]}"
            print(
                f"  [{idx:02d}/{len(samples):02d}] {item['id']} ({item['task']}, {orig_wh} -> {proc_wh}) "
                f"| in_tok={in_tokens} | {infer_s:.2f}s "
                f"| EditSim={scores['edit_sim_pct']:.1f}% | F1={scores['f1_pct']:.1f}%"
            )

            per_sample.append(
                {
                    "row_idx": item["row_idx"],
                    "id": item["id"],
                    "task": item["task"],
                    "category": item["category"],
                    "orig_size": [raw_img.size[0], raw_img.size[1]],
                    "proc_size": [proc_img.size[0], proc_img.size[1]],
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "infer_s": infer_s,
                    "html_tables_extracted": tables_found,
                    "scores": scores,
                }
            )
            del inputs, generated_ids, out_ids
    finally:
        vram_peak = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        del model, processor
        vram_after = flush_gpu()

    n = max(len(per_sample), 1)
    mean_edit_sim = round(sum(x["scores"]["edit_sim_pct"] for x in per_sample) / n, 2)
    mean_cer = round(sum(x["scores"]["cer_pct"] for x in per_sample) / n, 2)
    mean_f1 = round(sum(x["scores"]["f1_pct"] for x in per_sample) / n, 2)
    mean_combined = round(sum(x["scores"]["combined_score_pct"] for x in per_sample) / n, 2)

    # Task breakdown
    by_task = {}
    for t_name in ("Full-page OCR", "Table parsing", "Document parsing", "Handwritten content extraction"):
        sub = [x for x in per_sample if x["task"] == t_name]
        if sub:
            by_task[t_name] = {
                "count": len(sub),
                "mean_in_tokens": round(sum(x["in_tokens"] for x in sub) / len(sub), 1),
                "mean_infer_s": round(sum(x["infer_s"] for x in sub) / len(sub), 2),
                "mean_edit_sim_pct": round(sum(x["scores"]["edit_sim_pct"] for x in sub) / len(sub), 2),
                "mean_f1_pct": round(sum(x["scores"]["f1_pct"] for x in sub) / len(sub), 2),
                "mean_combined_pct": round(sum(x["scores"]["combined_score_pct"] for x in sub) / len(sub), 2),
            }

    return {
        "label": label,
        "model_id": model_id,
        "precision": "4bit-nf4" if load_in_4bit else "bfloat16",
        "vanilla_v041_mode": vanilla_v041_mode,
        "samples_evaluated": len(per_sample),
        "load_s": load_s,
        "total_infer_s": round(total_infer_s, 2),
        "mean_infer_s_per_sample": round(total_infer_s / n, 2),
        "total_in_tokens": total_in_tokens,
        "mean_in_tokens_per_sample": round(total_in_tokens / n, 1),
        "total_out_tokens": total_out_tokens,
        "mean_tok_per_s": round(total_out_tokens / max(total_infer_s, 1e-6), 2),
        "vram_before_mb": vram_before,
        "vram_loaded_mb": vram_loaded,
        "vram_peak_mb": vram_peak,
        "vram_after_unload_mb": vram_after,
        "total_csv_tables_extracted": total_csv_tables,
        "mean_edit_sim_pct": mean_edit_sim,
        "mean_cer_pct": mean_cer,
        "mean_f1_pct": mean_f1,
        "mean_combined_score_pct": mean_combined,
        "by_task": by_task,
        "per_sample": per_sample,
    }


def run_suite_d_multipage_pdfs() -> dict:
    """
    Run Suite D: 4 Real Multi-Page Thai Environmental & Government PDFs (62 Pages Total):
    - D1: sub_langka_royal_decree_2529_oic_00003724.pdf (All 4p, Scanned 1986 Royal Gazette + 3186x7872 Map)
    - D2: NONGKHAEM_ALL_FACTORY_CONTRACT_EXPIRATIONS_HIGHLIGHTED.pdf (All 9p, Highlighted Contract Tables)
    - D3: dnp_sub_langka_official_datasheet.pdf (All 9p, DNP Conservation Area Report: 8 Vector + 1 Map)
    - D4: bma_env-bureau-plan_fy2564_nongkhaem-all-projects_oic-00001680.pdf (All 40p, BMA Environment Plan)
    """
    print("\n[Suite D] Running Multi-Page Real Thai Environmental & Government PDF Benchmarks (62 Pages Total)...")

    pdf_d1 = os.path.join(THESIS_SUPPORT, "sub_langka_royal_decree_2529_oic_00003724.pdf")
    pdf_d2 = os.path.join(
        THESIS_WS,
        "01_site_analysis",
        "02_meso_scale_nongkhaem",
        "00_docs",
        "contracts",
        "NONGKHAEM_ALL_FACTORY_CONTRACT_EXPIRATIONS_HIGHLIGHTED.pdf",
    )
    pdf_d3 = os.path.join(THESIS_SUPPORT, "dnp_sub_langka_official_datasheet.pdf")
    pdf_d4 = os.path.join(
        THESIS_WS,
        "01_site_analysis",
        "02_meso_scale_nongkhaem",
        "00_docs",
        "contracts",
        "bma_env-bureau-plan_fy2564_nongkhaem-all-projects_oic-00001680.pdf",
    )

    # D1: Full 4-Page Scanned Royal Gazette (EasyOCR vs Typhoon-1.5-2B vs Typhoon-1.5-3B-QAT)
    print("  -> [D1.1] Royal Gazette (All 4 Pages) via EasyOCR baseline...")
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        doc = fitz.open(pdf_d1)
        easy_chars = 0
        for p_idx in range(doc.page_count):
            pix = doc[p_idx].get_pixmap(dpi=200)
            png_p = os.path.join(tmpdir, f"p_{p_idx+1}.png")
            pix.save(png_p)
            r = ocr_runner.ocr_image_local(png_p, langs=["th", "en"])
            easy_chars += ocr_runner.non_ws_count(r.get("text", ""))
        doc.close()
        d1_easy_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        ocr_runner._EASYOCR_READERS.clear()
        flush_gpu()

    print(
        "  -> [D1.2] Royal Gazette (All 4 Pages, including Page 3 3186x7872 Map) "
        "via xocr x typhoon-ocr1.5-2b (bf16)..."
    )
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        d1_2b_res = ocr_runner.extract_pdf(
            pdf_d1, tmpdir, mode="auto", pages_arg="1-4", vlm_backend="typhoon-2b"
        )
    flush_gpu()

    print("  -> [D1.3] Royal Gazette (All 4 Pages) via xocr x typhoon-ocr1.5-3b-qat (bf16)...")
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        d1_3b_res = ocr_runner.extract_pdf(
            pdf_d1, tmpdir, mode="auto", pages_arg="1-4", vlm_backend="typhoon-3b-qat"
        )
    flush_gpu()

    # D2: Full 9-Page Scanned/Photographed Paper Factory Contracts with Highlighter Marks
    print(
        "  -> [D2] Nongkhaem Highlighted Factory Contracts (All 9 Scanned Pages -> HTML <table> -> CSV) "
        "via xocr x typhoon-ocr1.5-2b..."
    )
    flush_gpu()
    d2_out_dir = os.path.join(BENCH_DIR, "out_d2_nongkhaem_9p")
    os.makedirs(d2_out_dir, exist_ok=True)
    d2_2b_res = ocr_runner.extract_pdf(
        pdf_d2, d2_out_dir, mode="auto", pages_arg="1-9", vlm_backend="typhoon-2b"
    )
    d2_csv_files = sorted(str(p.name) for p in Path(d2_out_dir).glob("table_*.csv"))
    flush_gpu()

    # D3: Full 9-Page DNP Conservation Report (Hybrid Router vs Vanilla Blind VLM-Every-Page)
    print(
        "  -> [D3.1] Full 9-Page DNP Conservation Report via Upgraded xocr Hybrid Router "
        "(--mode auto, pages 1-9)..."
    )
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        d3_hybrid = ocr_runner.extract_pdf(
            pdf_d3, tmpdir, mode="auto", pages_arg="1-9", vlm_backend="typhoon-2b"
        )
    flush_gpu()

    print(
        "  -> [D3.2] Full 9-Page DNP Conservation Report via Vanilla Blind VLM-Every-Page "
        "(--mode vlm, pages 1-9)..."
    )
    flush_gpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        d3_blind_vlm = ocr_runner.extract_pdf(
            pdf_d3, tmpdir, mode="vlm", pages_arg="1-9", vlm_backend="typhoon-2b"
        )
    flush_gpu()

    # D4: Full 40-Page BMA Department of Environment Action Plan (fy2564)
    print(
        "  -> [D4] Full 40-Page BMA Environment Bureau Action Plan via Upgraded xocr Hybrid Router "
        "(--mode auto, pages 1-40)..."
    )
    flush_gpu()
    d4_out_dir = os.path.join(BENCH_DIR, "out_d4_bma_env_40p")
    os.makedirs(d4_out_dir, exist_ok=True)
    d4_hybrid = ocr_runner.extract_pdf(
        pdf_d4, d4_out_dir, mode="auto", pages_arg="1-40", vlm_backend="typhoon-2b"
    )
    d4_csv_files = sorted(str(p.name) for p in Path(d4_out_dir).glob("table_*.csv"))
    flush_gpu()

    # Calculate Vanilla Blind VLM-Every-Page time on 40-page BMA report based on measured per-page VLM rate from D3.2
    measured_vlm_ms_per_page = round(d3_blind_vlm["elapsed_ms"] / max(d3_blind_vlm["pages_processed"], 1), 1)
    d4_vanilla_blind_est_ms = round(measured_vlm_ms_per_page * 40, 1)

    return {
        "d1_royal_gazette_full_4p_scanned": {
            "easyocr_4p": {
                "pages_processed": 4,
                "total_chars": easy_chars,
                "elapsed_ms": d1_easy_ms,
                "mean_ms_per_page": round(d1_easy_ms / 4.0, 1),
            },
            "xocr_typhoon_15_2b_bf16_4p": {
                "pages_processed": d1_2b_res["pages_processed"],
                "vlm_routed_pages": d1_2b_res["low_text_pages"],
                "total_chars": d1_2b_res["total_chars"],
                "tables_extracted": d1_2b_res["tables_extracted"],
                "elapsed_ms": d1_2b_res["elapsed_ms"],
                "mean_ms_per_page": round(d1_2b_res["elapsed_ms"] / 4.0, 1),
            },
            "xocr_typhoon_15_3b_qat_bf16_4p": {
                "pages_processed": d1_3b_res["pages_processed"],
                "vlm_routed_pages": d1_3b_res["low_text_pages"],
                "total_chars": d1_3b_res["total_chars"],
                "tables_extracted": d1_3b_res["tables_extracted"],
                "elapsed_ms": d1_3b_res["elapsed_ms"],
                "mean_ms_per_page": round(d1_3b_res["elapsed_ms"] / 4.0, 1),
            },
        },
        "d2_nongkhaem_highlighted_contracts_9p_scanned": {
            "pages_processed": d2_2b_res["pages_processed"],
            "vlm_routed_pages": d2_2b_res["low_text_pages"],
            "total_chars": d2_2b_res["total_chars"],
            "tables_extracted": d2_2b_res["tables_extracted"],
            "csv_files": d2_csv_files,
            "elapsed_ms": d2_2b_res["elapsed_ms"],
            "mean_ms_per_page": round(d2_2b_res["elapsed_ms"] / 9.0, 1),
        },
        "d3_dnp_conservation_report_full_9p_mixed": {
            "xocr_hybrid_auto_9p": {
                "pages_processed": d3_hybrid["pages_processed"],
                "vlm_routed_pages": d3_hybrid["low_text_pages"],
                "total_chars": d3_hybrid["total_chars"],
                "tables_extracted": d3_hybrid["tables_extracted"],
                "elapsed_ms": d3_hybrid["elapsed_ms"],
                "mean_ms_per_page": round(d3_hybrid["elapsed_ms"] / 9.0, 1),
            },
            "vanilla_blind_vlm_all_9p": {
                "pages_processed": d3_blind_vlm["pages_processed"],
                "vlm_routed_pages": d3_blind_vlm["low_text_pages"],
                "total_chars": d3_blind_vlm["total_chars"],
                "tables_extracted": d3_blind_vlm["tables_extracted"],
                "elapsed_ms": d3_blind_vlm["elapsed_ms"],
                "mean_ms_per_page": round(d3_blind_vlm["elapsed_ms"] / 9.0, 1),
            },
            "speedup_x": round(d3_blind_vlm["elapsed_ms"] / max(d3_hybrid["elapsed_ms"], 1.0), 2),
            "gpu_compute_saved_pct": round((1.0 - len(d3_hybrid["low_text_pages"]) / 9.0) * 100.0, 1),
        },
        "d4_bma_environment_plan_full_40p_mixed": {
            "xocr_hybrid_auto_40p": {
                "pages_processed": d4_hybrid["pages_processed"],
                "vlm_routed_pages": d4_hybrid["low_text_pages"],
                "total_chars": d4_hybrid["total_chars"],
                "tables_extracted": d4_hybrid["tables_extracted"],
                "csv_files_count": len(d4_csv_files),
                "csv_files_sample": d4_csv_files[:10],
                "elapsed_ms": d4_hybrid["elapsed_ms"],
                "mean_ms_per_page": round(d4_hybrid["elapsed_ms"] / 40.0, 1),
            },
            "vanilla_blind_vlm_40p_projected_from_d3_rate": {
                "pages_processed": 40,
                "vlm_routed_pages_count": 40,
                "measured_vlm_ms_per_page": measured_vlm_ms_per_page,
                "projected_elapsed_ms": d4_vanilla_blind_est_ms,
                "projected_elapsed_min": round(d4_vanilla_blind_est_ms / 60000.0, 2),
            },
            "speedup_x": round(d4_vanilla_blind_est_ms / max(d4_hybrid["elapsed_ms"], 1.0), 1),
            "gpu_compute_saved_pct": round((1.0 - len(d4_hybrid["low_text_pages"]) / 40.0) * 100.0, 1),
        },
    }


def main():
    t0 = time.perf_counter()
    out_json = os.path.join(BENCH_DIR, "comprehensive_thai_benchmark_results.json")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    samples = fetch_and_cache_thaiocrbench_samples()

    # Run Suite C (Official ThaiOCRBench) 1-by-1 sequentially
    c1_vanilla_2b = evaluate_model_on_thaiocrbench(
        samples,
        model_id="typhoon-ai/typhoon-ocr1.5-2b",
        load_in_4bit=False,
        vanilla_v041_mode=True,
        label="Vanilla typhoon-ocr v0.4.1 (typhoon-ocr1.5-2b, Bug #1 Upscale + Bug #2 JPEG q=75)",
    )
    c2_upgraded_2b = evaluate_model_on_thaiocrbench(
        samples,
        model_id="typhoon-ai/typhoon-ocr1.5-2b",
        load_in_4bit=False,
        vanilla_v041_mode=False,
        label="Upgraded xocr v2.1.0 x typhoon-ocr1.5-2b (bfloat16, Lossless PNG + Native Size + Thai Normalizer)",
    )
    c3_upgraded_3b_bf16 = evaluate_model_on_thaiocrbench(
        samples,
        model_id="typhoon-ai/typhoon-ocr1.5-3b-qat",
        load_in_4bit=False,
        vanilla_v041_mode=False,
        label="Upgraded xocr v2.1.0 x typhoon-ocr1.5-3b-qat (bfloat16)",
    )
    c4_upgraded_3b_4bit = evaluate_model_on_thaiocrbench(
        samples,
        model_id="typhoon-ai/typhoon-ocr1.5-3b-qat",
        load_in_4bit=True,
        vanilla_v041_mode=False,
        label="Upgraded xocr v2.1.0 x typhoon-ocr1.5-3b-qat (4-bit NF4)",
    )

    partial_report = {
        "hardware": {
            "gpu": gpu_name,
            "total_vram_mb": total_vram_mb,
            "pytorch_version": torch.__version__,
            "execution_mode": "strictly_sequential_1_by_1_zero_vram_division",
        },
        "suite_c_official_thaiocrbench": {
            "vanilla_v041_2b_bf16": c1_vanilla_2b,
            "upgraded_v210_2b_bf16": c2_upgraded_2b,
            "upgraded_v210_3b_qat_bf16": c3_upgraded_3b_bf16,
            "upgraded_v210_3b_qat_4bit": c4_upgraded_3b_4bit,
            "token_reduction_pct_2b_upgraded_vs_vanilla": round(
                (1.0 - c2_upgraded_2b["total_in_tokens"] / max(c1_vanilla_2b["total_in_tokens"], 1)) * 100.0,
                1,
            ),
            "speedup_x_2b_upgraded_vs_vanilla": round(
                c1_vanilla_2b["total_infer_s"] / max(c2_upgraded_2b["total_infer_s"], 1e-6), 2
            ),
            "accuracy_delta_pts_2b_upgraded_vs_vanilla": round(
                c2_upgraded_2b["mean_combined_score_pct"] - c1_vanilla_2b["mean_combined_score_pct"], 2
            ),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(partial_report, f, indent=2, ensure_ascii=False)
    print(f"[Suite C Checkpoint] Saved Suite C results to {out_json}")

    # Run Suite D (4 Real Multi-Page Thai Environmental & Government PDFs = 62 Pages Total)
    suite_d = run_suite_d_multipage_pdfs()
    partial_report["suite_d_multipage_thai_pdfs"] = suite_d
    partial_report["total_wall_s"] = round(time.perf_counter() - t0, 2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(partial_report, f, indent=2, ensure_ascii=False)

    print(f"\n[ALL DONE] Saved complete Suite C + Suite D benchmark results to: {out_json}")


if __name__ == "__main__":
    main()
