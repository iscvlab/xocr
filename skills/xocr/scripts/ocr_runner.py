#!/usr/bin/env python3
"""High-speed, zero-corruption PDF & Image OCR engine for /xocr.

Features:
1. Single-pass C++ PyMuPDF (`fitz`) coordinate-sorted text (`sort=True`) and
   Two-Stage Grid + LaTeX `booktabs` table extraction (`page.find_tables()`),
   with optional `pdfplumber` fallback.
2. Deterministic Thai/English glyph normalization (fixes legacy Thai PUA
   codepoints `U+F700..U+F71A`, decomposed Sara Am `\u0e4d\u0e32` -> `\u0e33`,
   and detects `(cid:...)` / `\ufffd` mojibake).
3. Surgical Hybrid Page Router: scores each PDF page for text density, image
   coverage, and garble ratio (`needs_ocr`), rasterizing only scanned or
   corrupted pages in `--mode auto`.
4. 4-Tier OCR Backend Ladder for scanned pages and standalone images:
   Tier 1: Live SGLang Server (`http://127.0.0.1:10000` / `infer.py` with
           exact `--pdf` / `--image_dir` / `--output_dir` flags)
   Tier 2: HuggingFace `baidu/Unlimited-OCR` VLM (`no_repeat_ngram_size=35`)
   Tier 3: Local `EasyOCR` (`['th', 'en']` / `['en']`)
   Tier 4: Local `pytesseract` / `fitz` Tesseract bridge
5. Parallel page rasterization (`--workers`) and surgical page slicing
   (`--pages "1-5,8,10-12"`).
"""

import argparse
import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=UserWarning)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Legacy Windows / Adobe Thai Private Use Area (PUA) glyph remapping table
THAI_PUA_MAP = {
    "\uf700": "\u0e10",  # Tho Than (baseless)
    "\uf701": "\u0e34",  # Sara I (shifted left)
    "\uf702": "\u0e35",  # Sara Ii (shifted left)
    "\uf703": "\u0e36",  # Sara Ue (shifted left)
    "\uf704": "\u0e37",  # Sara Uee (shifted left)
    "\uf705": "\u0e48",  # Mai Ek (low left)
    "\uf706": "\u0e49",  # Mai Tho (low left)
    "\uf707": "\u0e4a",  # Mai Tri (low left)
    "\uf708": "\u0e4b",  # Mai Chattawa (low left)
    "\uf709": "\u0e4c",  # Thanthakhat (low left)
    "\uf70a": "\u0e48",  # Mai Ek (low)
    "\uf70b": "\u0e49",  # Mai Tho (low)
    "\uf70c": "\u0e4a",  # Mai Tri (low)
    "\uf70d": "\u0e4b",  # Mai Chattawa (low)
    "\uf70e": "\u0e4c",  # Thanthakhat (low)
    "\uf70f": "\u0e0d",  # Yo Ying (baseless)
    "\uf710": "\u0e31",  # Mai Han-Akat (shifted left)
    "\uf711": "\u0e4d",  # Nikhahit (shifted left)
    "\uf712": "\u0e47",  # Maitaikhu (shifted left)
    "\uf713": "\u0e48",  # Mai Ek (high left)
    "\uf714": "\u0e49",  # Mai Tho (high left)
    "\uf715": "\u0e4a",  # Mai Tri (high left)
    "\uf716": "\u0e4b",  # Mai Chattawa (high left)
    "\uf717": "\u0e4c",  # Thanthakhat (high left)
    "\uf718": "\u0e38",  # Sara U (low)
    "\uf719": "\u0e39",  # Sara Uu (low)
    "\uf71a": "\u0e3a",  # Phinthu (low)
}
THAI_PUA_TRANS = str.maketrans(THAI_PUA_MAP)
CID_PATTERN = re.compile(r"\(cid:\d+\)")
SARA_AM_PATTERN = re.compile(r"\u0e4d([\u0e48-\u0e4b]?)\u0e32")


def non_ws_count(text):
    """Return count of non-whitespace characters in text."""
    return len(re.sub(r"\s+", "", text or ""))


def normalize_text(raw_text):
    """Normalize Thai PUA codepoints, Sara Am, and zero-width chars."""
    if not raw_text:
        return ""
    text = raw_text.translate(THAI_PUA_TRANS)
    # Normalize decomposed Nikhahit + optional tone mark + Sara Aa -> Sara Am
    text = SARA_AM_PATTERN.sub("\\1\u0e33", text)
    # Remove zero-width space / BOM artifacts inside text streams
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text


def evaluate_page_quality(raw_text, page):
    """Compute text quality metrics and decide whether a page needs OCR."""
    cid_matches = CID_PATTERN.findall(raw_text or "")
    cid_count = len(cid_matches)
    rep_count = (raw_text or "").count("\ufffd")
    cleaned = CID_PATTERN.sub("", normalize_text(raw_text or "")).strip()
    non_ws_len = non_ws_count(cleaned)
    total_len = max(1, len(raw_text or ""))
    garble_ratio = round((cid_count * 6 + rep_count) / total_len, 4)

    image_count = 0
    image_area_ratio = 0.0
    try:
        image_count = len(page.get_images(full=False))
        page_area = max(1.0, abs(page.rect.width * page.rect.height))
        img_infos = page.get_image_info() or []
        max_img_area = 0.0
        for info in img_infos:
            bbox = info.get("bbox")
            if bbox and len(bbox) == 4:
                area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if area > max_img_area:
                    max_img_area = area
        image_area_ratio = round(min(1.0, max_img_area / page_area), 4)
    except Exception:  # nosec B110
        image_count = 0

    needs_ocr = (
        non_ws_len < 25
        or garble_ratio > 0.05
        or (image_count > 0 and non_ws_len < 60)
        or (image_area_ratio >= 0.45 and non_ws_len < 250)
    )
    return {
        "clean_text": cleaned,
        "char_count": non_ws_len,
        "cid_count": cid_count,
        "replacement_count": rep_count,
        "garble_ratio": garble_ratio,
        "image_count": image_count,
        "image_area_ratio": image_area_ratio,
        "needs_ocr": needs_ocr,
    }


def parse_page_selection(pages_arg, total_pages):
    """Parse 1-based page spec like '1-5,8,11-13' into 0-based indices."""
    if not pages_arg or pages_arg.strip().lower() in ("all", "*"):
        return list(range(total_pages))
    selected = set()
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            start = max(1, int(bounds[0]))
            end = min(total_pages, int(bounds[1]))
            for p_num in range(start, end + 1):
                selected.add(p_num - 1)
        else:
            p_num = int(part)
            if 1 <= p_num <= total_pages:
                selected.add(p_num - 1)
    return sorted(selected)


def clean_table_rows(raw_rows):
    """Normalize table cells, prune empty columns/rows, and validate tables."""
    if not raw_rows:
        return []
    rows = []
    for raw_r in raw_rows:
        if not raw_r:
            continue
        norm_r = []
        has_val = False
        for cell in raw_r:
            val = normalize_text(str(cell or "")).replace("\n", " ").strip()
            val = re.sub(r"\s{2,}", " ", val)
            if val:
                has_val = True
            norm_r.append(val)
        if has_val:
            rows.append(norm_r)

    if len(rows) < 2:
        return []

    ncol = max(len(r) for r in rows)
    keep_cols = [
        ci for ci in range(ncol) if any(ci < len(r) and r[ci] for r in rows)
    ]
    if len(keep_cols) < 2:
        return []

    pruned = [
        [r[ci] if ci < len(r) else "" for ci in keep_cols] for r in rows
    ]
    multi_col_rows = sum(1 for r in pruned if sum(1 for c in r if c) >= 2)
    if multi_col_rows < 2 or (multi_col_rows / len(pruned)) < 0.60:
        return []
    total_cells = len(pruned) * len(keep_cols)
    raw_total = max(1, len(raw_rows) * ncol)
    filled_cells = sum(1 for r in pruned for c in r if c)
    if (
        total_cells > 0
        and (
            (filled_cells / total_cells) < 0.52
            or (filled_cells / raw_total) < 0.32
        )
    ):
        return []
    return pruned


def _is_booktabs_strip(rows):
    """Return True if a table region looks like a 1-2 row booktabs strip."""
    if len(rows) <= 2:
        return True
    max_nl = max(
        (str(c or "").count("\n") for r in rows for c in r), default=0
    )
    return max_nl >= 2


def _find_booktabs_rule_bboxes(page, drawings=None):
    """Detect LaTeX booktabs horizontal rule clusters (top/mid/bottomrule)."""
    if drawings is None:
        try:
            drawings = page.get_drawings() or []
        except Exception:  # nosec B110
            return []

    h_rules = []
    for drw in drawings:
        rect = drw.get("rect")
        if not rect:
            continue
        width = rect.x1 - rect.x0
        height = rect.y1 - rect.y0
        if width >= 200 and height <= 4.5:
            h_rules.append(
                [
                    float(rect.x0),
                    float(rect.y0),
                    float(rect.x1),
                    float(rect.y1),
                ]
            )

    if len(h_rules) < 3:
        return []

    h_rules.sort(key=lambda r: (r[1], r[0]))
    clusters = []
    for rule in h_rules:
        placed = False
        for cl in clusters:
            if (
                abs(rule[0] - cl["x0"]) < 25
                and abs(rule[2] - cl["x1"]) < 25
                and 0 <= (rule[1] - cl["last_y"]) < 240
            ):
                if (rule[1] - cl["last_y"]) > 3.5:
                    cl["rule_count"] += 1
                cl["y1"] = max(cl["y1"], rule[3])
                cl["last_y"] = rule[3]
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "x0": rule[0],
                    "y0": rule[1],
                    "x1": rule[2],
                    "y1": rule[3],
                    "last_y": rule[3],
                    "rule_count": 1,
                }
            )

    return [
        [cl["x0"], cl["y0"], cl["x1"], cl["y1"]]
        for cl in clusters
        if cl["rule_count"] >= 3 and (cl["y1"] - cl["y0"]) >= 25
    ]


def extract_tables_from_page(page, fitz_mod, raw_text=""):
    """Two-Stage Grid + LaTeX booktabs table extractor using PyMuPDF."""
    extracted = []
    try:
        drawings = page.get_drawings() or []
    except Exception:  # nosec B110
        drawings = []

    if not drawings:
        return extracted

    h_rules_cnt = 0
    g_rects_cnt = 0
    for drw in drawings:
        rect = drw.get("rect")
        if not rect:
            continue
        w = rect.x1 - rect.x0
        h = rect.y1 - rect.y0
        if w >= 150 and h <= 5:
            h_rules_cnt += 1
        elif w >= 25 and h >= 10 and drw.get("type") != "f":
            g_rects_cnt += 1

    if h_rules_cnt < 3 and g_rects_cnt < 8:
        return extracted

    if len(drawings) > 200 and not re.search(
        r"(?:^|\n)\s*(?:Table|Tab\.|ตาราง)\s*\d*\s*[:.|]",
        raw_text or "",
        re.IGNORECASE,
    ):
        return extracted

    try:
        raw_tabs = page.find_tables().tables
    except Exception:  # nosec B110
        raw_tabs = []

    # Pre-extract rows before any subsequent page.find_tables(clip=...) call,
    # which mutates PyMuPDF's internal table buffers on the same Page object.
    cached_tabs = []
    for tab in sorted(raw_tabs, key=lambda x: (x.bbox[1], x.bbox[0])):
        try:
            cached_tabs.append((list(tab.bbox), tab.extract() or []))
        except Exception:  # nosec B110
            pass

    groups = []
    for box, rows in cached_tabs:
        if len(rows) >= 5:
            r_ncol = max((len(r) for r in rows if r), default=1)
            r_fill = sum(
                1 for r in rows for c in (r or []) if c and str(c).strip()
            ) / max(1, len(rows) * r_ncol)
            if r_fill < 0.08:
                continue
        if (
            groups
            and abs(box[1] - groups[-1]["bbox"][3]) < 15
            and abs(box[0] - groups[-1]["bbox"][0]) < 20
            and abs(box[2] - groups[-1]["bbox"][2]) < 20
            and (
                _is_booktabs_strip(groups[-1]["last_rows"])
                or _is_booktabs_strip(rows)
            )
        ):
            groups[-1]["bbox"][3] = max(groups[-1]["bbox"][3], box[3])
            groups[-1]["rows_list"].append(rows)
            groups[-1]["last_rows"] = rows
        else:
            groups.append(
                {
                    "bbox": box,
                    "rows_list": [rows],
                    "last_rows": rows,
                    "reextract": False,
                }
            )

    # Merge or append pure horizontal-rule booktabs clusters (3-rule tables
    # or header bands between \toprule and \midrule missed by lines strategy).
    for rbox in _find_booktabs_rule_bboxes(page, drawings=drawings):
        matched = False
        for grp in groups:
            gbox = grp["bbox"]
            y_overlap = min(gbox[3], rbox[3]) - max(gbox[1], rbox[1])
            if (
                abs(gbox[0] - rbox[0]) < 25
                and abs(gbox[2] - rbox[2]) < 25
                and y_overlap > -15
            ):
                if rbox[1] < gbox[1] - 4 or rbox[3] > gbox[3] + 4:
                    grp["bbox"] = [
                        min(gbox[0], rbox[0]),
                        min(gbox[1], rbox[1]),
                        max(gbox[2], rbox[2]),
                        max(gbox[3], rbox[3]),
                    ]
                    grp["reextract"] = True
                matched = True
                break
        if not matched:
            near_rect = fitz_mod.Rect(
                0, rbox[1] - 48, page.rect.width, rbox[3] + 48
            )
            top_rect = fitz_mod.Rect(rbox[0], rbox[1], rbox[2], rbox[1] + 24)
            near_txt = page.get_text("text", clip=near_rect) or ""
            top_txt = (page.get_text("text", clip=top_rect) or "").strip()
            if re.search(
                r"(?:^|\n)\s*(?:Table|Tab\.|ตาราง)\s*\d+\s*[:.|]",
                near_txt,
                re.IGNORECASE,
            ) and not re.match(
                r"^(Algorithm|Figure|Fig\.)\b", top_txt, re.IGNORECASE
            ):
                groups.append(
                    {
                        "bbox": rbox,
                        "rows_list": [],
                        "last_rows": [],
                        "reextract": True,
                    }
                )

    groups.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))

    for grp in groups:
        t0_rows = grp["rows_list"][0] if grp["rows_list"] else []
        max_nl = max(
            (str(c or "").count("\n") for r in t0_rows for c in r), default=0
        )
        candidates = []
        grp_w = grp["bbox"][2] - grp["bbox"][0]
        if (
            grp_w < 200
            or (
                not grp.get("reextract")
                and len(grp["rows_list"]) == 1
                and len(t0_rows) >= 2
                and max_nl < 2
            )
        ):
            candidates.append(t0_rows)
        else:
            try:
                clip_rect = fitz_mod.Rect(grp["bbox"])
                min_v = 2 if not grp["rows_list"] else 3
                sub_tabs = page.find_tables(
                    clip=clip_rect,
                    vertical_strategy="text",
                    horizontal_strategy="text",
                    min_words_vertical=min_v,
                ).tables
                for st in sub_tabs:
                    candidates.append(st.extract())
            except Exception:  # nosec B110
                pass
            if not candidates and t0_rows:
                candidates.append(t0_rows)

        for cand in candidates:
            cleaned = clean_table_rows(cand)
            if cleaned:
                extracted.append(cleaned)

    return extracted


def rows_to_markdown(rows):
    """Format a 2D list of cleaned table rows as a GFM Markdown table."""
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]
    header = [c.replace("|", "\\|") for c in padded[0]]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for r in padded[1:]:
        escaped = [c.replace("|", "\\|") for c in r]
        md_lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(md_lines)


def _render_page_batch(batch_args):
    """Worker helper opening fitz.Document once per worker batch of pages."""
    pdf_path, dpi, items = batch_args
    import fitz

    if hasattr(fitz, "no_recommend_layout"):
        fitz.no_recommend_layout()

    rendered = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx, out_png in items:
            pix = doc[page_idx].get_pixmap(dpi=dpi)
            pix.save(out_png)
            rendered.append((page_idx, out_png))
        return rendered
    finally:
        doc.close()


_EASYOCR_READERS = {}


def ocr_image_local(image_path, langs=None):
    """Run local offline OCR on an image via EasyOCR or pytesseract."""
    if langs is None:
        langs = ["th", "en"]
    lang_key = tuple(langs)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import easyocr

            reader = _EASYOCR_READERS.get(lang_key)
            if reader is None:
                try:
                    import torch

                    use_gpu = bool(torch.cuda.is_available())
                except ImportError:
                    use_gpu = False
                try:
                    reader = easyocr.Reader(
                        list(langs), gpu=use_gpu, verbose=False
                    )
                except Exception:  # nosec B110
                    reader = easyocr.Reader(
                        ["en"], gpu=use_gpu, verbose=False
                    )
                _EASYOCR_READERS[lang_key] = reader
            lines = reader.readtext(image_path, detail=0, paragraph=True)
        text = normalize_text("\n".join(lines).strip())
        if text:
            return {"engine": "easyocr", "text": text}
    except Exception:  # nosec B110
        pass

    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        text = normalize_text(pytesseract.image_to_string(img).strip())
        if text:
            return {"engine": "pytesseract", "text": text}
    except Exception:  # nosec B110
        pass

    return {"engine": "none", "text": ""}


def _read_text_file(path):
    """Read UTF-8 text from a file if it exists, otherwise return empty."""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return normalize_text(fh.read().strip())
    return ""


def run_vlm_engine(target_path, output_dir, image_paths=None):
    """Attempt SGLang server (`infer.py`) or HF `baidu/Unlimited-OCR` VLM."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "infer.py"),
        os.path.join(os.getcwd(), "tools", "Unlimited-OCR", "infer.py"),
    ]
    infer_script = next(
        (p for p in candidates if p and os.path.isfile(p)), None
    )

    sglang_live = False
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://127.0.0.1:10000/health", timeout=1.5
        ) as resp:
            sglang_live = resp.status == 200
    except Exception:  # nosec B110
        sglang_live = False

    if infer_script and sglang_live:
        stage_dir = os.path.join(output_dir, "_vlm_stage")
        shutil.rmtree(stage_dir, ignore_errors=True)
        os.makedirs(stage_dir, exist_ok=True)
        targets = image_paths or [target_path]
        staged_map = {}
        for img_p in targets:
            if img_p.lower().endswith(".pdf"):
                continue
            base_name = os.path.basename(img_p)
            stem, ext = os.path.splitext(base_name)
            if ext.lower() in (".tif", ".tiff"):
                dst_p = os.path.join(stage_dir, f"{stem}.png")
                from PIL import Image

                Image.open(img_p).save(dst_p)
            else:
                dst_p = os.path.join(stage_dir, base_name)
                if os.path.abspath(img_p) != os.path.abspath(dst_p):
                    shutil.copy2(img_p, dst_p)
            staged_map[img_p] = os.path.join(output_dir, f"{stem}.md")

        flag = "--image_dir" if staged_map else "--pdf"
        flag_val = stage_dir if staged_map else target_path
        cmd = [
            sys.executable,
            infer_script,
            flag,
            flag_val,
            "--output_dir",
            output_dir,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.returncode == 0:
            vlm_texts = {
                img_p: _read_text_file(md_p)
                for img_p, md_p in staged_map.items()
            }
            return {
                "vlm_executed": True,
                "engine": "sglang_infer_py",
                "vlm_texts": vlm_texts,
                "stdout_tail": (proc.stdout or "")[-400:],
            }

    try:
        import torch

        if torch.cuda.is_available():
            from transformers import AutoModel, AutoTokenizer

            env_model = os.environ.get("UNLIMITED_OCR_MODEL", "").strip()
            model_candidates = (
                [env_model]
                if env_model
                else ["baidu/Unlimited-OCR-2", "baidu/Unlimited-OCR"]
            )
            tokenizer = None
            model = None
            model_name = model_candidates[0]
            for cand_model in model_candidates:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        cand_model, trust_remote_code=True
                    )
                    model = (
                        AutoModel.from_pretrained(
                            cand_model,
                            trust_remote_code=True,
                            use_safetensors=True,
                            torch_dtype=torch.bfloat16,
                        )
                        .eval()
                        .cuda()
                    )
                    model_name = cand_model
                    break
                except Exception:  # nosec B110
                    continue
            if model is None or tokenizer is None:
                raise RuntimeError("No Unlimited-OCR model available")

            vlm_texts = {}
            if image_paths and len(image_paths) > 1:
                res = model.infer_multi(
                    tokenizer,
                    prompt="<image>Multi page parsing.",
                    image_files=image_paths,
                    output_path=output_dir,
                    image_size=1024,
                    max_length=32768,
                    no_repeat_ngram_size=35,
                    ngram_window=1024,
                    save_results=True,
                )
                if isinstance(res, tuple):
                    res = res[0]
                if isinstance(res, list):
                    for img_p, txt in zip(image_paths, res):
                        vlm_texts[img_p] = normalize_text(str(txt or ""))
                elif isinstance(res, str):
                    if "<PAGE>" in res:
                        parts = [
                            p.strip()
                            for p in res.split("<PAGE>")
                            if p.strip()
                        ]
                        for img_p, txt in zip(image_paths, parts):
                            vlm_texts[img_p] = normalize_text(txt)
                    else:
                        vlm_texts[image_paths[0]] = normalize_text(res)
            else:
                img_target = image_paths[0] if image_paths else target_path
                res = model.infer(
                    tokenizer,
                    prompt="<image>document parsing.",
                    image_file=img_target,
                    output_path=output_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    max_length=32768,
                    no_repeat_ngram_size=35,
                    ngram_window=128,
                    save_results=True,
                )
                if isinstance(res, tuple):
                    res = res[0]
                if isinstance(res, str):
                    vlm_texts[img_target] = normalize_text(
                        res.replace("<PAGE>", "").strip()
                    )
            return {
                "vlm_executed": True,
                "engine": f"hf_{model_name.split('/')[-1].lower()}_cuda",
                "vlm_texts": vlm_texts,
            }
    except Exception:  # nosec B110
        pass

    return {
        "vlm_executed": False,
        "engine": "local_fallback",
        "vlm_texts": {},
        "reason": "No active CUDA/SGLang VLM server; using local OCR engine.",
    }


def extract_pdf(
    pdf_path,
    output_dir,
    mode="auto",
    dpi=300,
    pages_arg=None,
    workers=4,
    langs=None,
):
    """Single-pass C++ PyMuPDF text/table extractor with surgical OCR."""
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    text_out_path = os.path.join(output_dir, "extracted_text.txt")
    md_out_path = os.path.join(output_dir, "extracted_document.md")
    manifest_path = os.path.join(output_dir, "ocr_manifest.json")
    pages_dir = os.path.join(output_dir, "_tmp_pages")

    import fitz

    if hasattr(fitz, "no_recommend_layout"):
        fitz.no_recommend_layout()

    doc = fitz.open(pdf_path)
    total_doc_pages = len(doc)
    selected_indices = parse_page_selection(pages_arg, total_doc_pages)

    page_records = {}
    pages_needing_ocr = []
    tables_extracted = 0
    has_fitz_tables = hasattr(fitz.Page, "find_tables")

    for page_idx in selected_indices:
        page = doc[page_idx]
        page_num = page_idx + 1
        raw_text = page.get_text("text", sort=True) or ""
        qual = evaluate_page_quality(raw_text, page)

        page_md_tables = []
        page_table_count = 0

        if has_fitz_tables:
            page_tables = extract_tables_from_page(
                page, fitz, raw_text=raw_text
            )
            for rows in page_tables:
                page_table_count += 1
                tables_extracted += 1
                csv_name = f"table_p{page_num}_{page_table_count}.csv"
                csv_path = os.path.join(output_dir, csv_name)
                with open(
                    csv_path, "w", encoding="utf-8-sig", newline=""
                ) as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerows(rows)
                page_md_tables.append(rows_to_markdown(rows))

        if mode == "vlm" or (mode == "auto" and qual["needs_ocr"]):
            pages_needing_ocr.append(page_idx)

        page_records[page_idx] = {
            "page": page_num,
            "text": qual["clean_text"],
            "char_count": qual["char_count"],
            "garble_ratio": qual["garble_ratio"],
            "needs_ocr": qual["needs_ocr"],
            "table_count": page_table_count,
            "md_tables": page_md_tables,
            "engine": "pymupdf_vector",
        }

    doc.close()

    if not has_fitz_tables:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page_idx in selected_indices:
                    page_num = page_idx + 1
                    raw_tabs = pdf.pages[page_idx].extract_tables() or []
                    for raw_tab in raw_tabs:
                        rows = clean_table_rows(raw_tab)
                        if not rows:
                            continue
                        page_records[page_idx]["table_count"] += 1
                        tables_extracted += 1
                        t_idx = page_records[page_idx]["table_count"]
                        csv_name = f"table_p{page_num}_{t_idx}.csv"
                        csv_path = os.path.join(output_dir, csv_name)
                        with open(
                            csv_path, "w", encoding="utf-8-sig", newline=""
                        ) as f_csv:
                            writer = csv.writer(f_csv)
                            writer.writerows(rows)
                        page_records[page_idx]["md_tables"].append(
                            rows_to_markdown(rows)
                        )
        except ImportError:
            pass

    image_paths = []
    vlm_info = None
    if pages_needing_ocr and mode in ("auto", "vlm"):
        os.makedirs(pages_dir, exist_ok=True)
        render_items = [
            (
                p_idx,
                os.path.join(pages_dir, f"page_{p_idx + 1:03d}.png"),
            )
            for p_idx in pages_needing_ocr
        ]
        max_w = max(1, min(int(workers), len(render_items)))
        batches = [
            (pdf_path, dpi, render_items[i::max_w])
            for i in range(max_w)
            if render_items[i::max_w]
        ]
        rendered_map = {}
        if len(batches) > 1:
            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                for batch_res in pool.map(_render_page_batch, batches):
                    for p_idx, out_png in batch_res:
                        rendered_map[p_idx] = out_png
        else:
            for p_idx, out_png in _render_page_batch(batches[0]):
                rendered_map[p_idx] = out_png
        image_paths = [rendered_map[p_idx] for p_idx in pages_needing_ocr]

        vlm_info = run_vlm_engine(
            pdf_path, output_dir, image_paths=image_paths
        )
        if vlm_info.get("vlm_executed"):
            vlm_map = vlm_info.get("vlm_texts") or {}
            vlm_eng = vlm_info.get("engine", "vlm")
            for p_idx, img_path in zip(pages_needing_ocr, image_paths):
                v_txt = vlm_map.get(img_path, "")
                if v_txt:
                    page_records[p_idx]["text"] = v_txt
                    page_records[p_idx]["char_count"] = non_ws_count(v_txt)
                    page_records[p_idx]["engine"] = vlm_eng
        else:
            for p_idx, img_path in zip(pages_needing_ocr, image_paths):
                ocr_res = ocr_image_local(img_path, langs=langs)
                ocr_nws = non_ws_count(ocr_res["text"])
                if ocr_res["text"] and (
                    ocr_nws > page_records[p_idx]["char_count"]
                    or page_records[p_idx]["garble_ratio"] > 0.05
                ):
                    page_records[p_idx]["text"] = ocr_res["text"]
                    page_records[p_idx]["char_count"] = ocr_nws
                    page_records[p_idx]["engine"] = ocr_res["engine"]

    txt_chunks = []
    md_chunks = []
    manifest_pages = []
    total_chars = 0

    for page_idx in selected_indices:
        rec = page_records[page_idx]
        p_num = rec["page"]
        total_chars += rec["char_count"]
        txt_chunks.append(f"--- PAGE {p_num} ---\n{rec['text']}\n")

        md_sec = [f"## Page {p_num}", "", rec["text"]]
        if rec["md_tables"]:
            md_sec.append(f"\n### Extracted Tables (Page {p_num})")
            for t_i, md_tab in enumerate(rec["md_tables"], start=1):
                md_sec.append(f"\n#### Table {p_num}.{t_i}\n\n{md_tab}")
        md_chunks.append("\n".join(md_sec))

        manifest_pages.append(
            {
                "page": p_num,
                "char_count": rec["char_count"],
                "table_count": rec["table_count"],
                "garble_ratio": rec["garble_ratio"],
                "needs_ocr": rec["needs_ocr"],
                "engine": rec["engine"],
            }
        )

    with open(text_out_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(txt_chunks))

    with open(md_out_path, "w", encoding="utf-8") as f_md:
        f_md.write("\n\n---\n\n".join(md_chunks) + "\n")

    elapsed_ms = round((time.time() - t_start) * 1000, 1)
    manifest_data = {
        "input": pdf_path,
        "total_pdf_pages": total_doc_pages,
        "pages_processed": len(selected_indices),
        "total_chars": total_chars,
        "tables_extracted": tables_extracted,
        "low_text_or_ocr_pages": [p + 1 for p in pages_needing_ocr],
        "elapsed_ms": elapsed_ms,
        "pages": manifest_pages,
    }
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest_data, f_man, indent=2, ensure_ascii=False)

    res = {
        "pages_processed": len(selected_indices),
        "total_chars": total_chars,
        "tables_extracted": tables_extracted,
        "low_text_pages": [p + 1 for p in pages_needing_ocr],
        "elapsed_ms": elapsed_ms,
        "text_path": text_out_path,
        "markdown_path": md_out_path,
        "manifest_path": manifest_path,
        "image_paths": image_paths,
    }
    if vlm_info:
        res["vlm"] = vlm_info
    return res


def extract_standalone_image(image_path, output_dir, langs=None):
    """Extract text from a standalone image (.png/.jpg/.webp/.tif)."""
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    text_out_path = os.path.join(output_dir, "extracted_text.txt")
    md_out_path = os.path.join(output_dir, "extracted_document.md")
    manifest_path = os.path.join(output_dir, "ocr_manifest.json")

    vlm_info = run_vlm_engine(
        image_path, output_dir, image_paths=[image_path]
    )
    extracted_text = ""
    engine_used = vlm_info.get("engine", "none")

    if vlm_info.get("vlm_executed"):
        vlm_map = vlm_info.get("vlm_texts") or {}
        extracted_text = vlm_map.get(image_path, "")
    if not extracted_text:
        ocr_res = ocr_image_local(image_path, langs=langs)
        extracted_text = ocr_res["text"]
        engine_used = ocr_res["engine"]

    with open(text_out_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(extracted_text + "\n")
    with open(md_out_path, "w", encoding="utf-8") as f_md:
        f_md.write(f"## Image OCR ({os.path.basename(image_path)})\n\n")
        f_md.write(extracted_text + "\n")

    elapsed_ms = round((time.time() - t_start) * 1000, 1)
    nws_chars = non_ws_count(extracted_text)
    manifest_data = {
        "input": image_path,
        "pages_processed": 1,
        "total_chars": nws_chars,
        "tables_extracted": 0,
        "engine_used": engine_used,
        "elapsed_ms": elapsed_ms,
        "pages": [
            {
                "page": 1,
                "char_count": nws_chars,
                "table_count": 0,
                "garble_ratio": 0.0,
                "needs_ocr": True,
                "engine": engine_used,
            }
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest_data, f_man, indent=2, ensure_ascii=False)

    return {
        "status": "success" if extracted_text else "empty_or_failed",
        "pages_processed": 1,
        "total_chars": nws_chars,
        "tables_extracted": 0,
        "engine_used": engine_used,
        "elapsed_ms": elapsed_ms,
        "text_path": text_out_path,
        "markdown_path": md_out_path,
        "manifest_path": manifest_path,
        "vlm": vlm_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR & High-Speed PDF/Table Extractor"
    )
    parser.add_argument(
        "input_path", nargs="?", help="Path to target PDF or image file"
    )
    parser.add_argument(
        "--input", "-i", dest="input_flag", help="Path to target PDF or image"
    )
    parser.add_argument(
        "--output", "-o", default="./ocr_output", help="Output directory"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pdf", "vlm"],
        default="auto",
        help="Extraction mode: auto (default hybrid), pdf (vector), or vlm",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Render DPI for scanned pages"
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Optional 1-based page range, e.g. '1-5,8,12-15'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel worker threads for page rasterization",
    )
    parser.add_argument(
        "--langs",
        default="th,en",
        help="Comma-separated language codes for local OCR (default: th,en)",
    )
    args = parser.parse_args()

    raw_input = args.input_flag or args.input_path
    if not raw_input:
        parser.error("Provide an input file path positionally or via --input")

    input_path = os.path.abspath(raw_input)
    output_dir = os.path.abspath(args.output)
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    if not os.path.exists(input_path):
        err = {"status": "error", "message": f"Input not found: {input_path}"}
        print(json.dumps(err))
        sys.exit(1)

    summary = {
        "status": "success",
        "input": input_path,
        "output_dir": output_dir,
        "mode": args.mode,
    }

    if input_path.lower().endswith(".pdf"):
        pdf_res = extract_pdf(
            input_path,
            output_dir,
            mode=args.mode,
            dpi=args.dpi,
            pages_arg=args.pages,
            workers=args.workers,
            langs=langs,
        )
        summary.update(pdf_res)
    else:
        img_res = extract_standalone_image(
            input_path, output_dir, langs=langs
        )
        summary.update(img_res)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
