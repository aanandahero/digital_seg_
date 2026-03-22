"""
app.py -- Digital Pratilipi: Nepali Handwritten OCR
Kantipur Engineering College -- CT 755 Major Project  v7

Architecture (proven by profiling real images):
  LINE DETECTION  : Connected Component Y-centroid clustering
  WORD DETECTION  : VPP (Vertical Projection Profile) per line band

Why CC for lines (not HPP):
  On close-together handwritten lines, binarized ink is often
  CONTINUOUS between lines (descenders touching ascenders + noise).
  HPP never reaches zero → zero-crossing and valley detection both fail.
  CC blob Y-centroids have a clear gap (>25px) between lines even when
  ink is physically touching. Immune to vertical ink continuity.

Why VPP for words (not CC):
  Words in Devanagari are connected by the shirorekha horizontally,
  making them one CC blob. VPP after shirorekha suppression cleanly
  finds inter-word gaps (>15px) while intra-word gaps stay <5px.
"""

import os, base64, logging, re, unicodedata
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

TROCR_MODEL = os.getenv(
    "TROCR_MODEL",
    r"C:\Users\Aananda Sagar Thapa\OneDrive\Desktop\ANANDA HERO\model"
)


# ============================================================
# PREPROCESSING
# ============================================================

def _remove_background(gray: np.ndarray) -> np.ndarray:
    """Safe rolling-ball BG removal for noisy/shadowed phone photos."""
    bg      = cv2.GaussianBlur(gray, (0, 0), sigmaX=50)
    bg_safe = np.minimum(bg, gray)
    out     = cv2.addWeighted(gray, 1.0, bg_safe.astype(np.uint8), -1.0, 128)
    return np.clip(out, 20, 235).astype(np.uint8)


def _sauvola_binarize(gray: np.ndarray,
                      window: int = 25, k: float = 0.15) -> np.ndarray:
    """Sauvola local binarisation. INK = WHITE (255)."""
    f   = gray.astype(np.float32)
    m   = cv2.boxFilter(f, -1, (window, window))
    m2  = cv2.boxFilter(f * f, -1, (window, window))
    std = np.sqrt(np.maximum(m2 - m * m, 0))
    thr = m * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(f < thr, 255, 0).astype(np.uint8)


def _remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """
    Detect and remove horizontal ruled lines (notebook paper).

    A ruled line row satisfies BOTH:
      1. Ink density 15–30% of image width
         < 15%: sparse character fragment — not a ruled line
         > 30%: dense text / shirorekha — never a ruled line
      2. Isolation: this row's ink > 2.5× the average of ±5-row neighbors
         Ruled lines sit in whitespace; text rows are in dense clusters.

    After detection: erase ±2px, then repair broken character strokes
    via vertical dilation masked back to original ink footprint.
    """
    img_h, img_w = binary.shape
    hpp = np.sum(binary // 255, axis=1).astype(np.float32) / img_w

    ruled = []
    for r in range(img_h):
        if hpp[r] < 0.15 or hpp[r] > 0.30:
            continue
        window  = [hpp[max(0, r+dr)] for dr in range(-5, 6) if dr != 0]
        avg_nbr = sum(window) / len(window)
        if avg_nbr > 0 and hpp[r] > avg_nbr * 2.5:
            ruled.append(r)

    if not ruled:
        return binary

    expanded = set()
    for r in ruled:
        for dr in range(-2, 3):
            if 0 <= r + dr < img_h:
                expanded.add(r + dr)

    cleaned = binary.copy()
    for r in expanded:
        cleaned[r, :] = 0

    vk       = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    bridged  = cv2.dilate(cleaned, vk, iterations=1)
    repaired = cv2.bitwise_and(bridged, binary)
    for r in expanded:
        repaired[r, :] = 0

    logger.info(f"Ruled line removal: {len(ruled)} rows "
                f"({ruled[:4]}{'…' if len(ruled)>4 else ''})")
    return repaired


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive binarisation + ruled line removal + stroke reconnection.

    CLEAN image (mean > 190 — good scan / well-lit photo):
      Otsu global threshold.

    NOISY image (mean ≤ 190 — phone photo with shadows):
      Rolling-ball BG removal → CLAHE → Sauvola local threshold.

    Post-processing (both paths):
      1. Erase top/bottom border artifacts (3% of height)
      2. Remove ruled notebook lines (rows with > 50% width ink)
      3. Remove isolated large blobs (> 8% of image = camera artifacts)
      4. MORPH_CLOSE (3×2) — reconnects broken halant joins and matras
      5. MORPH_OPEN  (2×2) — removes isolated noise dots
    """
    mean_brightness = float(np.mean(gray))
    logger.info(f"Image brightness: {mean_brightness:.0f} "
                f"→ {'clean' if mean_brightness > 190 else 'noisy'}")

    if mean_brightness > 190:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        method = "otsu"
    else:
        bg_rem = _remove_background(gray)
        eq     = cv2.createCLAHE(3.0, (8, 8)).apply(bg_rem)
        binary = _sauvola_binarize(eq)
        method = "sauvola"
        # Sanity check: Sauvola sometimes under-thresholds dark images,
        # leaving < 1% ink. Fall back to Otsu if that happens.
        ink_pct = np.sum(binary > 0) / (gray.shape[0] * gray.shape[1])
        if ink_pct < 0.01:
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            method = "otsu-fallback"
            logger.info("Sauvola gave < 1% ink — falling back to Otsu")

    h, w     = binary.shape
    img_area = h * w

    # 1. Erase border strips
    binary[:max(1, h // 12), :]          = 0   # top ~8%
    binary[min(h-1, h - h // 12):, :]   = 0   # bottom ~8%

    # 2. Remove ruled notebook lines BEFORE blob filtering
    binary = _remove_ruled_lines(binary)

    # 3. Remove very large blobs (> 8% = camera lens flare, vignette)
    #    NOTE: raised from 3% → 8% so word-sized blobs (1-4%) survive
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > img_area * 0.08:
            binary[labels == i] = 0

    # 4. Reconnect broken strokes (halant ् , matras ि ी ु ू)
    nk     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, nk)

    # 5. Remove isolated noise dots
    nk_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, nk_open)

    logger.info(f"Binarize ({method}): ink={np.sum(binary>0)/img_area*100:.1f}%")
    return binary


def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    1. Upscale to >= 960px wide  (better resolution for CC + TrOCR)
    2. Bilateral denoise          (preserves ink edges)
    3. Sharpen                    (enhances stroke definition)
    4. Adaptive binarisation      (Otsu for clean, Sauvola for noisy)
    5. Stroke reconnection        (morphological close — fixes broken joins)
    6. Hough skew correction

    Returns (colour_bgr, binary_ink_white).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")),
                            cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    if w < 960:
        s       = 960 / w
        img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)),
                             interpolation=cv2.INTER_CUBIC)

    den    = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    shr    = cv2.filter2D(den, -1,
                          np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32))
    gray   = cv2.cvtColor(shr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    # Skew correction via Hough lines
    edges = cv2.Canny(cv2.bitwise_not(binary), 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is not None:
        angles = [(l[0][1] - np.pi/2)*180/np.pi for l in lines]
        skew   = float(np.median(angles))
        if 1.0 < abs(skew) < 15:
            h2, w2 = img_bgr.shape[:2]
            M       = cv2.getRotationMatrix2D((w2/2, h2/2), skew, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, M, (w2, h2),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            binary  = cv2.warpAffine(binary, M, (w2, h2),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    return img_bgr, binary


# ============================================================
# LINE DETECTION — Connected Component Y-centroid clustering
# ============================================================

def _find_line_bands(binary: np.ndarray,
                     min_line_height: int = 15) -> list[tuple[int,int]]:
    """
    Detect text line bands by CC Y-centroid clustering.

    LINE_GAP = 20px (reduced from 30):
      Handwritten Devanagari lines photographed on a phone are often only
      20-30px apart in blob-centroid space after upscaling to 960px.
      Using 30px caused adjacent lines to be merged into one band.

    Post-cluster split:
      Any cluster taller than 1.5× the median cluster height is likely
      two merged lines. Re-cluster those blobs with a tighter gap of 12px
      to try to split them apart.

    Blob filter: bh < 35% of image height (was 25%).
      Large Devanagari word blobs (shirorekha + body) can reach 30% of
      the image height in close-up phone photos.
    """
    img_h, img_w = binary.shape
    img_area     = img_h * img_w

    n, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    max_blob = img_area * 0.05
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        top  = stats[i, cv2.CC_STAT_TOP]
        left = stats[i, cv2.CC_STAT_LEFT]
        cy   = float(centroids[i][1])
        if (30 < area < max_blob
                and bh > 5
                and bh < img_h * 0.35   # raised from 0.25 → 0.35
                and bw < img_w * 0.80
                and cy > img_h * 0.05):
            blobs.append({
                'top': top, 'bot': top + bh,
                'left': left, 'right': left + bw,
                'cy': cy
            })

    if not blobs:
        logger.warning("CC: no character blobs found")
        return []

    blobs.sort(key=lambda b: b['cy'])

    # Primary clustering with tighter gap
    LINE_GAP = 20   # reduced from 30
    clusters = []
    cur      = [blobs[0]]
    for i in range(1, len(blobs)):
        if blobs[i]['cy'] - blobs[i-1]['cy'] > LINE_GAP:
            clusters.append(cur)
            cur = [blobs[i]]
        else:
            cur.append(blobs[i])
    if cur:
        clusters.append(cur)

    # Post-cluster split: re-split clusters taller than 1.5× median
    if len(clusters) >= 2:
        heights = sorted([max(b['bot'] for b in cl) - min(b['top'] for b in cl)
                          for cl in clusters])
        median_h = heights[len(heights) // 2]
        SPLIT_THR = median_h * 1.5
    else:
        SPLIT_THR = float('inf')

    final_clusters = []
    for cl in clusters:
        cl_h = max(b['bot'] for b in cl) - min(b['top'] for b in cl)
        if cl_h > SPLIT_THR and len(cl) > 1:
            # Re-cluster with tighter gap
            cl_sorted = sorted(cl, key=lambda b: b['cy'])
            sub = [cl_sorted[0]]
            for i in range(1, len(cl_sorted)):
                if cl_sorted[i]['cy'] - cl_sorted[i-1]['cy'] > 12:
                    final_clusters.append(sub)
                    sub = [cl_sorted[i]]
                else:
                    sub.append(cl_sorted[i])
            final_clusters.append(sub)
        else:
            final_clusters.append(cl)

    bands = []
    for cluster in final_clusters:
        y1  = min(b['top'] for b in cluster)
        y2  = max(b['bot'] for b in cluster)
        if y2 - y1 < min_line_height:
            continue
        pad = max(5, (y2 - y1) // 8)
        bands.append((max(0, y1 - pad), min(img_h, y2 + pad)))

    bands.sort(key=lambda b: b[0])
    logger.info(f"CC line detection: {len(bands)} bands from "
                f"{len(blobs)} blobs / {len(final_clusters)} clusters")
    return bands


# ============================================================
# WORD DETECTION — VPP per line band
# ============================================================

def _find_words_in_band(binary: np.ndarray,
                        y1: int, y2: int,
                        img_w: int) -> list[dict]:
    """
    Find word bounding boxes within a line band using VPP.

    Fixes vs old version:
      - Shirorekha suppression: 35% (was 20%) — the headline bar often
        extends well into the top third of a line strip.
      - Adaptive word gap: uses the median gap width in this strip.
        Gaps narrower than median are intra-word (merge them).
        Gaps wider than median are inter-word (keep as boundaries).
        This self-calibrates across different writing sizes and styles.

    Steps:
      1. Extract and pad strip vertically
      2. Blank top 35% (kills shirorekha)
      3. VPP → column ink sums
      4. Find all zero-runs (gaps) and their widths
      5. Adaptive merge: merge gaps < median_gap_width
      6. Filter noise spans
    """
    img_h  = binary.shape[0]
    line_h = y2 - y1

    pad  = max(5, line_h // 8)
    py1  = max(0, y1 - pad)
    py2  = min(img_h, y2 + pad)
    strip = binary[py1:py2, :].copy()
    sh    = strip.shape[0]

    # Suppress shirorekha — top 30% of the strip
    # Capped at 40px so short line bands don't lose all their ink
    shiro_px = min(int(sh * 0.30), 40)
    strip[:max(1, shiro_px), :] = 0

    # VPP
    vpp = np.sum(strip, axis=0) // 255

    # Collect ink spans and gap widths
    raw_spans  = []
    gap_widths = []
    in_s = False; cs = 0; prev_end = 0
    for c in range(img_w):
        if vpp[c] > 0 and not in_s:
            if raw_spans:
                gap_widths.append(c - prev_end)
            in_s, cs = True, c
        elif vpp[c] == 0 and in_s:
            in_s = False
            raw_spans.append([cs, c])
            prev_end = c
    if in_s:
        raw_spans.append([cs, img_w])

    if not raw_spans:
        return []

    # Adaptive merge threshold using gap bimodality:
    # Gap widths have two groups: small (intra-word broken strokes)
    # and large (inter-word spaces). Find the split point.
    #
    # Method: find the largest gap in gap_widths that is LESS THAN
    # half the maximum gap. Everything below this is intra-word.
    # This robustly separates the two populations even when a few
    # intra-word gaps are larger than expected (e.g. 15-20px).
    if gap_widths:
        sg       = sorted(gap_widths)
        max_gap  = sg[-1]
        # Split: intra-word gaps are < max_gap / 3
        # (inter-word gaps are always much larger than intra-word)
        split    = max_gap / 3.0
        intra    = [g for g in sg if g < split]
        merge_thr = max(5, int(max(intra)) + 1) if intra else max(5, int(split))
        merge_thr = min(merge_thr, img_w // 8)  # hard cap
    else:
        merge_thr = 5

    merged = []
    for s in raw_spans:
        if merged and s[0] - merged[-1][1] <= merge_thr:
            merged[-1][1] = s[1]
        else:
            merged.append(s[:])

    bh    = py2 - py1
    boxes = []
    for x1, x2 in merged:
        bw = x2 - x1
        if bw < 12:             continue  # noise fragment
        if bw > img_w * 0.75:  continue  # full-width artifact
        if x1 < 3:             continue  # left-border artifact
        if bw * bh < 200:      continue  # too small
        boxes.append({"x": x1, "y": py1, "w": bw, "h": bh})
    return boxes


# ============================================================
# MAIN DETECTION
# ============================================================

_easyocr_reader = None

def get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(
                ['ne','en'], gpu=False, recognizer=False, verbose=False)
            logger.info("EasyOCR loaded.")
        except Exception as e:
            logger.error(f"EasyOCR: {e}")
    return _easyocr_reader


def detect_words(img_bgr: np.ndarray,
                 binary: np.ndarray = None,
                 min_area: int = 300) -> tuple[list, list]:
    """
    Full detection pipeline:
      1. CC Y-centroid clustering → line bands
      2. VPP per band → word boxes
      3. EasyOCR emergency fallback if nothing found

    Returns (flat_boxes, line_groups).
    """
    if binary is None:
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        eq     = cv2.createCLAHE(3.0,(8,8)).apply(_remove_background(gray))
        binary = _sauvola_binarize(eq)
        binary = _remove_large_blobs(binary)
        nk     = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,nk)
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,nk)

    img_h, img_w = binary.shape

    # Step 1: CC → line bands
    line_bands = _find_line_bands(binary)

    # Step 2: VPP → word boxes per band
    all_boxes, line_groups = [], []
    for (y1, y2) in line_bands:
        word_boxes = _find_words_in_band(binary, y1, y2, img_w)
        if word_boxes:
            word_boxes = sorted(word_boxes, key=lambda b: b["x"])
            line_groups.append(word_boxes)
            all_boxes.extend(word_boxes)

    if all_boxes:
        logger.info(f"detect_words: {len(all_boxes)} words / "
                    f"{len(line_groups)} lines  [CC+VPP]")
        return all_boxes, line_groups

    # Fallback
    logger.warning("CC+VPP found nothing — EasyOCR fallback")
    return _easyocr_fallback(img_bgr, binary, min_area)


def _easyocr_fallback(img_bgr, binary, min_area=300):
    reader = get_easyocr()
    if reader is None: return [], []
    img_h, img_w = img_bgr.shape[:2]
    try:
        clean   = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
        results = reader.detect(clean)
        raw     = results[0][0] if results and results[0] else []
    except Exception as e:
        logger.warning(f"EasyOCR: {e}"); return [], []
    boxes = []
    for b in raw:
        x1=max(0,int(b[0])); x2=min(img_w,int(b[1]))
        y1=max(0,int(b[2])); y2=min(img_h,int(b[3]))
        w,h=x2-x1,y2-y1
        if y2>img_h*0.85 or y1<img_h*0.05 or w*h<min_area: continue
        boxes.append({"x":x1,"y":y1,"w":w,"h":h})
    if not boxes: return [], []
    boxes.sort(key=lambda b: b["y"])
    lines,cur,cy=[],[],None
    for b in boxes:
        my=b["y"]+b["h"]//2
        if cy is None or abs(my-cy)<=30: cur.append(b); cy=my
        else: lines.append(sorted(cur,key=lambda b:b["x"])); cur=[b]; cy=my
    if cur: lines.append(sorted(cur,key=lambda b:b["x"]))
    return [b for ln in lines for b in ln], lines


# ============================================================
# WORD CROP
# ============================================================

def crop_word(img_bgr: np.ndarray, box: dict, padding: int = 14) -> Image.Image:
    """
    Crop a word region from the original colour image.

    Padding = 14px (was 8px) on all sides:
      Devanagari characters have ascending strokes (shirorekha, upper vowels)
      and descending strokes (lower vowel marks) that extend well beyond
      the VPP bounding box. Tight crops cause TrOCR to see partial characters.
      14px ensures all strokes are captured without including adjacent words
      (inter-word gaps are typically 20-40px so 14px padding is safe).

    Returns RGB PIL Image (TrOCR expects RGB).
    """
    h, w  = img_bgr.shape[:2]
    x1    = max(0, box["x"] - padding)
    y1    = max(0, box["y"] - padding)
    x2    = min(w, box["x"] + box["w"] + padding)
    y2    = min(h, box["y"] + box["h"] + padding)
    crop  = img_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return Image.new("RGB", (128, 64), color=(255, 255, 255))

    # Resize to fixed height 64px preserving aspect ratio
    tgt_h = 64
    scale = tgt_h / crop.shape[0]
    tgt_w = max(32, int(crop.shape[1] * scale))
    crop  = cv2.resize(crop, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement (normalises across different lighting)
    lab       = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    l         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    crop      = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ============================================================
# TROCR
# ============================================================

_processor = None
_model     = None

def get_trocr():
    global _processor, _model
    if _processor is None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            logger.info(f"Loading TrOCR from {TROCR_MODEL}...")
            _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL,local_files_only=True)
            _model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL,local_files_only=True)
            _model.eval()
            logger.info("TrOCR loaded.")
        except Exception as e:
            logger.error(f"TrOCR: {e}")
    return _processor, _model


def trocr_predict(word_img: Image.Image) -> str:
    import torch
    proc, model = get_trocr()
    if not proc: return ""
    try:
        pv = proc(images=word_img.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(
                pv,
                max_new_tokens=48,        # longer for conjunct consonants
                num_beams=5,              # wider beam search
                early_stopping=True,
                repetition_penalty=1.2,   # penalise repeated tokens
                length_penalty=1.0,
            )
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        logger.warning(f"TrOCR: {e}"); return ""


# ============================================================
# POSTPROCESSING
# ============================================================

NEPALI_DICT = {
    "म","मेरो","हाम्रो","तिमी","तिम्रो","उ","उसको","यो","त्यो",
    "हामी","तपाई","आफ्नो","छ","छन्","हो","हुन्","गर्छ","गयो",
    "आयो","भयो","गर्नु","आउनु","जानु","खानु","पिउनु","पढ्नु",
    "लेख्नु","हेर्नु","बोल्नु","सुन्नु","भन्नु","हुन्छ","गर्छु",
    "जान्छु","आउँछु","पर्छ","नाम","घर","देश","नेपाल","मान्छे",
    "आमा","बाबा","दाजु","भाइ","दिदी","बहिनी","साथी","स्कुल",
    "किताब","कलम","पानी","खाना","दूध","चिया","काम","पैसा",
    "समय","दिन","रात","बिहान","साँझ","सहर","गाउँ","जीवन",
    "संसार","मन","राम्रो","नराम्रो","ठूलो","सानो","धेरै",
    "थोरै","नया","खुसी","दुखी","सुन्दर","रमाइलो",
    "मलाई","हामीलाई","तिमीलाई","उसलाई","पढ्न","लेख्न",
    "खान","जान","आउन","त्यस्तै","यस्तै","पर्छ","लाग्छ",
    "मा","को","का","की","लाई","बाट","देखि","सम्म","साथ",
    "पनि","नै","र","तर","वा","भने","भनेर","छु","छौ","छन",
    "थियो","थिए","थिएन","छैन","गरे","गरेको","भएको",
    "हुने","गर्ने","आउने","जाने","खाने","भन्ने","हेर्ने",
    "।","?","!",",",".",
}

def levenshtein(a,b):
    if len(a)<len(b): return levenshtein(b,a)
    if not b: return len(a)
    prev=list(range(len(b)+1))
    for ca in a:
        curr=[prev[0]+1]
        for j,cb in enumerate(b):
            curr.append(min(prev[j+1]+1,curr[j]+1,prev[j]+(0 if ca==cb else 1)))
        prev=curr
    return prev[-1]

def spell_correct(word, max_dist=1):
    if not word or len(word)<3: return word,False
    if word in NEPALI_DICT: return word,False
    if re.match(r'^[०-९0-9।,.!?\s]+$',word): return word,False
    cands=[w for w in NEPALI_DICT if abs(len(w)-len(word))<=max_dist]
    best_w,best_d=None,max_dist+1
    for c in cands:
        d=levenshtein(word,c)
        if d<best_d: best_d,best_w=d,c
    return (best_w,True) if best_w and best_d<=max_dist else (word,False)

def postprocess(lines):
    corr,res=[],[]
    for line in lines:
        out=[]
        for word in line:
            word=unicodedata.normalize("NFC",word)
            c,ch=spell_correct(word)
            if ch: corr.append({"original":word,"corrected":c})
            out.append(c)
        res.append(" ".join(out))
    return "\n".join(res),corr


# ============================================================
# TTS
# ============================================================

def speak_nepali(text,path="output_audio.mp3"):
    if not text.strip(): return None
    try:
        from gtts import gTTS
        gTTS(text=text,lang="ne",slow=False).save(path); return path
    except Exception as e:
        logger.error(f"TTS: {e}"); return None


# ============================================================
# PIPELINE
# ============================================================

def _render_detection_viz(img_bgr: np.ndarray,
                          line_groups: list) -> str:
    """
    Draw coloured bounding boxes on the image and return as base64 JPEG.
    Called inside run_pipeline so the frontend gets viz + text in one request.
    """
    vis    = img_bgr.copy()
    COLORS = [(60,200,100),(60,160,220),(220,80,60),
              (220,160,40),(180,60,220),(40,200,200)]
    for li, line in enumerate(line_groups):
        col = COLORS[li % len(COLORS)]
        for wi, box in enumerate(line):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
            cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def run_pipeline(pil_image: Image.Image, speak=False) -> dict:
    logger.info("=== Pipeline start ===")
    img_bgr, binary = preprocess(pil_image)
    boxes, line_groups = detect_words(img_bgr, binary=binary)

    # Always render detection viz — even if no boxes found
    viz_b64    = _render_detection_viz(img_bgr, line_groups)
    line_count = len(line_groups)
    word_count = len(boxes)

    if not boxes:
        return {"raw_text":"","final_text":"","corrections":[],
                "word_count":0,"regions":0,"audio_url":None,
                "lines":[],"method":"none",
                "viz_b64":viz_b64,"line_count":0}

    proc, _ = get_trocr()
    rec_lines = []
    for line in line_groups:
        words = []
        for box in line:
            w = trocr_predict(crop_word(img_bgr, box)) if proc else ""
            if w: words.append(w)
        if words: rec_lines.append(words)

    raw   = "\n".join(" ".join(wl) for wl in rec_lines)
    final, corr = postprocess(rec_lines)

    audio = None
    if speak and final:
        p = speak_nepali(final)
        if p: audio = "/audio"

    method = "cc+vpp+trocr" if proc else "cc+vpp+boxes"
    logger.info(f"=== Done: {word_count} words / {line_count} lines ===")
    return {"raw_text":raw,"final_text":final,"corrections":corr,
            "word_count":len(final.split()),"regions":word_count,
            "lines":[" ".join(wl) for wl in rec_lines],
            "audio_url":audio,"method":method,
            "viz_b64":viz_b64,"line_count":line_count}


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def serve_frontend():
    from flask import send_from_directory
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    p,_=get_trocr()
    return jsonify({"status":"ok","trocr":p is not None,
                    "easyocr":_easyocr_reader is not None})

@app.route("/ocr",methods=["POST"])
@app.route("/ocr_full",methods=["POST"])
def ocr_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image"}),400
        img   = Image.open(request.files["image"].stream).convert("RGB")
        speak = request.form.get("speak","0")=="1"
        return jsonify(run_pipeline(img,speak=speak))
    except Exception as e:
        logger.exception("Pipeline error"); return jsonify({"error":str(e)}),500

@app.route("/tts",methods=["POST"])
def tts_endpoint():
    try:
        text=request.get_json(force=True).get("text","")
        if not text: return jsonify({"error":"No text"}),400
        path=speak_nepali(text)
        if not path: return jsonify({"error":"TTS failed"}),500
        return send_file(path,mimetype="audio/mpeg",download_name="speech.mp3")
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/audio")
def get_audio():
    if not os.path.exists("output_audio.mp3"):
        return jsonify({"error":"No audio"}),404
    return send_file("output_audio.mp3",mimetype="audio/mpeg",
                     download_name="speech.mp3")

@app.route("/detect_viz",methods=["POST"])
def detect_viz():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image"}),400
        pil = Image.open(request.files["image"].stream).convert("RGB")
        img_bgr,binary = preprocess(pil)
        boxes,line_groups = detect_words(img_bgr,binary=binary)

        vis = img_bgr.copy()
        COLORS=[(60,200,100),(60,160,220),(220,80,60),
                (220,160,40),(180,60,220),(40,200,200)]
        for li,line in enumerate(line_groups):
            col=COLORS[li%len(COLORS)]
            for wi,box in enumerate(line):
                x,y,w,h=box["x"],box["y"],box["w"],box["h"]
                cv2.rectangle(vis,(x,y),(x+w,y+h),col,2)
                cv2.putText(vis,f"L{li+1}W{wi+1}",(x,max(y-4,14)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1,cv2.LINE_AA)

        _,buf=cv2.imencode(".jpg",vis,[cv2.IMWRITE_JPEG_QUALITY,90])
        return jsonify({"image_b64":base64.b64encode(buf.tobytes()).decode(),
                        "word_count":len(boxes),"line_count":len(line_groups),
                        "boxes":[{"x":b["x"],"y":b["y"],"w":b["w"],"h":b["h"]}
                                 for b in boxes]})
    except Exception as e:
        logger.exception("detect_viz error"); return jsonify({"error":str(e)}),500

if __name__ == "__main__":
    print("\n"+"="*60)
    print("  Digital Pratilipi OCR  v7  [CC+VPP]")
    print("="*60)
    print(f"  TROCR_MODEL : {TROCR_MODEL}")
    print(f"  URL         : http://localhost:5000")
    print("="*60+"\n")
    get_trocr(); get_easyocr()
    app.run(host="0.0.0.0",port=5000,debug=False)