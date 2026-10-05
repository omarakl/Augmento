"""Augmento v2.0 - image dataset augmentation, renaming and resizing tool.

Requires: pip install pillow opencv-python numpy
"""
import csv
import os
import queue
import random
import re
import shutil
import threading
import uuid
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".jpe", ".bmp", ".gif", ".tif", ".tiff",
              ".webp", ".ppm", ".pgm", ".tga")
KEEP_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXPLAIN_URL = "https://augmento.pythonanywhere.com/explanation"
BORDER_MODES = {
    "Reflect": cv2.BORDER_REFLECT,
    "Constant (black)": cv2.BORDER_CONSTANT,
    "Replicate": cv2.BORDER_REPLICATE,
}


# ---------------------------------------------------------------------------
# Geometric augmentations: return a 3x3 matrix (applied with one warp, so
# bounding boxes can be transformed with the exact same matrix)
# ---------------------------------------------------------------------------
def _about_center(h, w, A):
    c = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], float)
    ci = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)
    return c @ A @ ci


def g_flip(h, w, rng, k):
    return np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], float)


def g_rotate(h, w, rng, k):
    angle = rng.uniform(-1, 1) * min(45 * k, 180)
    return np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), [0, 0, 1]])


def g_translate(h, w, rng, k):
    f = min(0.1 * k, 0.5)
    A = np.eye(3)
    A[0, 2] = rng.uniform(-f, f) * w
    A[1, 2] = rng.uniform(-f, f) * h
    return A


def g_zoom(h, w, rng, k):
    s = 1 + rng.uniform(0.1, 0.5) * k
    return _about_center(h, w, np.diag([s, s, 1.0]))


def g_shear(h, w, rng, k):
    f = min(0.3 * k, 0.8)
    A = np.array([[1, rng.uniform(-f, f), 0], [rng.uniform(-f, f), 1, 0], [0, 0, 1]], float)
    return _about_center(h, w, A)


def g_crop(h, w, rng, k):
    """Random crop, resized back to the original size (keeps dataset uniform)."""
    s = rng.uniform(max(0.2, 1 - 0.5 * k), 1.0)
    cw, ch = w * s, h * s
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    return np.array([[1 / s, 0, -x0 / s], [0, 1 / s, -y0 / s], [0, 0, 1]], float)


def g_perspective(h, w, rng, k):
    d = min(0.15 * k, 0.3)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-d, d, (4, 2)) * np.array([w, h])
    return cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))


def g_affine(h, w, rng, k):
    for _ in range(10):  # avoid near-singular matrices
        L = np.eye(2) + rng.uniform(-0.15, 0.15, (2, 2)) * k
        if np.linalg.det(L) > 0.4:
            break
    else:
        L = np.eye(2)
    A = np.eye(3)
    A[:2, :2] = L
    f = min(0.1 * k, 0.4)
    A[0, 2], A[1, 2] = rng.uniform(-f, f) * w, rng.uniform(-f, f) * h
    return _about_center(h, w, A)


# ---------------------------------------------------------------------------
# Photometric / noise augmentations: f(img, rng, k) -> img (uint8 RGB)
# ---------------------------------------------------------------------------
def _enhance(img, cls, factor):
    return np.array(cls(Image.fromarray(img)).enhance(factor))


def _pil_filter(img, filt):
    return np.array(Image.fromarray(img).filter(filt))


def p_brightness(img, rng, k):
    return _enhance(img, ImageEnhance.Brightness, max(0.1, 1 + rng.uniform(-0.5, 0.5) * k))


def p_contrast(img, rng, k):
    return _enhance(img, ImageEnhance.Contrast, max(0.1, 1 + rng.uniform(-0.5, 0.5) * k))


def p_saturation(img, rng, k):
    return _enhance(img, ImageEnhance.Color, max(0.0, 1 + rng.uniform(-0.5, 0.5) * k))


def p_hue(img, rng, k):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # OpenCV hue range is 0..179
    shift = int(rng.uniform(-0.15, 0.15) * k * 180)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def p_blur(img, rng, k):
    return cv2.GaussianBlur(img, (0, 0), max(0.1, rng.uniform(0.5, 2.0) * k))


def p_salt_pepper(img, rng, k):
    out = img.copy()
    amount = min(rng.uniform(0.005, 0.03) * k, 0.5)
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out


def p_gaussian_noise(img, rng, k):
    noise = rng.normal(0, 25 * k, img.shape)  # float: no uint8 wrap-around
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def p_speckle(img, rng, k):
    noise = rng.normal(0, 0.1 * k, img.shape)
    return np.clip(img + img * noise, 0, 255).astype(np.uint8)


def p_motion_blur(img, rng, k):
    size = int(max(3, round(rng.uniform(5, 15) * k)))
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1
    rot = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    total = kernel.sum()
    if total == 0:
        return img
    return cv2.filter2D(img, -1, kernel / total, borderType=cv2.BORDER_REFLECT)


def p_cutout(img, rng, k):
    h, w = img.shape[:2]
    out = img.copy()
    cw = max(1, min(w, int(w * rng.uniform(0.125, 0.25) * min(k, 2))))
    ch = max(1, min(h, int(h * rng.uniform(0.125, 0.25) * min(k, 2))))
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    out[y:y + ch, x:x + cw] = 0
    return out


def p_random_erasing(img, rng, k):
    """Like cutout, but random area/aspect ratio and filled with random noise."""
    h, w = img.shape[:2]
    out = img.copy()
    area = min(rng.uniform(0.02, 0.2) * k, 0.6) * h * w
    aspect = rng.uniform(0.3, 3.3)
    ew = max(1, min(w, int(np.sqrt(area * aspect))))
    eh = max(1, min(h, int(np.sqrt(area / aspect))))
    x, y = rng.integers(0, w - ew + 1), rng.integers(0, h - eh + 1)
    out[y:y + eh, x:x + ew] = rng.integers(0, 256, (eh, ew, 3), dtype=np.uint8)
    return out


def p_edges(img, rng, k):
    return _pil_filter(img, ImageFilter.FIND_EDGES)


def p_emboss(img, rng, k):
    return _pil_filter(img, ImageFilter.EMBOSS)


def p_sharpen(img, rng, k):
    return _pil_filter(img, ImageFilter.SHARPEN)


def p_channel_shuffle(img, rng, k):
    return img[..., rng.permutation(3)]


def p_channel_drop(img, rng, k):
    out = img.copy()
    out[..., rng.integers(0, 3)] = 0  # zero a channel, keep 3-channel shape
    return out


def n_elastic(img, rng, k):
    """Non-linear warp. Strength scales with image size."""
    h, w = img.shape[:2]
    sigma = max(4.0, 0.04 * min(h, w))
    amp = 0.02 * min(h, w) * k

    def field():
        f = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        return (f / (f.std() + 1e-8) * amp).astype(np.float32)

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + field(), ys + field(), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


# name -> (kind, function). Order here is the order used in pipelines.
REGISTRY = {
    "Flipping": ("geo", g_flip),
    "Rotation": ("geo", g_rotate),
    "Translation": ("geo", g_translate),
    "Zooming": ("geo", g_zoom),
    "Shearing": ("geo", g_shear),
    "Random Cropping": ("geo", g_crop),
    "Perspective Warp": ("geo", g_perspective),
    "Affine Transformation": ("geo", g_affine),
    "Elastic Transformation": ("nonlinear", n_elastic),
    "Brightness": ("photo", p_brightness),
    "Contrast": ("photo", p_contrast),
    "Saturation": ("photo", p_saturation),
    "Hue": ("photo", p_hue),
    "Gaussian Blur": ("photo", p_blur),
    "Motion Blur": ("photo", p_motion_blur),
    "Salt & Pepper": ("photo", p_salt_pepper),
    "Gaussian Noise": ("photo", p_gaussian_noise),
    "Speckle Noise": ("photo", p_speckle),
    "Cutout": ("photo", p_cutout),
    "Random Erasing": ("photo", p_random_erasing),
    "Edge Detection": ("photo", p_edges),
    "Embossing": ("photo", p_emboss),
    "Sharpening": ("photo", p_sharpen),
    "Channel Shuffling": ("photo", p_channel_shuffle),
    "Channel Dropping": ("photo", p_channel_drop),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def transform_boxes(boxes, M, w, h, out_w=None, out_h=None, min_visible=0.3):
    """Transform normalized YOLO boxes [cls, xc, yc, bw, bh] with homography M.
    (w, h) is the source size, (out_w, out_h) the output size (defaults to the same)."""
    ow, oh = out_w or w, out_h or h
    out = []
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], float).T
        t = M @ pts
        t = t[:2] / t[2]
        nx1, ny1, nx2, ny2 = t[0].min(), t[1].min(), t[0].max(), t[1].max()
        full = (nx2 - nx1) * (ny2 - ny1)
        cx1, cy1, cx2, cy2 = max(nx1, 0), max(ny1, 0), min(nx2, ow), min(ny2, oh)
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 2 or ch < 2 or full <= 0 or cw * ch / full < min_visible:
            continue  # box left the frame or is mostly cropped away
        out.append([cls, (cx1 + cx2) / 2 / ow, (cy1 + cy2) / 2 / oh, cw / ow, ch / oh])
    return np.array(out, float).reshape(-1, 5)


def apply_methods(img, boxes, names, rng, k, border):
    """Apply methods to img. All geometric ones are merged into a single warp."""
    h, w = img.shape[:2]
    geo = [n for n in names if REGISTRY[n][0] == "geo"]
    if geo:
        M = np.eye(3)
        for n in geo:
            M = REGISTRY[n][1](h, w, rng, k) @ M
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=BORDER_MODES[border])
        if boxes is not None:
            boxes = transform_boxes(boxes, M, w, h)
    for n in names:
        if REGISTRY[n][0] != "geo":
            img = REGISTRY[n][1](img, rng, k)
    return img, boxes


def effective_methods(names, use_labels):
    """Non-linear warps cannot move bounding boxes, so drop them when labels are used."""
    return [n for n in names if not (use_labels and REGISTRY[n][0] == "nonlinear")]


def choose_for_pipeline(names, prob, rng):
    chosen = [n for n in names if rng.random() < prob]
    return chosen or [str(rng.choice(names))]


def collect_images(src, dst=None):
    dst_abs = os.path.abspath(dst) if dst else None
    files = []
    for root, dirs, fs in os.walk(src):
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != dst_abs]
        files += [os.path.join(root, f) for f in sorted(fs) if f.lower().endswith(IMAGE_EXTS)]
    return files


def no_images_message(src):
    exts = set()
    for _, _, fs in os.walk(src):
        exts.update(os.path.splitext(f)[1].lower() or "(no extension)" for f in fs)
    seen = ", ".join(sorted(exts)) if exts else "none (folder is empty)"
    return (f"No supported images found in:\n{src}\n\nFile types seen: {seen}\n"
            f"Supported: {', '.join(IMAGE_EXTS)}")


def load_image(path):
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def save_image(arr, path):
    Image.fromarray(arr).save(path, quality=95)  # quality is ignored by non-JPEG formats


def read_labels(img_path):
    """YOLO bbox labels next to the image (same stem, .txt). None if no label file."""
    p = os.path.splitext(img_path)[0] + ".txt"
    if not os.path.isfile(p):
        return None
    rows = []
    with open(p) as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{os.path.basename(p)} line {n}: only 5-value YOLO boxes are supported")
            rows.append([float(x) for x in parts])
    return np.array(rows, float).reshape(-1, 5)


def write_labels(boxes, path):
    with open(path, "w") as f:
        for c, xc, yc, bw, bh in boxes:
            f.write(f"{int(c)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_job(cfg, q, stop):
    """Worker thread. Talks to the GUI only through the queue."""
    generated = failed = 0
    try:
        files = collect_images(cfg["src"], cfg["dst"])
        rng = np.random.default_rng(cfg["seed"])
        separate = cfg["mode"] == "separate"
        jobs = [(slug(n), [n]) for n in cfg["names"]] if separate else [("aug", cfg["names"])]
        for idx, path in enumerate(files, 1):
            if stop.is_set():
                break
            try:
                img = load_image(path)
                boxes = read_labels(path) if cfg["use_labels"] else None
                rel = os.path.relpath(os.path.dirname(path), cfg["src"])
                out_dir = os.path.join(cfg["dst"], rel)
                os.makedirs(out_dir, exist_ok=True)
                stem, ext = os.path.splitext(os.path.basename(path))
                out_ext = ext.lower() if ext.lower() in KEEP_EXTS else ".png"

                if cfg["originals"]:
                    dest = os.path.join(out_dir, os.path.basename(path))
                    if not os.path.exists(dest):
                        shutil.copy2(path, dest)
                        if boxes is not None:
                            shutil.copy2(os.path.splitext(path)[0] + ".txt",
                                         os.path.splitext(dest)[0] + ".txt")

                for tag, methods in jobs:
                    for i in range(cfg["copies"]):
                        chosen = methods if separate else choose_for_pipeline(methods, cfg["prob"], rng)
                        out, b = apply_methods(img, boxes, chosen, rng, cfg["k"], cfg["border"])
                        out_path = unique_path(os.path.join(out_dir, f"{stem}_{tag}_{i + 1}{out_ext}"))
                        save_image(out, out_path)
                        if b is not None:
                            write_labels(b, os.path.splitext(out_path)[0] + ".txt")
                        generated += 1
            except Exception as e:  # one bad file must not stop the run
                failed += 1
                q.put(("log", f"Skipped {os.path.basename(path)}: {e}"))
            q.put(("progress", idx, len(files)))
    except Exception as e:
        q.put(("log", f"Fatal error: {e}"))
    q.put(("done", generated, failed, stop.is_set()))


# ---------------------------------------------------------------------------
# Rename and resize tools
# ---------------------------------------------------------------------------
RESIZE_MODES = {
    "Stretch to exact size (may distort)": "stretch",
    "Keep aspect ratio, pad (letterbox)": "letterbox",
    "Keep aspect ratio, center crop": "crop",
}
RESIZE_PRESETS = ["320x320", "416x416", "512x512", "640x640", "720x720", "768x768", "1024x1024"]


def natural_key(name):
    """Sort key so that 2.jpg comes before 10.jpg."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_images(folder, natural=True):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))]
    files.sort(key=natural_key if natural else None)
    return files


def plan_renames(folder, start, pad, natural, with_labels):
    """Return [(old_name, new_name), ...]. Matching .txt labels follow their image."""
    pairs, used_labels = [], set()
    for i, f in enumerate(list_images(folder, natural), start):
        stem, ext = os.path.splitext(f)
        new_stem = str(i).zfill(pad)
        pairs.append((f, new_stem + ext))
        label = stem + ".txt"
        if with_labels and label not in used_labels and os.path.isfile(os.path.join(folder, label)):
            used_labels.add(label)
            pairs.append((label, new_stem + ".txt"))
    return pairs


def apply_renames(folder, pairs):
    """Two-phase rename (old -> temp -> new) so names can never overwrite each other.
    Rolls everything back if any step fails."""
    pairs = [(o, n) for o, n in pairs if o != n]
    if not pairs:
        return
    existing = {f.lower() for f in os.listdir(folder)}
    sources = {o.lower() for o, _ in pairs}
    clashes = [n for _, n in pairs if n.lower() in existing and n.lower() not in sources]
    if clashes:
        raise FileExistsError("would overwrite existing file(s): " + ", ".join(clashes[:5]))
    tag = uuid.uuid4().hex[:8]
    performed = []

    def move(a, b):
        os.rename(os.path.join(folder, a), os.path.join(folder, b))
        performed.append((a, b))

    try:
        temps = []
        for i, (o, n) in enumerate(pairs):
            t = f"__augmento_{tag}_{i}{os.path.splitext(o)[1]}"
            move(o, t)
            temps.append((t, n))
        for t, n in temps:
            move(t, n)
    except Exception:
        for a, b in reversed(performed):
            try:
                os.rename(os.path.join(folder, b), os.path.join(folder, a))
            except OSError:
                pass
        raise


def run_rename(cfg, q, stop):
    renamed = failed = 0
    rows = []
    try:
        src = cfg["src"]
        dirs = [r for r, _, _ in os.walk(src)] if cfg["recursive"] else [src]
        for i, d in enumerate(dirs, 1):
            if stop.is_set():
                break
            try:
                pairs = plan_renames(d, cfg["start"], cfg["pad"], cfg["natural"], cfg["labels"])
                changed = [(o, n) for o, n in pairs if o != n]
                apply_renames(d, changed)
                renamed += sum(1 for o, _ in pairs if o.lower().endswith(IMAGE_EXTS))
                rel = os.path.relpath(d, src)
                rows += [(os.path.normpath(os.path.join(rel, o)), os.path.normpath(os.path.join(rel, n)))
                         for o, n in changed]
            except Exception as e:
                failed += 1
                q.put(("log", f"Skipped folder {d}: {e}"))
            q.put(("progress", i, len(dirs)))
        if cfg["save_map"] and rows:
            try:
                map_path = unique_path(os.path.join(src, "rename_map.csv"))
                with open(map_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["old_name", "new_name"])
                    w.writerows(rows)
                q.put(("log", f"Saved rename map: {map_path}"))
            except OSError as e:
                q.put(("log", f"Could not save rename map: {e}"))
    except Exception as e:
        q.put(("log", f"Fatal error: {e}"))
    q.put(("done", renamed, failed, stop.is_set()))


def resize_image(img, W, H, mode):
    """Resize to exactly WxH. Returns (image, 3x3 matrix mapping old pixels to new)."""
    h, w = img.shape[:2]
    if mode == "stretch":
        nw, nh = W, H
    elif mode == "letterbox":
        s = min(W / w, H / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
    else:  # crop
        s = max(W / w, H / h)
        nw, nh = max(W, round(w * s)), max(H, round(h * s))
    interp = cv2.INTER_AREA if nw * nh < w * h else cv2.INTER_CUBIC
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    ox = oy = 0
    if mode == "letterbox":
        out = np.full((H, W, 3), 114, np.uint8)
        ox, oy = (W - nw) // 2, (H - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = r
    elif mode == "crop":
        cx, cy = (nw - W) // 2, (nh - H) // 2
        out = r[cy:cy + H, cx:cx + W]
        ox, oy = -cx, -cy
    else:
        out = r
    M = np.array([[nw / w, 0, ox], [0, nh / h, oy], [0, 0, 1]], float)
    return out, M


def run_resize(cfg, q, stop):
    done = failed = 0
    try:
        files = collect_images(cfg["src"], cfg["dst"])
        W, H, mode = cfg["w"], cfg["h"], cfg["mode"]
        for idx, path in enumerate(files, 1):
            if stop.is_set():
                break
            try:
                img = load_image(path)
                h, w = img.shape[:2]
                label_path = os.path.splitext(path)[0] + ".txt"
                has_label = cfg["use_labels"] and os.path.isfile(label_path)
                boxes = read_labels(path) if has_label and mode != "stretch" else None
                out, M = resize_image(img, W, H, mode)
                rel = os.path.relpath(os.path.dirname(path), cfg["src"])
                out_dir = os.path.join(cfg["dst"], rel)
                os.makedirs(out_dir, exist_ok=True)
                stem, ext = os.path.splitext(os.path.basename(path))
                out_path = os.path.join(out_dir, stem + (ext if ext.lower() in KEEP_EXTS else ".png"))
                save_image(out, out_path)
                if has_label:
                    dest = os.path.splitext(out_path)[0] + ".txt"
                    if mode == "stretch":  # normalized labels stay valid when stretching
                        shutil.copy2(label_path, dest)
                    else:
                        write_labels(transform_boxes(boxes, M, w, h, W, H), dest)
                done += 1
            except Exception as e:
                failed += 1
                q.put(("log", f"Skipped {os.path.basename(path)}: {e}"))
            q.put(("progress", idx, len(files)))
    except Exception as e:
        q.put(("log", f"Fatal error: {e}"))
    q.put(("done", done, failed, stop.is_set()))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def draw_boxes(img, boxes):
    if boxes is None or len(boxes) == 0:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    t = max(2, min(h, w) // 200)
    for _, xc, yc, bw, bh in boxes:
        p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        cv2.rectangle(out, p1, p2, (0, 255, 0), t)
    return out


def to_photo(arr, size=380):
    pil = Image.fromarray(arr)
    pil.thumbnail((size, size))
    return ImageTk.PhotoImage(pil)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Augmento v2.0")
        root.minsize(900, 800)
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.running = False
        self.verb = "written"
        self.image_count = 0

        # augment
        self.src, self.dst = tk.StringVar(), tk.StringVar()
        self.mode = tk.StringVar(value="separate")
        self.copies = tk.IntVar(value=1)
        self.prob = tk.DoubleVar(value=0.5)
        self.intensity = tk.DoubleVar(value=1.0)
        self.seed = tk.StringVar()
        self.border = tk.StringVar(value="Reflect")
        self.use_labels = tk.BooleanVar(value=False)
        self.originals = tk.BooleanVar(value=False)
        self.method_vars = {n: tk.BooleanVar() for n in REGISTRY}
        # rename
        self.rn_dir = tk.StringVar()
        self.rn_start = tk.IntVar(value=1)
        self.rn_pad = tk.IntVar(value=0)
        self.rn_natural = tk.BooleanVar(value=True)
        self.rn_sub = tk.BooleanVar(value=False)
        self.rn_labels = tk.BooleanVar(value=True)
        self.rn_map = tk.BooleanVar(value=True)
        # resize
        self.rs_src, self.rs_dst = tk.StringVar(), tk.StringVar()
        self.rs_w, self.rs_h = tk.IntVar(value=640), tk.IntVar(value=640)
        self.rs_preset = tk.StringVar(value="640x640")
        self.rs_mode = tk.StringVar(value=list(RESIZE_MODES)[0])
        self.rs_labels = tk.BooleanVar(value=False)

        self.build_ui()
        for v in [self.mode, self.copies, self.originals, self.use_labels, *self.method_vars.values()]:
            v.trace_add("write", lambda *a: self.update_estimate())
        self.rn_sub.trace_add("write", lambda *a: self.refresh_rn())
        self.update_estimate()
        self.refresh_rn()
        self.refresh_rs()

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _get(var, default):
        try:
            return var.get()
        except tk.TclError:  # empty spinbox
            return default

    def selected(self):
        names = [n for n, v in self.method_vars.items() if v.get()]
        return effective_methods(names, self.use_labels.get())

    def log(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def browse(self, var, after=None):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)
            if after:
                after()

    def path_row(self, parent, row, label, var, browse_cmd, on_change=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(parent, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        if on_change:
            e.bind("<FocusOut>", lambda ev: on_change())
        ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    # -- UI ---------------------------------------------------------------
    def build_ui(self):
        r = self.root
        ttk.Label(r, text="Augmento", font=("Trebuchet MS", 20, "bold"),
                  foreground="green").pack(pady=(10, 0))
        nb = ttk.Notebook(r)
        nb.pack(fill="x", padx=6, pady=4)
        for title, builder in (("Augment", self.build_augment_tab),
                               ("Rename", self.build_rename_tab),
                               ("Resize", self.build_resize_tab)):
            tab = ttk.Frame(nb)
            nb.add(tab, text=f"   {title}   ")
            builder(tab)

        bar = ttk.Frame(r)
        bar.pack(fill="x", padx=10)
        self.progress = ttk.Progressbar(bar)
        self.progress.pack(side="left", fill="x", expand=True)
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self.stop.set, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.status = ttk.Label(r, text="Ready.", foreground="blue")
        self.status.pack(anchor="w", padx=10, pady=4)
        self.log_box = tk.Text(r, height=6, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def build_augment_tab(self, tab):
        f = ttk.LabelFrame(tab, text="Folders", padding=8)
        f.pack(fill="x", padx=10, pady=6)
        self.path_row(f, 0, "Original dataset:", self.src,
                      lambda: self.browse(self.src, self.refresh_count), self.refresh_count)
        self.path_row(f, 1, "Output folder:", self.dst, lambda: self.browse(self.dst))

        o = ttk.LabelFrame(tab, text="Options", padding=8)
        o.pack(fill="x", padx=10, pady=6)
        ttk.Radiobutton(o, text="Each method separately", variable=self.mode,
                        value="separate").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(o, text="Combined pipeline (random mix per copy)", variable=self.mode,
                        value="pipeline").grid(row=0, column=1, columnspan=3, sticky="w")
        ttk.Label(o, text="Copies per image/method:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(o, from_=1, to=100, textvariable=self.copies, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(o, text="Pipeline probability per step:").grid(row=1, column=2, sticky="e")
        ttk.Spinbox(o, from_=0.1, to=1.0, increment=0.1, format="%.1f", textvariable=self.prob,
                    width=5).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(o, text="Intensity:").grid(row=2, column=0, sticky="w")
        self.int_lbl = ttk.Label(o, text="1.0x", width=5)
        ttk.Scale(o, from_=0.2, to=2.0, variable=self.intensity, length=160,
                  command=lambda v: self.int_lbl.config(text=f"{float(v):.1f}x")
                  ).grid(row=2, column=1, sticky="w")
        self.int_lbl.grid(row=2, column=2, sticky="w")
        ttk.Label(o, text="Seed (optional):").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(o, textvariable=self.seed, width=12).grid(row=3, column=1, sticky="w")
        ttk.Label(o, text="Border fill:").grid(row=3, column=2, sticky="e")
        ttk.Combobox(o, textvariable=self.border, values=list(BORDER_MODES), state="readonly",
                     width=16).grid(row=3, column=3, sticky="w", padx=4)
        ttk.Checkbutton(o, text="Use YOLO bbox labels (.txt next to each image)",
                        variable=self.use_labels).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(o, text="Copy originals to output too",
                        variable=self.originals).grid(row=4, column=2, columnspan=2, sticky="w")

        m = ttk.LabelFrame(tab, text="Augmentations", padding=8)
        m.pack(fill="x", padx=10, pady=6)
        for i, name in enumerate(REGISTRY):
            ttk.Checkbutton(m, text=name, variable=self.method_vars[name]).grid(
                row=i // 5, column=i % 5, sticky="w", padx=6, pady=2)
        bar = ttk.Frame(m)
        bar.grid(row=10, column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Button(bar, text="Select all", command=lambda: self.set_all(True)).pack(side="left")
        ttk.Button(bar, text="Clear", command=lambda: self.set_all(False)).pack(side="left", padx=6)

        self.estimate = ttk.Label(tab, foreground="blue")
        self.estimate.pack(anchor="w", padx=10)
        b = ttk.Frame(tab)
        b.pack(fill="x", padx=10, pady=6)
        ttk.Button(b, text="Preview", command=self.preview).pack(side="left")
        self.gen_btn = ttk.Button(b, text="Generate Augmented Images", command=self.start)
        self.gen_btn.pack(side="left", padx=6)
        ttk.Button(b, text="See explanations",
                   command=lambda: webbrowser.open(EXPLAIN_URL)).pack(side="right")

    def build_rename_tab(self, tab):
        f = ttk.LabelFrame(tab, text="Folder", padding=8)
        f.pack(fill="x", padx=10, pady=6)
        self.path_row(f, 0, "Image folder:", self.rn_dir,
                      lambda: self.browse(self.rn_dir, self.refresh_rn), self.refresh_rn)

        o = ttk.LabelFrame(tab, text="Options", padding=8)
        o.pack(fill="x", padx=10, pady=6)
        ttk.Label(o, text="Start number:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(o, from_=0, to=999999, textvariable=self.rn_start, width=8).grid(
            row=0, column=1, sticky="w", padx=6)
        ttk.Label(o, text="Zero padding (0 = none):").grid(row=0, column=2, sticky="e", padx=(20, 0))
        ttk.Spinbox(o, from_=0, to=10, textvariable=self.rn_pad, width=5).grid(
            row=0, column=3, sticky="w", padx=6)
        ttk.Checkbutton(o, text="Natural sort (2 before 10)", variable=self.rn_natural).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(o, text="Include subfolders (each folder is numbered separately)",
                        variable=self.rn_sub).grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(o, text="Also rename matching .txt label files",
                        variable=self.rn_labels).grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(o, text="Save rename_map.csv (old name -> new name) in the folder",
                        variable=self.rn_map).grid(row=4, column=0, columnspan=4, sticky="w")

        ttk.Label(tab, text="Files are renamed in place, e.g. photo.jpg becomes 1.jpg, 2.jpg, 3.jpg ...",
                  foreground="gray").pack(anchor="w", padx=10)
        self.rn_info = ttk.Label(tab, foreground="blue")
        self.rn_info.pack(anchor="w", padx=10, pady=(4, 0))
        b = ttk.Frame(tab)
        b.pack(fill="x", padx=10, pady=6)
        self.rename_btn = ttk.Button(b, text="Rename images", command=self.start_rename)
        self.rename_btn.pack(side="left")

    def build_resize_tab(self, tab):
        f = ttk.LabelFrame(tab, text="Folders", padding=8)
        f.pack(fill="x", padx=10, pady=6)
        self.path_row(f, 0, "Input folder:", self.rs_src,
                      lambda: self.browse(self.rs_src, self.refresh_rs), self.refresh_rs)
        self.path_row(f, 1, "Output folder:", self.rs_dst, lambda: self.browse(self.rs_dst))

        o = ttk.LabelFrame(tab, text="Options", padding=8)
        o.pack(fill="x", padx=10, pady=6)
        ttk.Label(o, text="Preset:").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(o, textvariable=self.rs_preset, values=RESIZE_PRESETS, width=12)
        cb.grid(row=0, column=1, sticky="w", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())
        ttk.Label(o, text="Width:").grid(row=0, column=2, sticky="e", padx=(20, 0))
        ttk.Spinbox(o, from_=16, to=8192, textvariable=self.rs_w, width=7).grid(row=0, column=3, padx=6)
        ttk.Label(o, text="Height:").grid(row=0, column=4, sticky="e")
        ttk.Spinbox(o, from_=16, to=8192, textvariable=self.rs_h, width=7).grid(row=0, column=5, padx=6)
        ttk.Label(o, text="Mode:").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(o, textvariable=self.rs_mode, values=list(RESIZE_MODES), state="readonly",
                     width=36).grid(row=1, column=1, columnspan=5, sticky="w", padx=6)
        ttk.Checkbutton(o, text="Also process YOLO labels (.txt next to each image)",
                        variable=self.rs_labels).grid(row=2, column=0, columnspan=6, sticky="w")

        ttk.Label(tab, text="Stretch keeps YOLO labels valid as-is. Letterbox and crop update the boxes.\n"
                            "Originals are never modified. Output files keep their names.",
                  foreground="gray").pack(anchor="w", padx=10)
        self.rs_info = ttk.Label(tab, foreground="blue")
        self.rs_info.pack(anchor="w", padx=10, pady=(4, 0))
        b = ttk.Frame(tab)
        b.pack(fill="x", padx=10, pady=6)
        self.resize_btn = ttk.Button(b, text="Resize images", command=self.start_resize)
        self.resize_btn.pack(side="left")

    # -- shared task runner -----------------------------------------------
    def set_busy(self, busy):
        for btn in (self.gen_btn, self.rename_btn, self.resize_btn):
            btn.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")

    def begin_task(self, target, cfg, verb):
        self.stop.clear()
        self.running = True
        self.verb = verb
        self.set_busy(True)
        self.progress["value"] = 0
        threading.Thread(target=target, args=(cfg, self.q, self.stop), daemon=True).start()
        self.root.after(100, self.poll)

    def poll(self):
        try:
            while True:
                kind, *data = self.q.get_nowait()
                if kind == "progress":
                    self.progress["maximum"], self.progress["value"] = data[1], data[0]
                elif kind == "log":
                    self.log(data[0])
                elif kind == "done":
                    count, failed, cancelled = data
                    self.running = False
                    self.set_busy(False)
                    msg = f"{'Cancelled' if cancelled else 'Finished'}: {count} image(s) {self.verb}"
                    msg += f", {failed} item(s) skipped (see log)." if failed else "."
                    self.status.config(text=msg)
                    self.refresh_rn()
                    messagebox.showinfo("Augmento", msg)
        except queue.Empty:
            pass
        if self.running:
            self.root.after(100, self.poll)

    # -- augment ----------------------------------------------------------
    def set_all(self, value):
        for v in self.method_vars.values():
            v.set(value)

    def refresh_count(self):
        src = self.src.get()
        self.image_count = len(collect_images(src, self.dst.get() or None)) if os.path.isdir(src) else 0
        self.update_estimate()

    def update_estimate(self):
        n = len(self.selected())
        copies = max(1, int(self._get(self.copies, 1)))
        if self.image_count == 0:
            text = "Choose an original dataset folder that contains images."
        elif n == 0:
            text = f"{self.image_count} image(s) found. Select at least one augmentation."
        else:
            per = (n if self.mode.get() == "separate" else 1) * copies + int(self.originals.get())
            text = f"{self.image_count} image(s) found. This run will write {self.image_count * per} image(s)."
        self.estimate.config(text=text)

    def make_config(self):
        src, dst = self.src.get().strip(), self.dst.get().strip()
        if not os.path.isdir(src):
            messagebox.showerror("Error", "Please specify a valid original dataset folder.")
            return None
        if not dst:
            messagebox.showerror("Error", "Please specify the output folder.")
            return None
        if os.path.abspath(src) == os.path.abspath(dst):
            messagebox.showerror("Error", "Output folder must differ from the original folder.")
            return None
        names = self.selected()
        if not names:
            messagebox.showerror("Error", "Please select at least one augmentation.")
            return None
        if not collect_images(src, dst):
            messagebox.showerror("Error", no_images_message(src))
            return None
        if self.use_labels.get() and any(self.method_vars[n].get() and REGISTRY[n][0] == "nonlinear"
                                         for n in REGISTRY):
            self.log("Elastic Transformation skipped: it cannot be applied with bounding box labels.")
        seed_text = self.seed.get().strip()
        try:
            seed = int(seed_text) if seed_text else random.randrange(2 ** 32)
        except ValueError:
            messagebox.showerror("Error", "Seed must be an integer.")
            return None
        os.makedirs(dst, exist_ok=True)
        return {
            "src": src, "dst": dst, "names": names, "mode": self.mode.get(),
            "copies": max(1, int(self._get(self.copies, 1))),
            "prob": min(1.0, max(0.0, float(self._get(self.prob, 0.5)))),
            "k": float(self.intensity.get()), "seed": seed, "border": self.border.get(),
            "use_labels": self.use_labels.get(), "originals": self.originals.get(),
        }

    def start(self):
        if self.running:
            return
        cfg = self.make_config()
        if not cfg:
            return
        self.log(f"Started. Seed: {cfg['seed']}")
        self.begin_task(run_job, cfg, "written")

    def preview(self):
        src = self.src.get().strip()
        names = self.selected()
        if not os.path.isdir(src):
            messagebox.showerror("Error", "Please specify a valid original dataset folder.")
            return
        if not names:
            messagebox.showerror("Error", "Please select at least one augmentation.")
            return
        files = collect_images(src, self.dst.get().strip() or None)
        if not files:
            messagebox.showerror("Error", no_images_message(src))
            return
        k, prob, border = float(self.intensity.get()), float(self._get(self.prob, 0.5)), self.border.get()
        use_labels, separate = self.use_labels.get(), self.mode.get() == "separate"

        win = tk.Toplevel(self.root)
        win.title("Preview (left: original, right: augmented)")
        left, right, cap = ttk.Label(win), ttk.Label(win), ttk.Label(win)
        left.grid(row=0, column=0, padx=6, pady=6)
        right.grid(row=0, column=1, padx=6, pady=6)
        cap.grid(row=1, column=0, columnspan=2)

        def roll():
            rng = np.random.default_rng()
            try:
                path = random.choice(files)
                img = load_image(path)
                boxes = read_labels(path) if use_labels else None
                chosen = [str(rng.choice(names))] if separate else choose_for_pipeline(names, prob, rng)
                out, b = apply_methods(img, boxes, chosen, rng, k, border)
            except Exception as e:
                messagebox.showerror("Preview failed", str(e), parent=win)
                return
            win.photos = [to_photo(draw_boxes(img, boxes)), to_photo(draw_boxes(out, b))]
            left.config(image=win.photos[0])
            right.config(image=win.photos[1])
            cap.config(text=", ".join(chosen))

        ttk.Button(win, text="Re-roll", command=roll).grid(row=2, column=0, columnspan=2, pady=6)
        roll()

    # -- rename -----------------------------------------------------------
    @staticmethod
    def count_rename(folder, recursive):
        dirs = [r for r, _, _ in os.walk(folder)] if recursive else [folder]
        return sum(len(list_images(d)) for d in dirs)

    def refresh_rn(self):
        folder = self.rn_dir.get().strip()
        if not os.path.isdir(folder):
            self.rn_info.config(text="Choose a folder that contains images.")
            return
        n = self.count_rename(folder, self.rn_sub.get())
        self.rn_info.config(text=f"{n} image(s) found." if n else "No supported images found.")

    def start_rename(self):
        if self.running:
            return
        folder = self.rn_dir.get().strip()
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Please specify a valid image folder.")
            return
        recursive = self.rn_sub.get()
        n = self.count_rename(folder, recursive)
        if n == 0:
            messagebox.showerror("Error", no_images_message(folder))
            return
        cfg = {
            "src": folder, "recursive": recursive,
            "start": max(0, int(self._get(self.rn_start, 1))),
            "pad": max(0, int(self._get(self.rn_pad, 0))),
            "natural": self.rn_natural.get(), "labels": self.rn_labels.get(),
            "save_map": self.rn_map.get(),
        }
        example = ""
        pairs = plan_renames(folder, cfg["start"], cfg["pad"], cfg["natural"], cfg["labels"])
        if pairs:
            example = f"\n\nExample: {pairs[0][0]}  ->  {pairs[0][1]}"
        if not messagebox.askyesno(
                "Rename images",
                f"Rename {n} image(s) in place in:\n{folder}{example}\n\n"
                "This changes files on disk. Continue?"):
            return
        self.begin_task(run_rename, cfg, "renamed")

    # -- resize -----------------------------------------------------------
    def apply_preset(self):
        try:
            w, h = self.rs_preset.get().lower().split("x")
            self.rs_w.set(int(w))
            self.rs_h.set(int(h))
        except ValueError:
            pass

    def refresh_rs(self):
        src = self.rs_src.get().strip()
        if not os.path.isdir(src):
            self.rs_info.config(text="Choose an input folder that contains images.")
            return
        n = len(collect_images(src, self.rs_dst.get().strip() or None))
        self.rs_info.config(text=f"{n} image(s) found." if n else "No supported images found.")

    def start_resize(self):
        if self.running:
            return
        src, dst = self.rs_src.get().strip(), self.rs_dst.get().strip()
        if not os.path.isdir(src):
            messagebox.showerror("Error", "Please specify a valid input folder.")
            return
        if not dst:
            messagebox.showerror("Error", "Please specify the output folder.")
            return
        if os.path.abspath(src) == os.path.abspath(dst):
            messagebox.showerror("Error", "Output folder must differ from the input folder.")
            return
        w, h = int(self._get(self.rs_w, 0)), int(self._get(self.rs_h, 0))
        if w < 1 or h < 1:
            messagebox.showerror("Error", "Width and height must be positive numbers.")
            return
        if not collect_images(src, dst):
            messagebox.showerror("Error", no_images_message(src))
            return
        os.makedirs(dst, exist_ok=True)
        cfg = {"src": src, "dst": dst, "w": w, "h": h, "mode": RESIZE_MODES[self.rs_mode.get()],
               "use_labels": self.rs_labels.get()}
        self.begin_task(run_resize, cfg, "resized")


if __name__ == "__main__":
    try:  # sharper text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    root.report_callback_exception = lambda *a: messagebox.showerror(
        "Unexpected error", "".join(__import__("traceback").format_exception(*a))[-1500:])
    App(root)
    root.mainloop()
