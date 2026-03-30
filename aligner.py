# aligner.py
# GUI: Align multiple FITS to a reference + strict common-overlap crop + optional normalization + optional PSF match
# PLUS: Separate "JWST Destripe (rate/cal)" window (community method) opened by a button + Tools menu item
#
# Notes:
# - Destripe window works on *_rate.fits or *_cal.fits (NOT *_i2d/_drz/_drc)
# - Destripe uses photutils segmentation + mask dilation + row-median subtraction (optional 4-slice amp-like split)
# - No JWST pipeline dependency (no jwst install required)
#
# Dependencies:
#   pip install numpy scipy astropy reproject photutils
#
# Run:
#   python aligner.py

import os
import re
import time
import sys
import threading
import queue
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scipy.ndimage import gaussian_filter, binary_dilation

# Destripe deps (photutils)
HAVE_DESTRIPE = True
DESTRIPE_IMPORT_ERR = ""
try:
    from astropy.convolution import convolve
    from photutils.segmentation import detect_sources, make_2dgaussian_kernel
except Exception as e:
    HAVE_DESTRIPE = False
    DESTRIPE_IMPORT_ERR = str(e)


# =========================
# FITS + WCS utilities
# =========================

def load_2d_and_header(path: str):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
                data = hdu.data
                header = hdu.header
                break
        else:
            raise RuntimeError(f"No image data found in: {path}")

    if data.ndim > 2:
        data = data[0]

    data = np.array(data, dtype=np.float32)
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = np.nanmedian(data)

    return data, header


def bbox_from_mask(mask: np.ndarray):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return y0, y1, x0, x1


def crop_with_wcs(data: np.ndarray, header, bbox):
    y0, y1, x0, x1 = bbox
    cropped = data[y0:y1, x0:x1]

    w = WCS(header)
    w2 = w.slice((slice(y0, y1), slice(x0, x1)))
    new_header = w2.to_header()

    for k in header:
        if k not in new_header and k not in ("HISTORY", "COMMENT"):
            try:
                new_header[k] = header[k]
            except Exception:
                pass

    return cropped, new_header


def safe_filename(name: str):
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name.strip())
    out = out.replace(" ", "_")
    return out if out else "unnamed"


def try_get_header_value(hdr, keys):
    for k in keys:
        if k in hdr and hdr[k] not in (None, ""):
            return str(hdr[k]).strip()
    return ""


FILTER_RE = re.compile(r"(F\d{3,4}[A-Z])", re.IGNORECASE)


def parse_filter_code(text: str):
    if not text:
        return ""
    m = FILTER_RE.search(text.upper())
    return m.group(1) if m else ""


def filter_code_to_microns(code: str):
    if not code or not code.upper().startswith("F"):
        return None
    digits = re.findall(r"\d+", code)
    if not digits:
        return None
    val = int(digits[0])
    return val / 100.0


def suggest_name_from_fits(path: str):
    base = os.path.basename(path)
    try:
        _, hdr = load_2d_and_header(path)
    except Exception:
        code = parse_filter_code(base)
        return code or os.path.splitext(base)[0]

    instr = try_get_header_value(hdr, ["INSTRUME", "INSTRUMENT"]).lower()
    filt = try_get_header_value(hdr, ["FILTER", "FILTER1", "FILTER2"])
    pupil = try_get_header_value(hdr, ["PUPIL"])
    band = try_get_header_value(hdr, ["BAND"])

    code = (
        parse_filter_code(filt) or
        parse_filter_code(pupil) or
        parse_filter_code(band) or
        parse_filter_code(base)
    )

    if not code:
        return os.path.splitext(base)[0]

    if "nircam" in instr:
        return f"nircam_{code}"
    if "miri" in instr:
        return f"miri_{code}"
    if "nirspec" in instr:
        return f"nirspec_{code}"
    if "niriss" in instr:
        return f"niriss_{code}"
    return code


def estimate_pixscale_arcsec(header):
    try:
        w = WCS(header)
        ps = w.proj_plane_pixel_scales()  # degrees/pixel
        if ps is not None and len(ps) >= 2:
            deg_per_pix = float(np.mean(np.abs(ps[:2])))
            return deg_per_pix * 3600.0
    except Exception:
        pass

    if "CDELT1" in header and header["CDELT1"]:
        try:
            return abs(float(header["CDELT1"])) * 3600.0
        except Exception:
            return None
    return None


def robust_detail_score(data):
    finite = data[np.isfinite(data)]
    if finite.size < 1000:
        return 0.0
    lo = np.percentile(finite, 1.0)
    hi = np.percentile(finite, 99.5)
    x = np.clip(data, lo, hi)
    x = np.where(np.isfinite(x), x, np.nanmedian(finite))
    hp = x - gaussian_filter(x, sigma=2.0)
    return float(np.nanstd(hp))


def choose_best_reference(paths):
    rows = []
    for p in paths:
        try:
            d, h = load_2d_and_header(p)
            finite_frac = float(np.isfinite(d).mean())
            detail = robust_detail_score(d)
            ps = estimate_pixscale_arcsec(h)
            ps_term = 0.0 if ps is None or ps <= 0 else (1.0 / ps)
            rows.append((p, finite_frac, detail, ps, ps_term))
        except Exception:
            rows.append((p, 0.0, 0.0, None, 0.0))

    finite_vals = np.array([r[1] for r in rows], dtype=float)
    detail_vals = np.array([r[2] for r in rows], dtype=float)
    ps_vals = np.array([r[4] for r in rows], dtype=float)

    def norm(v):
        mn, mx = float(np.min(v)), float(np.max(v))
        if mx <= mn:
            return np.zeros_like(v)
        return (v - mn) / (mx - mn)

    nf = norm(finite_vals)
    nd = norm(detail_vals)
    npix = norm(ps_vals)

    scores = 0.6 * nd + 0.3 * nf + 0.1 * npix
    best_i = int(np.argmax(scores))
    best_path = rows[best_i][0]

    table = []
    for i, r in enumerate(rows):
        table.append({
            "path": r[0],
            "finite": r[1],
            "detail": r[2],
            "pixscale_arcsec": r[3],
            "score": float(scores[i]),
        })
    return best_path, table


def write_log(out_dir, lines):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
    return log_path


# =========================
# Processing core (align + common crop)
# =========================

def normalize_to_reference(arr, ref_arr, mask, mode="median"):
    a = arr[mask & np.isfinite(arr)]
    r = ref_arr[mask & np.isfinite(ref_arr)]
    if a.size < 100 or r.size < 100:
        return arr, 1.0

    if mode == "median":
        av = float(np.median(a))
        rv = float(np.median(r))
    elif mode == "p95":
        av = float(np.percentile(a, 95))
        rv = float(np.percentile(r, 95))
    elif mode == "background":
        av = float(np.percentile(a, 20))
        rv = float(np.percentile(r, 20))
    else:
        av = float(np.median(a))
        rv = float(np.median(r))

    if not np.isfinite(av) or not np.isfinite(rv) or abs(av) < 1e-12:
        return arr, 1.0

    s = rv / av
    return arr * s, s


def psf_match_to_worst(images_by_name, names_in_order):
    wl = {}
    for nm in names_in_order:
        code = parse_filter_code(nm)
        wl[nm] = filter_code_to_microns(code) if code else None

    known = [(nm, wl[nm]) for nm in names_in_order if wl[nm] is not None]
    if len(known) < 2:
        return images_by_name, ["PSF match skipped: could not parse wavelengths from names."]

    lam_min = min(v for _, v in known)
    lam_max = max(v for _, v in known)

    base_fwhm_min_pix = 2.0
    fwhm_target = base_fwhm_min_pix * (lam_max / lam_min)
    sig_target = fwhm_target / 2.355

    notes = [f"PSF match: lam_min={lam_min:.2f}um lam_max={lam_max:.2f}um target_fwhm={fwhm_target:.2f}px"]

    out = {}
    for nm in names_in_order:
        img = images_by_name[nm]
        mic = wl.get(nm, None)
        if mic is None:
            out[nm] = img
            notes.append(f"PSF match: {nm} skipped (no wavelength parsed).")
            continue

        fwhm_cur = base_fwhm_min_pix * (mic / lam_min)
        sig_cur = fwhm_cur / 2.355
        sig_add_sq = sig_target * sig_target - sig_cur * sig_cur
        if sig_add_sq <= 1e-6:
            out[nm] = img
            continue

        sig_add = float(np.sqrt(sig_add_sq))
        out[nm] = gaussian_filter(img, sigma=sig_add)
        notes.append(f"PSF match: {nm} convolved sigma_add={sig_add:.3f}px")

    return out, notes


def align_crop_many(reference_path: str, reference_name: str,
                    items: list, out_base_dir: str,
                    normalize_enabled: bool, normalize_mode: str,
                    psf_match_enabled: bool,
                    progress_cb=None, cancel_cb=None):
    out_dir = os.path.join(out_base_dir, "aligned_cropped")
    os.makedirs(out_dir, exist_ok=True)

    log_lines = []
    log_lines.append(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Output folder: {out_dir}")
    log_lines.append(f"Reference path: {reference_path}")
    log_lines.append(f"Reference name: {reference_name}")
    log_lines.append(f"Normalization: {normalize_enabled} mode={normalize_mode}")
    log_lines.append(f"PSF match: {psf_match_enabled}")
    log_lines.append("")

    ref_data, ref_hdr = load_2d_and_header(reference_path)

    total_steps = max(1, len(items) + 5)
    step = 0

    def tick(msg):
        nonlocal step
        step += 1
        if progress_cb:
            progress_cb(step, total_steps, msg)
        if cancel_cb and cancel_cb():
            raise RuntimeError("Cancelled by user.")

    tick("Loaded reference")

    reproj_data = {}
    footprints = {}

    for it in items:
        tick(f"Reprojecting {it['name']}")
        mov_data, mov_hdr = load_2d_and_header(it["path"])
        r, fp = reproject_interp((mov_data, mov_hdr), ref_hdr, shape_out=ref_data.shape)
        reproj_data[it["name"]] = np.array(r, dtype=np.float32)
        footprints[it["name"]] = np.array(fp, dtype=np.float32)

    tick("Computing common overlap")
    mask = np.isfinite(ref_data)
    for nm in reproj_data.keys():
        mask &= np.isfinite(reproj_data[nm]) & (footprints[nm] > 0)

    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("No common overlap found. Frames may not overlap.")

    y0, y1, x0, x1 = bbox
    log_lines.append(f"Crop bbox: y0={y0} y1={y1} x0={x0} x1={x1}")
    log_lines.append("")

    ref_crop, new_hdr = crop_with_wcs(ref_data, ref_hdr, bbox)

    outputs = {reference_name: ref_crop}
    fp_out = {}

    for nm, arr in reproj_data.items():
        outputs[nm] = arr[y0:y1, x0:x1]
        fp_out[nm] = footprints[nm][y0:y1, x0:x1]

    tick("Saving masks and footprints")
    common_mask_crop = mask[y0:y1, x0:x1].astype(np.uint8)
    common_mask_path = os.path.join(out_dir, "common_overlap_mask.fits")
    fits.PrimaryHDU(common_mask_crop, header=new_hdr).writeto(common_mask_path, overwrite=True)
    log_lines.append(f"Saved: {common_mask_path}")

    ref_fp = np.isfinite(ref_crop).astype(np.uint8)
    ref_fp_path = os.path.join(out_dir, f"footprint_{safe_filename(reference_name)}.fits")
    fits.PrimaryHDU(ref_fp, header=new_hdr).writeto(ref_fp_path, overwrite=True)
    log_lines.append(f"Saved: {ref_fp_path}")

    for nm, fp in fp_out.items():
        fp_u8 = (fp > 0).astype(np.uint8)
        p = os.path.join(out_dir, f"footprint_{safe_filename(nm)}.fits")
        fits.PrimaryHDU(fp_u8, header=new_hdr).writeto(p, overwrite=True)
        log_lines.append(f"Saved: {p}")

    if normalize_enabled:
        tick("Normalizing to reference")
        ref_arr = outputs[reference_name]
        nm_mask = common_mask_crop.astype(bool)
        for nm in list(outputs.keys()):
            if nm == reference_name:
                continue
            out_arr, s = normalize_to_reference(outputs[nm], ref_arr, nm_mask, mode=normalize_mode)
            outputs[nm] = out_arr.astype(np.float32)
            log_lines.append(f"Normalization scale {nm}: {float(s):.6g}")
        log_lines.append("")

    if psf_match_enabled:
        tick("PSF matching to worst")
        names_order = list(outputs.keys())
        matched, notes = psf_match_to_worst(outputs, names_order)
        outputs = {k: matched[k].astype(np.float32) for k in names_order}
        for n in notes:
            log_lines.append(n)
        log_lines.append("")

    tick("Writing FITS outputs")
    saved = []
    for nm, arr in outputs.items():
        out_path = os.path.join(out_dir, f"{safe_filename(nm)}.fits")
        fits.PrimaryHDU(arr, header=new_hdr).writeto(out_path, overwrite=True)
        cov = float(np.isfinite(arr).mean() * 100.0)
        saved.append((nm, out_path, cov))
        log_lines.append(f"Saved: {out_path}  coverage={cov:.2f}%")

    log_path = write_log(out_dir, log_lines)
    return out_dir, saved, log_path


# =========================
# JWST community destripe tool (rate/cal) - separate window
# =========================

def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 1000:
        return float(np.nanstd(x)) if x.size else 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad if mad > 0 else float(np.nanstd(x))


def read_sci_and_dq(hdul: fits.HDUList):
    if "SCI" in hdul and hdul["SCI"].data is not None:
        data = hdul["SCI"].data
        if data.ndim > 2:
            data = data[0]
        dq = None
        if "DQ" in hdul and hdul["DQ"].data is not None:
            dq = hdul["DQ"].data
            if dq.ndim > 2:
                dq = dq[0]
        return np.array(data, dtype=np.float32), (np.array(dq) if dq is not None else None), "SCI"

    data = hdul[0].data
    if data is None:
        raise RuntimeError("No SCI extension and no primary image data.")
    if data.ndim > 2:
        data = data[0]
    return np.array(data, dtype=np.float32), None, "PRIMARY"


def build_source_mask(data: np.ndarray, nsigma: float, npixels: int, dilate_iter: int) -> np.ndarray:
    if not HAVE_DESTRIPE:
        raise RuntimeError("Missing photutils/astropy convolution dependencies for destripe tool.")

    med = np.nanmedian(data[np.isfinite(data)])
    tmp = np.nan_to_num(data, nan=med)

    kernel = make_2dgaussian_kernel(2.0, size=5)
    sm = convolve(tmp, kernel)

    sig = robust_sigma(sm)
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.nanstd(sm)) if np.isfinite(np.nanstd(sm)) else 1.0

    thresh = np.nanmedian(sm) + nsigma * sig
    segm = detect_sources(sm, thresh, npixels=npixels)

    mask = np.zeros(data.shape, dtype=bool)
    if segm is not None:
        mask |= (segm.data > 0)

    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)

    return mask


def destripe_rows(data: np.ndarray, mask: np.ndarray, split_amps: bool) -> np.ndarray:
    h, w = data.shape
    corrected = data.copy()

    if split_amps:
        edges = np.linspace(0, w, 5, dtype=int)  # 4 slices across x
        for ai in range(4):
            x0, x1 = int(edges[ai]), int(edges[ai + 1])
            for y in range(h):
                row = corrected[y, x0:x1]
                m = mask[y, x0:x1]
                good = row[~m & np.isfinite(row)]
                med = np.median(good) if good.size > 20 else 0.0
                corrected[y, x0:x1] = row - med
        return corrected

    for y in range(h):
        row = corrected[y, :]
        m = mask[y, :]
        good = row[~m & np.isfinite(row)]
        med = np.median(good) if good.size > 20 else 0.0
        corrected[y, :] = row - med
    return corrected


def process_one_destripe_file(in_path: str, out_dir: str,
                              nsigma: float, npixels: int, dilate_iter: int,
                              use_dq: bool, split_amps: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)

    with fits.open(in_path, memmap=False) as hdul:
        data, dq, where = read_sci_and_dq(hdul)

        mask = build_source_mask(data, nsigma=nsigma, npixels=npixels, dilate_iter=dilate_iter)
        if use_dq and dq is not None:
            mask |= (dq != 0)

        corrected = destripe_rows(data, mask, split_amps=split_amps)

        if where == "SCI":
            sci = hdul["SCI"].data
            if sci.ndim > 2:
                sci = np.array(sci, copy=True)
                sci[0] = corrected
                hdul["SCI"].data = sci
            else:
                hdul["SCI"].data = corrected
        else:
            prim = hdul[0].data
            if prim.ndim > 2:
                prim = np.array(prim, copy=True)
                prim[0] = corrected
                hdul[0].data = prim
            else:
                hdul[0].data = corrected

        hdul[0].header["HISTORY"] = (
            f"Destripe (community): nsigma={nsigma} npixels={npixels} "
            f"dilate={dilate_iter} use_dq={use_dq} split_amps={split_amps}"
        )

        base = os.path.basename(in_path)
        if base.lower().endswith(".fits"):
            base = base[:-5]
        out_path = os.path.join(out_dir, base + "_destripe.fits")
        hdul.writeto(out_path, overwrite=True)

    return out_path


class DestripeWindow(tk.Toplevel):
    def __init__(self, master: "App"):
        super().__init__(master)
        self.master_app = master
        self.title("JWST Destripe (rate/cal) - community method")
        self.geometry("900x560")
        self.resizable(True, True)

        self.files = []
        self.cancel_requested = False

        self.nsigma = tk.StringVar(value="2.5")
        self.npixels = tk.StringVar(value="10")
        self.dilate_iter = tk.StringVar(value="3")
        self.use_dq = tk.BooleanVar(value=True)
        self.split_amps = tk.BooleanVar(value=False)

        self.out_dir = tk.StringVar(value=os.path.join(os.getcwd(), "jwst_destripe_output"))

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Add *_rate.fits / *_cal.fits", command=self.add_files).pack(side="left")
        ttk.Button(top, text="Clear", command=self.clear).pack(side="left", padx=8)

        out = ttk.Frame(self, padding=(10, 0, 10, 10))
        out.pack(fill="x")
        ttk.Label(out, text="Output folder:").pack(side="left")
        ttk.Entry(out, textvariable=self.out_dir, width=72).pack(side="left", padx=8)
        ttk.Button(out, text="Browse", command=self.pick_out_dir).pack(side="left")

        params = ttk.LabelFrame(self, text="Mask + Destripe Parameters", padding=10)
        params.pack(fill="x", padx=10, pady=6)

        r1 = ttk.Frame(params)
        r1.pack(fill="x")
        ttk.Label(r1, text="nsigma (detect):").pack(side="left")
        ttk.Entry(r1, textvariable=self.nsigma, width=8).pack(side="left", padx=6)

        ttk.Label(r1, text="npixels:").pack(side="left", padx=(18, 0))
        ttk.Entry(r1, textvariable=self.npixels, width=8).pack(side="left", padx=6)

        ttk.Label(r1, text="dilate:").pack(side="left", padx=(18, 0))
        ttk.Entry(r1, textvariable=self.dilate_iter, width=8).pack(side="left", padx=6)

        r2 = ttk.Frame(params)
        r2.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(r2, text="Mask DQ!=0 pixels (if DQ exists)", variable=self.use_dq).pack(side="left")
        ttk.Checkbutton(r2, text="Split into 4 amp-like regions (generic)", variable=self.split_amps).pack(side="left", padx=18)

        mid = ttk.Frame(self, padding=10)
        mid.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(mid)
        self.listbox.pack(side="left", fill="both", expand=True)

        scr = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        scr.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=scr.set)

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")

        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", length=420)
        self.progress.pack(side="left")

        self.status = ttk.Label(bottom, text="Idle")
        self.status.pack(side="left", padx=12)

        ttk.Button(bottom, text="Run Destripe", command=self.run).pack(side="right")
        ttk.Button(bottom, text="Cancel", command=self.cancel).pack(side="right", padx=8)

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select JWST *_rate.fits or *_cal.fits",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            base = os.path.basename(p).lower()
            if "_i2d" in base or "_drz" in base or "_drc" in base:
                messagebox.showwarning("Wrong product", f"That looks drizzled:\n{p}\n\nUse *_rate.fits or *_cal.fits.")
                continue
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", p)

    def clear(self):
        self.files = []
        self.listbox.delete(0, "end")

    def pick_out_dir(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.out_dir.set(p)

    def cancel(self):
        self.cancel_requested = True
        self.status.config(text="Cancel requested...")

    def _is_cancelled(self):
        return bool(self.cancel_requested)

    def run(self):
        if not HAVE_DESTRIPE:
            messagebox.showerror(
                "Missing dependencies",
                "Destripe window needs photutils.\n\n"
                "Install:\n"
                "  pip install photutils\n\n"
                f"Import error:\n{DESTRIPE_IMPORT_ERR}"
            )
            return

        if not self.files:
            messagebox.showerror("No files", "Add at least one *_rate.fits or *_cal.fits file.")
            return

        out_dir = self.out_dir.get().strip()
        if not out_dir:
            messagebox.showerror("Output", "Pick an output folder.")
            return

        try:
            nsigma = float(self.nsigma.get().strip())
            npixels = int(float(self.npixels.get().strip()))
            dilate_iter = int(float(self.dilate_iter.get().strip()))
        except Exception:
            messagebox.showerror("Invalid parameters", "nsigma must be float; npixels/dilate must be integers.")
            return

        use_dq = bool(self.use_dq.get())
        split_amps = bool(self.split_amps.get())

        self.cancel_requested = False
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.files)
        self.status.config(text="Starting...")

        prog_q = queue.Queue()

        def worker():
            try:
                print("\n----- DESTRIPE RUN START -----", flush=True)
                print(f"Files: {len(self.files)}", flush=True)
                print(f"Output: {out_dir}", flush=True)
                print(f"nsigma={nsigma} npixels={npixels} dilate={dilate_iter} use_dq={use_dq} split_amps={split_amps}\n",
                      flush=True)

                saved = []
                for i, fp in enumerate(self.files, start=1):
                    if self._is_cancelled():
                        raise RuntimeError("Cancelled by user.")
                    print(f"[{i}/{len(self.files)}] Processing: {fp}", flush=True)
                    outp = process_one_destripe_file(fp, out_dir, nsigma, npixels, dilate_iter, use_dq, split_amps)
                    saved.append(outp)
                    prog_q.put(("PROG", i, os.path.basename(fp)))

                prog_q.put(("DONE", saved))
            except Exception:
                prog_q.put(("FAIL", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                while True:
                    item = prog_q.get_nowait()
                    if item[0] == "PROG":
                        _, i, name = item
                        self.progress["value"] = i
                        self.status.config(text=f"Saved {i}/{len(self.files)}: {name}")

                    elif item[0] == "DONE":
                        _, saved = item
                        self.status.config(text="Done")
                        print("----- DESTRIPE RUN COMPLETE -----\n", flush=True)
                        messagebox.showinfo(
                            "Done",
                            "Saved files:\n\n" + "\n".join(saved[:25]) + ("\n\n(+ more)" if len(saved) > 25 else "")
                        )
                        return

                    elif item[0] == "FAIL":
                        _, tb = item
                        self.status.config(text="Failed")
                        print("\n----- DESTRIPE RUN FAILED -----\n", flush=True)
                        print(tb, flush=True)
                        messagebox.showerror("Failed", "Destripe failed. See Live Logs window.")
                        return
            except queue.Empty:
                pass
            self.after(120, poll)

        poll()


# =========================
# Always-on separate Log Window
# =========================

class TextRedirector:
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def write(self, s: str):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class LogWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Live Logs")
        self.geometry("900x520")
        self.resizable(True, True)

        self.q = queue.Queue()

        self.text = tk.Text(self, wrap="word")
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)

        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.text.insert("end", "Live log window started.\n\n")
        self.text.see("end")

        sys.stdout = TextRedirector(self.q)
        sys.stderr = TextRedirector(self.q)

        self.after(50, self._drain)

    def _drain(self):
        try:
            while True:
                s = self.q.get_nowait()
                self.text.insert("end", s)
                self.text.see("end")
        except queue.Empty:
            pass
        self.after(50, self._drain)


# =========================
# Main UI
# =========================

class FilterRow:
    def __init__(self, parent, idx: int, on_browse, on_name_hint):
        self.idx = idx
        self.path = tk.StringVar()
        self.name = tk.StringVar()

        self.frame = ttk.Frame(parent)

        ttk.Label(self.frame, text=f"Filter {idx}:").grid(row=0, column=0, sticky="w", padx=6, pady=3)

        ttk.Entry(self.frame, textvariable=self.path, width=55).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Button(self.frame, text="Browse", command=lambda: on_browse(self)).grid(row=0, column=2, sticky="w", padx=6, pady=3)

        ttk.Label(self.frame, text="Name:").grid(row=0, column=3, sticky="w", padx=6, pady=3)
        ttk.Entry(self.frame, textvariable=self.name, width=22).grid(row=0, column=4, sticky="w", padx=6, pady=3)

        ttk.Button(self.frame, text="Auto-name", command=lambda: on_name_hint(self)).grid(row=0, column=5, sticky="w", padx=6, pady=3)

    def grid(self, row: int):
        self.frame.grid(row=row, column=0, columnspan=3, sticky="w")

    def destroy(self):
        self.frame.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FITS Align + Common Crop (Multi-filter)")
        self.geometry("1100x780")
        self.resizable(True, True)

        self.log_window = LogWindow(self)
        print("GUI started. Live logs are active.\n", flush=True)

        self.reference_path = tk.StringVar()
        self.reference_name = tk.StringVar(value="reference")
        self.output_dir = tk.StringVar()

        self.num_filters = tk.IntVar(value=3)
        self.rows = []

        self.normalize_enabled = tk.BooleanVar(value=True)
        self.normalize_mode = tk.StringVar(value="median")
        self.psf_match_enabled = tk.BooleanVar(value=False)

        self.cancel_requested = False
        self.preset = tk.StringVar(value="Custom")

        self._build()

    def _build(self):
        # Menu
        menubar = tk.Menu(self)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="JWST Destripe (rate/cal)...", command=self.open_destripe_window)
        menubar.add_cascade(label="Tools", menu=tools)
        self.config(menu=menubar)

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(root, text="Reference", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)

        ref_frame = ttk.Frame(root)
        ref_frame.grid(row=1, column=0, columnspan=4, sticky="w")

        ttk.Label(ref_frame, text="Reference FITS:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(ref_frame, textvariable=self.reference_path, width=70).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(ref_frame, text="Browse", command=self.pick_reference).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        ttk.Label(ref_frame, text="Output name:").grid(row=0, column=3, sticky="w", padx=6, pady=4)
        ttk.Entry(ref_frame, textvariable=self.reference_name, width=22).grid(row=0, column=4, sticky="w", padx=6, pady=4)
        ttk.Button(ref_frame, text="Auto-name", command=self.auto_name_reference).grid(row=0, column=5, sticky="w", padx=6, pady=4)

        ttk.Button(ref_frame, text="Auto-pick reference", command=self.auto_pick_reference).grid(row=0, column=6, sticky="w", padx=6, pady=4)

        ttk.Separator(root).grid(row=2, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Label(root, text="Moving filters (blank rows are ignored)", font=("Segoe UI", 12, "bold")).grid(
            row=3, column=0, sticky="w", padx=6, pady=6
        )

        control = ttk.Frame(root)
        control.grid(row=4, column=0, columnspan=4, sticky="w")

        ttk.Label(control, text="Preset:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(
            control,
            values=["Custom", "NIRCam 3-filter RGB", "NIRCam 8-filter luminance workflow", "MIRI 5-filter set"],
            textvariable=self.preset,
            width=34,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(control, text="Apply preset", command=self.apply_preset).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        ttk.Label(control, text="How many rows?").grid(row=1, column=0, sticky="w", padx=6, pady=4)

        options = list(range(1, 61))
        self.combo = ttk.Combobox(control, values=options, width=6, state="readonly")
        self.combo.set(str(self.num_filters.get()))
        self.combo.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(control, text="Apply", command=self.apply_num_filters).grid(row=1, column=2, sticky="w", padx=6, pady=4)

        ttk.Separator(root).grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)

        self.canvas = tk.Canvas(root, height=420)
        self.scroll = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.scroll.grid(row=6, column=3, sticky="ns")
        self.canvas.grid(row=6, column=0, columnspan=3, sticky="nsew")

        root.grid_rowconfigure(6, weight=1)
        root.grid_columnconfigure(2, weight=1)

        self.filters_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.filters_container, anchor="nw")

        self.filters_container.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._rebuild_filter_rows(self.num_filters.get())

        ttk.Separator(root).grid(row=7, column=0, columnspan=4, sticky="ew", pady=10)

        opt = ttk.Frame(root)
        opt.grid(row=8, column=0, columnspan=4, sticky="w")

        ttk.Checkbutton(opt, text="Normalize to reference", variable=self.normalize_enabled).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(opt, text="Mode:").grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Combobox(opt, values=["median", "p95", "background"], textvariable=self.normalize_mode, width=12, state="readonly").grid(row=0, column=2, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(opt, text="PSF match to worst (approx)", variable=self.psf_match_enabled).grid(row=0, column=3, sticky="w", padx=16, pady=4)

        out_frame = ttk.Frame(root)
        out_frame.grid(row=9, column=0, columnspan=4, sticky="w")

        ttk.Label(out_frame, text="Output directory:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(out_frame, textvariable=self.output_dir, width=70).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(out_frame, text="Browse", command=self.pick_output_dir).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        run_frame = ttk.Frame(root)
        run_frame.grid(row=10, column=0, columnspan=4, sticky="ew", padx=6, pady=10)
        run_frame.grid_columnconfigure(1, weight=1)

        self.progress = ttk.Progressbar(run_frame, orient="horizontal", length=420, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="w", padx=6, pady=4)

        self.status = ttk.Label(run_frame, text="Idle")
        self.status.grid(row=0, column=1, sticky="w", padx=10, pady=4)

        ttk.Button(run_frame, text="JWST Destripe Window", command=self.open_destripe_window).grid(row=0, column=2, sticky="e", padx=6, pady=4)
        ttk.Button(run_frame, text="Run (Align + Common Crop)", command=self.run).grid(row=0, column=3, sticky="e", padx=6, pady=4)
        ttk.Button(run_frame, text="Cancel", command=self.cancel).grid(row=0, column=4, sticky="e", padx=6, pady=4)

    def open_destripe_window(self):
        if not HAVE_DESTRIPE:
            messagebox.showerror(
                "Missing dependencies",
                "Destripe window needs photutils.\n\n"
                "Install:\n"
                "  pip install photutils\n\n"
                f"Import error:\n{DESTRIPE_IMPORT_ERR}"
            )
            return
        DestripeWindow(self)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    # ---------- Presets ----------

    def apply_preset(self):
        p = self.preset.get()
        if p == "NIRCam 3-filter RGB":
            self.num_filters.set(3)
            self.combo.set("3")
            self._rebuild_filter_rows(3)
            self.normalize_enabled.set(True)
            self.normalize_mode.set("median")
            self.psf_match_enabled.set(False)
            defaults = ["nircam_F115W", "nircam_F210M", "nircam_F356W"]
            for i, r in enumerate(self.rows):
                r.name.set(defaults[i])

        elif p == "NIRCam 8-filter luminance workflow":
            self.num_filters.set(8)
            self.combo.set("8")
            self._rebuild_filter_rows(8)
            self.normalize_enabled.set(True)
            self.normalize_mode.set("median")
            self.psf_match_enabled.set(True)
            defaults = ["nircam_F115W", "nircam_F140M", "nircam_F150W", "nircam_F210M",
                        "nircam_F277W", "nircam_F335M", "nircam_F356W", "nircam_F480M"]
            for i, r in enumerate(self.rows):
                r.name.set(defaults[i])

        elif p == "MIRI 5-filter set":
            self.num_filters.set(5)
            self.combo.set("5")
            self._rebuild_filter_rows(5)
            self.normalize_enabled.set(True)
            self.normalize_mode.set("median")
            self.psf_match_enabled.set(False)
            defaults = ["miri_F770W", "miri_F1000W", "miri_F1130W", "miri_F1500W", "miri_F2550W"]
            for i, r in enumerate(self.rows):
                r.name.set(defaults[i])

    # ---------- Browse + auto-name ----------

    def pick_reference(self):
        p = filedialog.askopenfilename(
            title="Select reference FITS",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
        )
        if p:
            self.reference_path.set(p)
            if self.reference_name.get().strip().lower() in ("reference", "ref", ""):
                self.reference_name.set(suggest_name_from_fits(p))

    def auto_name_reference(self):
        p = self.reference_path.get().strip()
        if not p or not os.path.exists(p):
            messagebox.showerror("No reference", "Pick a reference FITS first.")
            return
        self.reference_name.set(suggest_name_from_fits(p))

    def pick_output_dir(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.output_dir.set(p)

    def browse_row(self, row: FilterRow):
        p = filedialog.askopenfilename(
            title=f"Select FITS for Filter {row.idx}",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
        )
        if p:
            row.path.set(p)
            cur = row.name.get().strip().lower()
            if cur in ("", f"filter_{row.idx}", f"filter{row.idx}"):
                row.name.set(suggest_name_from_fits(p))

    def auto_name_row(self, row: FilterRow):
        p = row.path.get().strip()
        if not p or not os.path.exists(p):
            messagebox.showerror("No file", f"Pick a FITS file for Filter {row.idx} first.")
            return
        row.name.set(suggest_name_from_fits(p))

    # ---------- Filters count (preserve selections) ----------

    def apply_num_filters(self):
        try:
            n = int(self.combo.get().strip())
        except Exception:
            messagebox.showerror("Invalid number", "Pick a valid number of rows.")
            return
        if n < 1:
            messagebox.showerror("Invalid number", "Need at least 1 row.")
            return
        self.num_filters.set(n)
        self._rebuild_filter_rows(n)

    def _rebuild_filter_rows(self, n: int):
        old = [(r.path.get(), r.name.get()) for r in self.rows]

        for r in self.rows:
            r.destroy()
        self.rows = []

        for i in range(1, n + 1):
            r = FilterRow(self.filters_container, i, self.browse_row, self.auto_name_row)
            r.grid(row=i - 1)

            if i - 1 < len(old):
                p, nm = old[i - 1]
                r.path.set(p)
                r.name.set(nm)
            else:
                r.path.set("")
                r.name.set(f"filter_{i}")

            self.rows.append(r)

    # ---------- Auto-pick reference (blank rows allowed) ----------

    def auto_pick_reference(self):
        paths = []
        ref_path = self.reference_path.get().strip()
        if ref_path and os.path.exists(ref_path):
            paths.append(ref_path)

        for r in self.rows:
            p = r.path.get().strip()
            if p and os.path.exists(p):
                paths.append(p)

        if not paths:
            messagebox.showerror("No files", "Select some FITS files first.")
            return

        best, _table = choose_best_reference(paths)

        if ref_path and os.path.exists(ref_path) and os.path.abspath(best) == os.path.abspath(ref_path):
            messagebox.showinfo("Auto-pick reference", "Current reference already looks best.")
            return

        if not ref_path or not os.path.exists(ref_path):
            self.reference_path.set(best)
            self.reference_name.set(suggest_name_from_fits(best))
            for r in self.rows:
                rp = r.path.get().strip()
                if rp and os.path.abspath(rp) == os.path.abspath(best):
                    r.path.set("")
                    r.name.set(f"filter_{r.idx}")
                    break
            messagebox.showinfo("Auto-pick reference", "Reference set. The promoted row is now blank (ignored).")
            return

        old_ref = ref_path
        self.reference_path.set(best)
        self.reference_name.set(suggest_name_from_fits(best))

        swapped = False
        for r in self.rows:
            rp = r.path.get().strip()
            if rp and os.path.abspath(rp) == os.path.abspath(best):
                r.path.set(old_ref)
                r.name.set(suggest_name_from_fits(old_ref))
                swapped = True
                break

        if swapped:
            messagebox.showinfo("Auto-pick reference", "Reference swapped with one of your selected filters.")
        else:
            messagebox.showinfo("Auto-pick reference", "Reference changed, but could not find a matching row to swap.")

    # ---------- Cancel + threaded run ----------

    def cancel(self):
        self.cancel_requested = True
        self.status.config(text="Cancel requested...")

    def _is_cancelled(self):
        return bool(self.cancel_requested)

    def run(self):
        self.cancel_requested = False
        self.status.config(text="Starting...")
        self.progress["value"] = 0

        ref = self.reference_path.get().strip()
        ref_name = self.reference_name.get().strip()
        out_dir = self.output_dir.get().strip()

        if not ref or not os.path.exists(ref):
            messagebox.showerror("Missing reference", "Pick a valid reference FITS file.")
            return
        if not ref_name:
            messagebox.showerror("Missing name", "Give the reference an output name.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Missing output directory", "Pick a valid output directory.")
            return

        items = []
        names_seen = set([safe_filename(ref_name).lower()])

        for r in self.rows:
            p = r.path.get().strip()
            nm = r.name.get().strip()

            if not p:
                continue

            if not os.path.exists(p):
                messagebox.showerror("Invalid file", f"Filter {r.idx} has an invalid FITS file:\n{p}")
                return

            if not nm:
                messagebox.showerror("Missing name", f"Filter {r.idx} needs an output name.")
                return

            nm_safe = safe_filename(nm).lower()
            if nm_safe in names_seen:
                messagebox.showerror("Duplicate name", f"Name '{nm}' is duplicated. Names must be unique.")
                return
            names_seen.add(nm_safe)

            items.append({"path": p, "name": nm})

        if len(items) == 0:
            messagebox.showerror("No moving filters", "All rows are blank. Pick at least one moving filter.")
            return

        prog_q = queue.Queue()

        def progress_cb(step, total, msg):
            print(msg, flush=True)
            prog_q.put(("PROG", step, total, msg))

        def worker():
            try:
                print("----- RUN START -----", flush=True)
                print(f"Reference: {ref_name}  ({ref})", flush=True)
                print(f"Moving filters used: {len(items)}", flush=True)
                print(f"Output dir: {out_dir}", flush=True)
                print(f"Normalize: {bool(self.normalize_enabled.get())} mode={self.normalize_mode.get()}", flush=True)
                print(f"PSF match: {bool(self.psf_match_enabled.get())}", flush=True)
                print("", flush=True)

                out_folder, saved, log_path = align_crop_many(
                    reference_path=ref,
                    reference_name=ref_name,
                    items=items,
                    out_base_dir=out_dir,
                    normalize_enabled=bool(self.normalize_enabled.get()),
                    normalize_mode=str(self.normalize_mode.get()),
                    psf_match_enabled=bool(self.psf_match_enabled.get()),
                    progress_cb=progress_cb,
                    cancel_cb=self._is_cancelled,
                )
                prog_q.put(("DONE", out_folder, saved, log_path))
            except Exception:
                prog_q.put(("FAIL", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            try:
                while True:
                    item = prog_q.get_nowait()

                    if item[0] == "PROG":
                        _, step, total, msg = item
                        self.progress["maximum"] = total
                        self.progress["value"] = step
                        self.status.config(text=msg)

                    elif item[0] == "DONE":
                        _, out_folder, saved, log_path = item
                        self.status.config(text="Done")
                        print("\n----- RUN COMPLETE -----\n", flush=True)

                        lines = [f"Saved in:\n{out_folder}\n", f"Log file:\n{log_path}\n"]
                        for nm, path, cov in saved:
                            lines.append(f"{nm} -> {os.path.basename(path)} | coverage: {cov:.2f}%")
                        lines.append("")
                        lines.append("Also saved:")
                        lines.append("common_overlap_mask.fits")
                        lines.append("footprint_<name>.fits for each output")

                        messagebox.showinfo("Done", "\n".join(lines))
                        return

                    elif item[0] == "FAIL":
                        _, tb = item
                        self.status.config(text="Failed")
                        print("\n----- RUN FAILED -----\n", flush=True)
                        print(tb, flush=True)
                        messagebox.showerror("Failed", "Processing failed. See the Live Logs window for details.")
                        return

            except queue.Empty:
                pass

            self.after(100, poll)

        poll()


if __name__ == "__main__":
    App().mainloop()