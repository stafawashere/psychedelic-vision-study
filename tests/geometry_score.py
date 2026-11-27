from __future__ import annotations
import sys
import os
import numpy as np
from PIL import Image

LATTICE_WL_LO, LATTICE_WL_HI = 18.0, 32.0  # established input-independent lattice sits at ~24px


def _gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L")).astype(np.float32)


def _radius(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def periodicity(im: Image.Image) -> tuple[float, float, int]:
    # mid-band peak prominence over the detrended log radial spectrum; + wavelength(px), peak radius.
    g = _gray(im)
    p = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    r = _radius(p.shape).astype(int)
    rad = np.bincount(r.ravel(), p.ravel()) / np.bincount(r.ravel())
    n = len(rad)
    lo, hi = int(n * 0.06), int(n * 0.45)
    band = np.log(rad[lo:hi] + 1e-9)
    xs = np.arange(len(band))
    resid = band - np.polyval(np.polyfit(xs, band, 1), xs)
    pk = lo + int(np.argmax(resid))
    return float(resid.max()), p.shape[0] / max(pk, 1), pk


def band_sharpness(im: Image.Image, pk: int) -> float:
    # variance of the image bandpassed to [0.7, 2.3]*peak — lattice fundamental through 1st harmonic.
    # crisp cells have strong harmonics here; soft mosaics leak energy below the band.
    g = _gray(im)
    F = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    r = _radius(F.shape)
    mask = (r >= pk * 0.7) & (r <= pk * 2.3)
    band_img = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    return float(band_img.var())


def acutance(im: Image.Image) -> float:
    # 90th-percentile gradient magnitude / median — edge contrast. crisp sparse walls -> high.
    g = _gray(im)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    # floor the denominator at 1.0 (on the 0-255 gradient scale): a near-flat image has a
    # ~zero median gradient, and p90/~0 would blow up — floor keeps it ~0 (no edges) instead.
    return float(np.percentile(mag, 90) / max(float(np.median(mag)), 1.0))


def clip_frac(im: Image.Image) -> float:
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    return float((a <= 2).mean() + (a >= 253).mean())


def clarity(im: Image.Image) -> dict:
    prom, wl, pk = periodicity(im)
    bs = band_sharpness(im, pk)
    ac = acutance(im)
    c = clip_frac(im)
    in_band = LATTICE_WL_LO <= wl <= LATTICE_WL_HI
    gate = max(0.0, 1.0 - 5.0 * c) * (1.0 if in_band else 0.0)
    return {
        "clarity": prom * (bs ** 0.5) * ac * gate,
        "periodicity": prom,
        "band_sharp": bs,
        "acutance": ac,
        "wavelength": wl,
        "clip": c,
        "in_band": in_band,
    }


def _iter_images(paths):
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    yield os.path.join(p, f)
        else:
            yield p


def main(argv: list[str]) -> None:
    paths = argv or ["."]
    print(f"{'image':40s} {'clarity':>9s} {'period':>7s} {'bandSh':>8s} {'acut':>6s} {'wl_px':>6s} {'clip%':>6s} {'band':>5s}")
    for path in _iter_images(paths):
        try:
            im = Image.open(path)
        except Exception:
            continue
        m = clarity(im)
        name = os.path.relpath(path)
        if len(name) > 40:
            name = "…" + name[-39:]
        print(f"{name:40s} {m['clarity']:9.2f} {m['periodicity']:7.2f} {m['band_sharp']:8.1f} "
              f"{m['acutance']:6.2f} {m['wavelength']:6.1f} {m['clip']*100:6.1f} {'in' if m['in_band'] else 'OUT':>5s}")


if __name__ == "__main__":
    main(sys.argv[1:])
