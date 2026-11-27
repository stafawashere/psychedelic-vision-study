"""Per-frame measurements.

The point of measuring is so the hypothesis can fail.

Four instruments, all read-only, all shipping with controls (`docs/development-plan.md` P0):

  clarity_ungated       the lattice wavelength as a datum, not a hard filter
  form_constant_score   M-1, is this a Kluver planform rather than merely periodic
  detail_preservation   M-2, was scene detail preserved or destroyed and resynthesised
  hold_dynamics         M-3, does a hold segment have dynamics or just parameter drift
  arc_closure           M-4, did the arc come back, measured so a texture scene cannot fake it

`scene_fidelity` is kept because every archived run is scored with it, but it is blind on texture
(granite 0.473 against hedge 0.453) and M-4 supersedes it for closure claims.

`tests/geometry_score.py` is deliberately left alone — its gated `clarity` stays the arm-A ranking
proxy so earlier runs remain comparable. These are wider instruments beside it, not replacements.

The ordering experiment is closed (see the depth sweep), so no face detector is built here.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

from tests.geometry_score import periodicity, band_sharpness, acutance, clip_frac, LATTICE_WL_LO, LATTICE_WL_HI
from tests.probe_seed_positions import offset_from

# below the ~24px structure, above the 8px VAE grid — measures layout, not texture or grid artifact
FIDELITY_DOWNSAMPLE = 4

# M-1 spectral band, in fft radius units on a 512px frame: k=8 is 64px, k=140 is 3.7px
BAND_K_LO, BAND_K_HI = 8, 140

# M-1 log-polar readout grid. r_min is a foveal floor: the centre is singular under log r
LOGPOLAR_RHO, LOGPOLAR_THETA, LOGPOLAR_R_MIN = 128, 256, 8.0
# a mode counts as pure-radial / pure-angular only within this many bins of the axis
LOGPOLAR_AXIS_TOL = 1

# M-2 band floor: 1/8 of Nyquist. scene_fidelity's 4x box downsample discards everything above
# 1/4 Nyquist, so this is the band it is structurally blind to
HF_NYQUIST_FRACTION = 0.125

# M-4 closure thresholds, pre-registered in docs/development-plan.md section 2 before any new run
CLOSURE_RMSE_MAX = 0.05
CLOSURE_DETAIL_MIN = 0.8
CLOSURE_CONTROL_MAX = 0.3

# M-3 verdict thresholds. below the churn floor consecutive frames are unrelated images
DYNAMICS_CONTINUITY_MIN = 0.7
DYNAMICS_CHURN_MAX = 0.5

CONTROL_SIZE = 512
CONTROL_WAVELENGTH = 24


def _gray01(im) -> np.ndarray:
    if isinstance(im, np.ndarray):
        return im.astype(np.float64)

    return np.asarray(im.convert("L")).astype(np.float64) / 255.0


def _radius_int(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    return np.sqrt((grid_y - height // 2) ** 2 + (grid_x - width // 2) ** 2).astype(int)
    

def _windowed(img: np.ndarray) -> np.ndarray:
    # hann both axes so the fft does not read the frame border as a hard edge
    vertical = np.hanning(img.shape[0])[:, None]
    horizontal = np.hanning(img.shape[1])[None, :]
    return (img - img.mean()) * vertical * horizontal


def _radial_profile(img: np.ndarray) -> np.ndarray:
    power = np.fft.fftshift(np.abs(np.fft.fft2(_windowed(img))) ** 2)
    radius = _radius_int(img.shape)

    bins = min(img.shape[0] // 2, img.shape[1] // 2)
    summed = np.bincount(radius.ravel(), power.ravel(), minlength=bins + 1)[:bins]
    counts = np.bincount(radius.ravel(), minlength=bins + 1)[:bins]

    return summed / np.maximum(counts, 1)


def _radial_magnitude(spectrum: np.ndarray) -> np.ndarray:
    radius = _radius_int(spectrum.shape)
    magnitude = np.abs(spectrum)
    summed = np.bincount(radius.ravel(), magnitude.ravel())
    counts = np.bincount(radius.ravel())

    return summed / np.maximum(counts, 1)


def _whitening_filter(spectrum: np.ndarray, k_lo: float, k_hi: float) -> np.ndarray:
    """The magnitude trend to divide out, as a power law fitted in log-log over the band.

    An EMPIRICAL radial profile cannot be used: a near-monochromatic field has near-empty radii on
    either side of its annulus, and dividing by those amplifies sampling residue by thousands. That
    is what collapsed the rings control's radial fraction from 1.000 to 0.250. A power law over a
    bounded band has bounded dynamic range, and normalising it to unit geometric mean keeps the
    whitening a pure shape correction with no amplitude term.
    """
    profile = _radial_magnitude(spectrum)
    wavenumber = np.arange(len(profile))
    band = slice(int(max(k_lo, 1)), int(min(k_hi, len(profile))))

    slope, intercept = np.polyfit(np.log(wavenumber[band]), np.log(profile[band] + 1e-20), 1)
    trend = np.exp(intercept) * np.maximum(wavenumber, 1.0) ** slope

    return trend / np.exp(np.log(trend[band]).mean())


def _whitened_band(gray: np.ndarray, k_lo: float, k_hi: float | None = None,
                   trend: np.ndarray | None = None) -> np.ndarray:
    """Band-limited image with the 1/f trend divided out.

    Natural images are 1/f^a, so any correlation or power fraction computed on the raw band is
    dominated by its lowest frequency. Flattening the trend first makes every octave in the band
    weigh the same, which is what lets a blurred image separate from a contrast-shifted one.
    Pass `trend` to whiten one image by ANOTHER image's trend (M-2 whitens both by the reference).
    """
    spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    radius = _radius_int(gray.shape)
    ceiling = k_hi if k_hi is not None else min(gray.shape) // 2

    if trend is None:
        trend = _whitening_filter(spectrum, k_lo, ceiling)

    clamped = np.minimum(radius, len(trend) - 1)
    keep = radius >= k_lo
    if k_hi is not None:
        keep &= radius <= k_hi

    scale = np.where(keep, 1.0 / trend[clamped], 0.0)

    return np.fft.ifft2(np.fft.ifftshift(spectrum * scale)).real


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())

    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def whitened_peak(img: np.ndarray, k_lo: int = BAND_K_LO, k_hi: int = BAND_K_HI) -> tuple[float, float]:
    # natural images are 1/f^a, so a raw spectral max just finds dc. fit the trend in log-log and
    # look for a BUMP above it -- that is what a selected wavelength would look like.
    profile = _radial_profile(img)
    wavenumber = np.arange(len(profile))
    band = slice(k_lo, k_hi)

    log_k = np.log(wavenumber[band])
    log_p = np.log(profile[band] + 1e-20)
    slope, intercept = np.polyfit(log_k, log_p, 1)
    residual = log_p - (slope * log_k + intercept)

    peak = int(np.argmax(residual))
    return img.shape[0] / wavenumber[band][peak], float(residual[peak])


def angular_anisotropy(img: np.ndarray, sectors: int = 36, r_lo: int = 8, r_hi: int = 120) -> float:
    power = np.fft.fftshift(np.abs(np.fft.fft2(_windowed(img))) ** 2)
    height, width = img.shape
    center_y, center_x = height // 2, width // 2
    grid_y, grid_x = np.mgrid[0:height, 0:width]

    radius = np.sqrt((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2)
    angle = np.degrees(np.arctan2(grid_y - center_y, grid_x - center_x)) % 180
    band = (radius > r_lo) & (radius < r_hi)

    per_sector = np.array([
        power[band & (angle >= i * 180 / sectors) & (angle < (i + 1) * 180 / sectors)].mean()
        for i in range(sectors)
    ])

    return float(per_sector.max() / max(per_sector.min(), 1e-20))


def rotational_symmetry(img: np.ndarray, degrees: float, margin: int = 96) -> float:
    # crop the margin off both: rotation about the centre pulls undefined corners into view
    as_u8 = Image.fromarray((img * 255).astype(np.uint8))
    rotated = np.asarray(as_u8.rotate(degrees, resample=Image.BILINEAR), dtype=np.float64) / 255.0
    height, width = img.shape

    return _correlate(img[margin:height - margin, margin:width - margin],
                      rotated[margin:height - margin, margin:width - margin])


def _bilinear(img: np.ndarray, sample_y: np.ndarray, sample_x: np.ndarray) -> np.ndarray:
    height, width = img.shape
    y0 = np.clip(np.floor(sample_y).astype(int), 0, height - 2)
    x0 = np.clip(np.floor(sample_x).astype(int), 0, width - 2)
    fy = np.clip(sample_y - y0, 0.0, 1.0)
    fx = np.clip(sample_x - x0, 0.0, 1.0)

    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx

    return top * (1 - fy) + bottom * fy


def _log_polar(img: np.ndarray) -> np.ndarray:
    """Resample onto (log r, theta). MEASUREMENT ONLY — this is not the arm-B warp."""
    height, width = img.shape
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    r_max = min(center_y, center_x)

    rho = np.exp(np.linspace(np.log(LOGPOLAR_R_MIN), np.log(r_max), LOGPOLAR_RHO))
    theta = np.linspace(0.0, 2 * np.pi, LOGPOLAR_THETA, endpoint=False)

    sample_y = center_y + rho[:, None] * np.sin(theta)[None, :]
    sample_x = center_x + rho[:, None] * np.cos(theta)[None, :]

    return _bilinear(img, sample_y, sample_x)


def radial_angular_split(img: np.ndarray) -> tuple[float, float, float]:
    """How the whitened band power divides between pure-radial, pure-angular and mixed modes.

    In (log r, theta) a concentric ring pattern is theta-invariant, so all its power sits on the
    zero-angular-frequency axis; a starburst sits on the zero-radial axis; a log spiral is a single
    plane wave OFF both axes; a stationary texture like a plane-wave lattice is a chirp and spreads.
    Concentration separates the last two, since a spiral is off-axis AND sharp while a texture is
    off-axis and diffuse.
    """
    field = _log_polar(_whitened_band(img, BAND_K_LO, BAND_K_HI))
    # hann along rho only: theta wraps, so it has no border to taper
    field = (field - field.mean()) * np.hanning(field.shape[0])[:, None]

    power = np.abs(np.fft.fft2(field)) ** 2
    freq_rho = np.abs(np.fft.fftfreq(LOGPOLAR_RHO) * LOGPOLAR_RHO)[:, None]
    freq_theta = np.abs(np.fft.fftfreq(LOGPOLAR_THETA) * LOGPOLAR_THETA)[None, :]

    near_radial = freq_theta <= LOGPOLAR_AXIS_TOL
    near_angular = freq_rho <= LOGPOLAR_AXIS_TOL
    near_dc = near_radial & near_angular

    band = ~near_dc
    total = power[band].sum()
    if total <= 0:
        return 0.0, 0.0, 0.0

    radial = power[near_radial & band].sum() / total
    angular = power[near_angular & band].sum() / total
    concentration = power[band].max() / power[band].mean()

    return float(radial), float(angular), float(concentration)


def form_constant_score(im, predicted_wavelength: float | None = None) -> dict:
    """M-1. Is this a Kluver planform, not merely periodic?

    Periodicity without symmetry is a saturation texture. The symmetry columns and the
    radial/angular split are what separate the two, so they are reported together or not at all.
    `predicted_wavelength` is echoed back so E-3 can regress measured against predicted; it is a
    label, never an input to any other number here.
    """
    img = _gray01(im)
    wavelength, bump = whitened_peak(img)
    radial, angular, concentration = radial_angular_split(img)

    score = {
        "wavelength": wavelength,
        "bump": bump,
        "excess": float(np.exp(bump)),
        "aniso": angular_anisotropy(img),
        "mirror_lr": _correlate(img, img[:, ::-1]),
        "rot60": rotational_symmetry(img, 60),
        "rot90": rotational_symmetry(img, 90),
        "radial_frac": radial,
        "angular_frac": angular,
        "lp_concentration": concentration,
        "predicted_wavelength": predicted_wavelength,
        "wavelength_ratio": None,
    }
    if predicted_wavelength:
        score["wavelength_ratio"] = wavelength / predicted_wavelength

    return score


def form_constant_controls() -> list[tuple[str, np.ndarray]]:
    grid_y, grid_x = np.mgrid[0:CONTROL_SIZE, 0:CONTROL_SIZE].astype(np.float64)
    center = CONTROL_SIZE / 2
    period = CONTROL_WAVELENGTH

    lattice = (np.cos(2 * np.pi * grid_x / period)
               + np.cos(2 * np.pi * (grid_x * 0.5 + grid_y * 0.866) / period)
               + np.cos(2 * np.pi * (-grid_x * 0.5 + grid_y * 0.866) / period))
    lattice = (lattice - lattice.min()) / (lattice.max() - lattice.min())

    radius = np.sqrt((grid_x - center) ** 2 + (grid_y - center) ** 2)
    theta = np.arctan2(grid_y - center, grid_x - center)

    rings = 0.5 + 0.5 * np.cos(2 * np.pi * radius / period)
    spiral = 0.5 + 0.5 * np.cos(6 * theta + 2 * np.pi * np.log(radius + 2) / 0.35)

    return [("CONTROL hex lattice", lattice),
            ("CONTROL rings", rings),
            ("CONTROL log spiral", spiral),
            ("CONTROL flat gray", np.full((CONTROL_SIZE, CONTROL_SIZE), 0.5))]


def _orientation_field(band: np.ndarray) -> np.ndarray:
    # (gx + i*gy)^2 has magnitude |grad|^2 and DOUBLE the gradient angle, so it is a proper
    # orientation (mod 180) rather than a direction: a contrast inversion does not flip it
    gx = np.zeros_like(band)
    gy = np.zeros_like(band)
    gx[:, 1:-1] = band[:, 2:] - band[:, :-2]
    gy[1:-1, :] = band[2:, :] - band[:-2, :]

    return (gx + 1j * gy) ** 2


def detail_preservation(frame, reference) -> float:
    """M-2. Was high-frequency scene detail preserved, or destroyed and resynthesised?

    Rising `ds` costs detail because `add_noise` scales the latent by sqrt(alpha_bar) against white
    noise while the latent spectrum is red, so the loss lands in the fine band — exactly the band
    `scene_fidelity`'s 4x downsample throws away. Correlating gradient ORIENTATION rather than
    intensity means a contrast or colour shift does not read as detail loss. Both images are
    whitened by the reference's own radial trend so every octave in the band counts equally.
    """
    if isinstance(frame, Image.Image) and isinstance(reference, Image.Image):
        reference = reference.convert("L").resize(frame.size)

    subject = _gray01(frame)
    target = _gray01(reference)

    cutoff = HF_NYQUIST_FRACTION * min(target.shape) / 2.0
    spectrum = np.fft.fftshift(np.fft.fft2(target - target.mean()))
    trend = _whitening_filter(spectrum, cutoff, min(target.shape) // 2)

    field_target = _orientation_field(_whitened_band(target, cutoff, trend=trend))
    field_subject = _orientation_field(_whitened_band(subject, cutoff, trend=trend))

    denom = np.sqrt((np.abs(field_target) ** 2).sum() * (np.abs(field_subject) ** 2).sum())
    if denom < 1e-20:
        return 0.0

    return float(np.real(np.vdot(field_target, field_subject)) / denom)


def _frame_change(a: Image.Image, b: Image.Image) -> float:
    # rmse on the 0-255 grey scale, the same units as the 22.30 frozen / 56.20 churn poles
    ga = np.asarray(a.convert("L")).astype(np.float64)
    gb = np.asarray(b.convert("L")).astype(np.float64)

    return float(np.sqrt(((ga - gb) ** 2).mean()))


def hold_dynamics(frames: list) -> dict:
    """M-3. On a hold segment (D pinned, every knob constant), is there motion or only drift?

    Three numbers, because the change rate alone cannot tell them apart. A decaying rate means the
    loop is converging to a fixed point. A sustained rate with a high lag-1 peak correlation and a
    consistently signed offset is a travelling wave; the same with a zero offset is a standing
    pattern; a low peak correlation is churn, which is not dynamics at all.
    """
    if len(frames) < 3:
        raise ValueError("a hold segment needs at least 3 frames to have a slope")

    changes = [_frame_change(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    offsets = [offset_from(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]

    slope = float(np.polyfit(np.arange(len(changes)), changes, 1)[0])
    peak_corr = float(np.mean([o[2] for o in offsets]))
    shifts = [(o[0], o[1]) for o in offsets]

    moved = [s for s in shifts if s != (0, 0)]
    signs_y = {np.sign(s[0]) for s in moved if s[0] != 0}
    signs_x = {np.sign(s[1]) for s in moved if s[1] != 0}
    consistently_signed = len(moved) == len(shifts) and (len(signs_y) == 1 or len(signs_x) == 1)

    if peak_corr < DYNAMICS_CHURN_MAX:
        verdict = "churn"
    elif consistently_signed:
        verdict = "travelling"
    elif not moved:
        verdict = "standing"
    else:
        verdict = "mixed"

    return {
        "mean_change": float(np.mean(changes)),
        "first_change": changes[0],
        "last_change": changes[-1],
        "slope": slope,
        "regime": "decaying" if slope < 0 else "sustained",
        "peak_corr": peak_corr,
        "offsets": shifts,
        "verdict": verdict,
    }


def arc_closure(frame, reference) -> dict:
    """M-4. Did the arc come back? Two numbers, and closure needs both.

    `scene_fidelity` is blind on texture, which is how granite (0.473) scored above a hedge (0.453)
    and the blindness went uncaught. Layout RMSE plus M-2's fine band closes that hole, and the
    different-photo control belongs in the same table every time.
    """
    if isinstance(frame, Image.Image) and isinstance(reference, Image.Image):
        reference = reference.convert("RGB").resize(frame.size)

    a = np.asarray(frame.convert("RGB")).astype(np.float64) / 255.0
    b = np.asarray(reference.convert("RGB")).astype(np.float64) / 255.0
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    detail = detail_preservation(frame, reference)

    return {
        "rmse": rmse,
        "detail": detail,
        "closed": rmse <= CLOSURE_RMSE_MAX and detail >= CLOSURE_DETAIL_MIN,
    }


def clarity_ungated(im: Image.Image) -> dict:
    prom, wl, pk = periodicity(im)
    bs = band_sharpness(im, pk)
    ac = acutance(im)
    c = clip_frac(im)
    clip_penalty = max(0.0, 1.0 - 5.0 * c)

    return {
        "clarity_ungated": prom * (bs ** 0.5) * ac * clip_penalty,
        "periodicity": prom,
        "band_sharp": bs,
        "acutance": ac,
        "wavelength": wl,
        "clip": c,
        "in_band": LATTICE_WL_LO <= wl <= LATTICE_WL_HI,
    }


def _box_downsample(a: np.ndarray, factor: int) -> np.ndarray:
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    trimmed = a[:h, :w]
    return trimmed.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def scene_fidelity(frame: Image.Image, x_scene: Image.Image) -> float:
    """Layout-scale closeness to the source photo. Should dip at peak and recover by the end.

    Known blind on texture scenes (granite 0.473 against hedge 0.453) — use `arc_closure` for any
    closure claim. Kept because every archived run carries this number.
    """
    f = np.asarray(frame.convert("L"), dtype=np.float32)
    s = np.asarray(x_scene.convert("L").resize(frame.size), dtype=np.float32)
    fd = _box_downsample(f, FIDELITY_DOWNSAMPLE).ravel()
    sd = _box_downsample(s, FIDELITY_DOWNSAMPLE).ravel()

    fd = fd - fd.mean()
    sd = sd - sd.mean()
    denom = np.sqrt((fd * fd).sum() * (sd * sd).sum())
    if denom < 1e-9:
        return 0.0

    return float((fd * sd).sum() / denom)
