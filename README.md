# xddsp_block_nyq

MODULE NAME:
`xddsp_block_nyq`

DESCRIPTION:
First-order IIR **Nyquist blocker** (notch at Nyquist, fs/2) implemented in **XDDSP style** with pure functional NumPy + Numba.
It applies a zero at `z = -1` and a pole at `z = b1` (negative real), where `|b1| < 1`, and supports **exponential parameter smoothing** of the feedback coefficient over time.

---

INPUTS:

* `x` : input sample or buffer (float or 1D ndarray, audio-rate)
* `width_hz` : desired analog-like notch bandwidth around Nyquist in Hz (controls pole radius)
* `sr` : sampling rate in Hz (float)
* `smooth_time` : coefficient smoothing time constant in seconds (float, ≥ 0)
* `state` : filter state tuple `(d1, b1_prev)`
* `params` : parameter tuple `(b1_target, alpha)`

OUTPUTS:

* `y` : output sample or buffer (same shape as `x`)
* `state` : updated filter state `(d1_new, b1_prev_new)`

---

STATE VARIABLES (tuple):
`(d1, b1_prev)`

* `d1` : previous internal state sample (`d[n-1]`)
* `b1_prev` : previous smoothed feedback coefficient value used on the last sample (`b1[n-1]`)

---

EQUATIONS / MATH:

**Core filter (Nyquist blocker)**

Let:

* `d[n]` be the internal state at time `n`
* `x[n]` the input sample
* `y[n]` the output
* `b1[n]` the (possibly time-varying, smoothed) feedback coefficient

Then:

1. State update:
   [
   d[n] = x[n] + b_1[n] \cdot d[n-1]
   ]

2. Output:
   [
   y[n] = d[n] + d[n-1]
   ]

So the transfer function (for constant `b1`) is:
[
H(z) = \frac{Y(z)}{X(z)} = \frac{1 + z^{-1}}{1 - b_1 z^{-1}}
]
which has:

* a zero at `z = -1` → **notch at Nyquist**
* a pole at `z = b1` on the negative real axis

---

**Mapping bandwidth to pole radius**

We first compute a positive radius ( r \in (0,1) ) from bandwidth:

[
r = \exp\left(- \frac{2\pi \cdot \text{width_hz}}{\text{sr}}\right)
]

Then the Nyquist-blocker pole is set to:

[
b_{1,\text{target}} = -r
]

---

**Parameter smoothing rule (per-sample exponential smoothing)**

We maintain a smoothed `b1[n]` that approaches `b1_target` with time constant `smooth_time`:

1. Smoothing coefficient:
   [
   \alpha =
   \begin{cases}
   \exp\left(-\frac{1}{\text{smooth_time} \cdot \text{sr}}\right) & \text{if } \text{smooth_time} > 0 \
   0 & \text{if } \text{smooth_time} \le 0
   \end{cases}
   ]

2. Recursive smoothing:
   [
   b_1[n] = \alpha \cdot b_1[n-1] + (1 - \alpha) \cdot b_{1,\text{target}}
   ]

`b1_prev` in the state is `b1[n-1]`.

---

**State update rule**

Given previous state `(d1_prev, b1_prev)` and current input `x[n]`:

1. Smoothed feedback coefficient:
   [
   b_1[n] = \alpha \cdot b1_prev + (1 - \alpha) \cdot b_{1,\text{target}}
   ]

2. Internal state:
   [
   d[n] = x[n] + b_1[n] \cdot d1_prev
   ]

3. Output:
   [
   y[n] = d[n] + d1_prev
   ]

4. New state:
   [
   (d1_new, b1_prev_new) = (d[n], b_1[n])
   ]

---

through-zero rules:
Not applicable (linear, no through-zero FM / PM here).

phase wrapping rules:
Not applicable (no explicit phase state).

nonlinearities:
None; fully linear filter.

interpolation rules:
Parameter interpolation is exponential over time as defined above; no other interpolation.

any time-varying coefficient rules:
All time variation is via `b1[n]` smoothed per sample with the exponential recursion.

---

NOTES:

* **Stability constraint**: Ensure `width_hz >= 0` and `sr > 0`. The mapping to radius guarantees `|b1_target| < 1` for finite positive `width_hz`.
* Very small `width_hz` → `|b1_target|` close to 1 → very narrow notch, very long time constant.
* Setting `smooth_time <= 0` disables smoothing (instant jump).
* For typical musical / audio use, `width_hz` in tens to hundreds of Hz at 44.1–48 kHz works well.

---

## FULL PYTHON MODULE: `xddsp_block_nyq.py`

```python
"""
xddsp_block_nyq
----------------

First-order IIR Nyquist blocker in XDDSP style.

Behavior
========
Implements a time-varying, smoothed Nyquist-block filter with transfer function

    H(z) = (1 + z^-1) / (1 - b1 z^-1),

where b1 is a (possibly time-varying) feedback coefficient with |b1| < 1 and
b1 < 0, placing a zero at Nyquist (z = -1) and a pole on the negative real axis.

The internal state d[n] is updated as:

    d[n] = x[n] + b1[n] * d[n-1]
    y[n] = d[n] + d[n-1]

The Nyquist notch bandwidth is controlled by a "width" parameter in Hz, which
is mapped to a positive radius r via:

    r = exp(-2 * pi * width_hz / sr)

and then the Nyquist-blocker coefficient is:

    b1_target = -r

Parameter smoothing (exponential in time) ensures click-free modulation of
width / b1:

    alpha = exp(-1 / (smooth_time * sr))    (smooth_time > 0)
    alpha = 0                               (smooth_time <= 0)

    b1[n] = alpha * b1[n-1] + (1 - alpha) * b1_target

State and parameters are passed as tuples, enabling pure functional updates.

API
===

Public functions (pure functional):

    block_nyq_init(width_hz, sr, smooth_time) -> state, params
    block_nyq_update_state(state, width_hz, sr, smooth_time) -> state, params
    block_nyq_tick(x, state, params) -> (y, new_state)
    block_nyq_process(x, state, params) -> (y, new_state)

State tuple:

    state = (d1, b1_prev)

    d1       : previous internal state sample d[n-1]
    b1_prev  : previous smoothed feedback coefficient b1[n-1]

Params tuple:

    params = (b1_target, alpha)

    b1_target : target feedback coefficient computed from width_hz
    alpha     : smoothing coefficient in [0, 1)

All heavy DSP loops are implemented inside Numba-jitted functions, with
no dynamic allocation inside jitted code and no Python objects.
"""

import numpy as np
from numba import njit
from math import exp, pi


# ---------------------------------------------------------------------------
# Internal helpers (pure Python, not jitted)
# ---------------------------------------------------------------------------

def _width_to_radius(width_hz: float, sr: float) -> float:
    """Map notch bandwidth (Hz) to pole radius r in (0, 1).

    r = exp(-2 * pi * width_hz / sr)

    width_hz >= 0, sr > 0.
    """
    if width_hz < 0.0:
        width_hz = 0.0
    if sr <= 0.0:
        raise ValueError("Sampling rate sr must be positive.")
    return exp(-2.0 * pi * width_hz / sr)


def _radius_to_b1_nyq(radius: float) -> float:
    """Map positive radius r to Nyquist blocker feedback coefficient b1.

    For Nyquist blocker, the pole is on the negative real axis:

        b1 = -r
    """
    if radius < 0.0:
        radius = 0.0
    if radius > 1.0:
        radius = 1.0
    return -radius


def _compute_alpha(smooth_time: float, sr: float) -> float:
    """Compute exponential smoothing coefficient alpha in [0, 1).

    alpha = exp(-1 / (smooth_time * sr)) for smooth_time > 0
    alpha = 0 for smooth_time <= 0 (no smoothing)
    """
    if smooth_time <= 0.0:
        return 0.0
    # Clamp extremely small values to avoid underflow
    t = 1.0 / (smooth_time * sr)
    return exp(-t)


# ---------------------------------------------------------------------------
# Numba-jitted core DSP kernels
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _block_nyq_tick_jit(x: float,
                        d1: float,
                        b1_prev: float,
                        b1_target: float,
                        alpha: float):
    """
    Single-sample Nyquist blocker tick, jitted.

    Inputs:
        x         : current input sample
        d1        : previous internal state d[n-1]
        b1_prev   : previous smoothed coefficient b1[n-1]
        b1_target : target feedback coefficient
        alpha     : smoothing coefficient in [0, 1)

    Returns:
        y         : output sample
        d1_new    : new internal state d[n]
        b1_new    : new smoothed coefficient b1[n]
    """
    # Exponential smoothing of b1
    b1_new = alpha * b1_prev + (1.0 - alpha) * b1_target

    # Nyquist blocker core:
    # d[n] = x[n] + b1[n] * d[n-1]
    # y[n] = d[n] + d[n-1]
    t = x + d1 * b1_new
    y = t + d1

    d1_new = t
    return y, d1_new, b1_new


@njit(cache=True, fastmath=True)
def _block_nyq_process_jit(x: np.ndarray,
                           d1: float,
                           b1_prev: float,
                           b1_target: float,
                           alpha: float,
                           y: np.ndarray):
    """
    Buffer-based Nyquist blocker processing, jitted.

    Inputs:
        x         : input buffer (1D array)
        d1        : previous internal state d[n-1]
        b1_prev   : previous smoothed coefficient b1[n-1]
        b1_target : target feedback coefficient
        alpha     : smoothing coefficient in [0, 1)
        y         : preallocated output buffer (same shape as x)

    Returns:
        d1_new    : final internal state after processing the buffer
        b1_new    : final smoothed coefficient after processing
    """
    n = x.shape[0]
    d1_local = d1
    b1_local = b1_prev

    for i in range(n):
        # Per-sample smoothing
        b1_local = alpha * b1_local + (1.0 - alpha) * b1_target

        # Filter core
        t = x[i] + d1_local * b1_local
        y[i] = t + d1_local
        d1_local = t

    return d1_local, b1_local


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def block_nyq_init(width_hz: float,
                   sr: float,
                   smooth_time: float = 0.01):
    """Initialize Nyquist blocker state and parameters.

    Parameters
    ----------
    width_hz : float
        Notch bandwidth around Nyquist in Hz (>= 0).
    sr : float
        Sampling rate in Hz (> 0).
    smooth_time : float, optional
        Time constant for parameter smoothing in seconds.
        smooth_time <= 0 disables smoothing (jumps).

    Returns
    -------
    state : tuple
        Initial state (d1, b1_prev).
    params : tuple
        Parameters (b1_target, alpha).
    """
    radius = _width_to_radius(width_hz, sr)
    b1_target = _radius_to_b1_nyq(radius)
    alpha = _compute_alpha(smooth_time, sr)

    d1 = 0.0
    # Start with already-reached target to avoid initial transient:
    b1_prev = b1_target

    state = (d1, b1_prev)
    params = (b1_target, alpha)
    return state, params


def block_nyq_update_state(state,
                           width_hz: float,
                           sr: float,
                           smooth_time: float = 0.01):
    """Update Nyquist blocker parameters given new width / smoothing.

    This does not modify the state itself (d1, b1_prev) but recomputes
    b1_target and alpha. Use this when you change width or smoothing between
    blocks.

    Parameters
    ----------
    state : tuple
        Current state (d1, b1_prev).
    width_hz : float
        New notch bandwidth around Nyquist in Hz (>= 0).
    sr : float
        Sampling rate in Hz (> 0).
    smooth_time : float, optional
        Time constant for parameter smoothing in seconds.

    Returns
    -------
    state : tuple
        Unchanged state (d1, b1_prev) for convenience.
    params : tuple
        Updated parameters (b1_target, alpha).
    """
    radius = _width_to_radius(width_hz, sr)
    b1_target = _radius_to_b1_nyq(radius)
    alpha = _compute_alpha(smooth_time, sr)
    params = (b1_target, alpha)
    return state, params


def block_nyq_tick(x: float, state, params):
    """Single-sample Nyquist blocker tick (pure functional).

    Parameters
    ----------
    x : float
        Input sample.
    state : tuple
        Current state (d1, b1_prev).
    params : tuple
        Parameters (b1_target, alpha).

    Returns
    -------
    y : float
        Output sample.
    new_state : tuple
        Updated state (d1_new, b1_prev_new).
    """
    d1, b1_prev = state
    b1_target, alpha = params

    y, d1_new, b1_new = _block_nyq_tick_jit(x, d1, b1_prev, b1_target, alpha)
    new_state = (d1_new, b1_new)
    return y, new_state


def block_nyq_process(x, state, params):
    """Block-based Nyquist blocker processing (pure functional).

    Parameters
    ----------
    x : array_like
        1D input buffer.
    state : tuple
        Current state (d1, b1_prev).
    params : tuple
        Parameters (b1_target, alpha).

    Returns
    -------
    y : np.ndarray
        Output buffer (same shape as x).
    new_state : tuple
        Updated state (d1_new, b1_prev_new).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("block_nyq_process: x must be a 1D array.")
    y = np.empty_like(x_arr)

    d1, b1_prev = state
    b1_target, alpha = params

    d1_new, b1_new = _block_nyq_process_jit(
        x_arr, d1, b1_prev, b1_target, alpha, y
    )
    new_state = (d1_new, b1_new)
    return y, new_state


# ---------------------------------------------------------------------------
# Smoke test, plot example, listen example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sr = 48000.0
    width_hz = 200.0
    smooth_time = 0.01

    # Initialize filter
    state, params = block_nyq_init(width_hz, sr, smooth_time)

    # Smoke test: process an impulse
    n_samples = 4096
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0

    y_imp, state_imp = block_nyq_process(impulse, state, params)

    # Frequency response via FFT of impulse response
    fft_size = 4096
    H = np.fft.rfft(y_imp, n=fft_size)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sr)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))

    # Plot magnitude response
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, mag_db, label="Nyquist blocker magnitude")
    plt.axvline(sr / 2.0, color="red", linestyle="--", label="Nyquist")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("xddsp_block_nyq - Nyquist blocker magnitude response")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: listen example with white noise
    try:
        import sounddevice as sd

        duration = 2.0  # seconds
        n_noise = int(sr * duration)
        noise = np.random.randn(n_noise).astype(np.float64) * 0.2

        # Re-init state for listening
        state, params = block_nyq_init(width_hz, sr, smooth_time)
        y_noise, state_noise = block_nyq_process(noise, state, params)

        print("Playing dry white noise...")
        sd.play(noise, int(sr))
        sd.wait()

        print("Playing Nyquist-blocked white noise...")
        sd.play(y_noise, int(sr))
        sd.wait()

    except Exception as e:
        print("sounddevice not available or audio error:", e)
        print("Skipping listen example.")
```

---

### Next Improvements / Modules We Could Build

Some natural follow-ups in the same XDDSP style:

1. **DC + Nyquist dual blocker module**

   * Combined structure with both a DC notch (zero at z=1) and a Nyquist notch (z=-1), with independent bandwidths and smoothing.

2. **Time-varying Nyquist shaper**

   * Modulate `width_hz` at audio rate (LFO or envelope) and stress-test the parameter smoothing.
   * Add utilities for building “ringy” Nyquist emphasis / suppression effects.

3. **Stereo / N-channel wrapper**

   * Vectorized process that handles `(channels, n_samples)` in a single call, maintaining separate states per channel.

4. **Cascaded Nyquist blockers**

   * Multiple stages to increase attenuation at Nyquist while controlling phase / group delay.

If you tell me how you’re planning to use this (anti-imaging, anti-aliasing, resonator cleanup, etc.), we can design the next module to slot right into that pipeline.
