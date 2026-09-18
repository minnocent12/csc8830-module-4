"""Small, explicit Fourier-domain helpers for Module 4 Question 3.

The public functions use NumPy's unshifted FFT layout by default: the zero-frequency
coefficient is at index ``[0, 0]``.  Passing ``shifted=True`` is a presentation/layout option
that applies ``fftshift`` to a spectrum or frequency mask; it does not change the represented
frequency content.  Spatial rows are the y axis and columns are the x axis, so ``v`` is built
from row frequencies and ``u`` from column frequencies.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _validate_scalar_image(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate a finite 2D real image and return a new float64 working copy."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D scalar image")
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise TypeError(f"{name} must contain real numeric values")
    values = image.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, Integral) for value in shape):
        raise TypeError("shape must contain two integer dimensions")
    rows, columns = (int(value) for value in shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive")
    return rows, columns


def _validate_spacing(spacing: float | tuple[float, float]) -> tuple[float, float]:
    """Return positive (row/y, column/x) sample spacings."""
    if isinstance(spacing, Real) and not isinstance(spacing, bool):
        row_spacing = column_spacing = float(spacing)
    else:
        if not isinstance(spacing, tuple) or len(spacing) != 2:
            raise TypeError("spacing must be a positive number or (row_spacing, column_spacing)")
        row_spacing, column_spacing = (float(value) for value in spacing)
    if not np.isfinite((row_spacing, column_spacing)).all() or row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("spacing values must be finite and positive")
    return row_spacing, column_spacing


def _validate_filter_mask(frequency_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(frequency_mask, np.ndarray) or frequency_mask.ndim != 2:
        raise TypeError("frequency_mask must be a 2D numpy.ndarray")
    if frequency_mask.shape != shape:
        raise ValueError(f"frequency_mask shape {frequency_mask.shape} does not match image shape {shape}")
    if not np.issubdtype(frequency_mask.dtype, np.number) or not np.isfinite(frequency_mask).all():
        raise ValueError("frequency_mask must contain finite numeric values")
    return frequency_mask.astype(np.complex128, copy=True)


def frequency_grid(
    shape: tuple[int, int],
    *,
    spacing: float | tuple[float, float] = 1.0,
    shifted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(u, v)`` frequency grids for columns/x and rows/y.

    Frequencies are cycles per spatial unit.  ``np.fft.fftfreq`` matches NumPy's FFT ordering;
    ``shifted=True`` applies the same layout change as ``np.fft.fftshift``.
    """
    rows, columns = _validate_shape(shape)
    row_spacing, column_spacing = _validate_spacing(spacing)
    v_values = np.fft.fftfreq(rows, d=row_spacing)
    u_values = np.fft.fftfreq(columns, d=column_spacing)
    u, v = np.meshgrid(u_values, v_values)
    if shifted:
        return np.fft.fftshift(u), np.fft.fftshift(v)
    return u, v


def compute_fft2(image: np.ndarray, *, shifted: bool = False) -> np.ndarray:
    """Compute a complex 2D DFT using NumPy's FFT convention."""
    values = _validate_scalar_image(image)
    spectrum = np.fft.fft2(values)
    return np.fft.fftshift(spectrum) if shifted else spectrum


def magnitude_spectrum(spectrum: np.ndarray, *, logarithmic: bool = True) -> np.ndarray:
    """Return ``|F|`` or the display-only ``log(1 + |F|)`` magnitude."""
    if not isinstance(spectrum, np.ndarray) or spectrum.ndim != 2:
        raise TypeError("spectrum must be a 2D numpy.ndarray")
    if not np.isfinite(spectrum).all():
        raise ValueError("spectrum contains non-finite values")
    magnitude = np.abs(spectrum)
    return np.log1p(magnitude) if logarithmic else magnitude


def reconstruct_from_spectrum(spectrum: np.ndarray, *, shifted: bool = False) -> np.ndarray:
    """Reconstruct a spatial image from a complex spectrum without taking its magnitude."""
    if not isinstance(spectrum, np.ndarray) or spectrum.ndim != 2:
        raise TypeError("spectrum must be a 2D numpy.ndarray")
    if not np.isfinite(spectrum).all():
        raise ValueError("spectrum contains non-finite values")
    unshifted = np.fft.ifftshift(spectrum) if shifted else spectrum
    return np.real_if_close(np.fft.ifft2(unshifted), tol=1000)


def gaussian_low_pass(
    shape: tuple[int, int],
    *,
    sigma: float,
    spacing: float | tuple[float, float] = 1.0,
    shifted: bool = False,
) -> np.ndarray:
    """Create a Gaussian low-pass mask in cycles-per-unit frequency coordinates."""
    if not isinstance(sigma, Real) or isinstance(sigma, bool) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    u, v = frequency_grid(shape, spacing=spacing, shifted=False)
    mask = np.exp(-((u * u) + (v * v)) / (2.0 * float(sigma) ** 2))
    return np.fft.fftshift(mask) if shifted else mask


def gaussian_high_pass(
    shape: tuple[int, int],
    *,
    sigma: float,
    spacing: float | tuple[float, float] = 1.0,
    shifted: bool = False,
) -> np.ndarray:
    """Create the complementary Gaussian high-pass mask ``1 - H_LP``."""
    return 1.0 - gaussian_low_pass(shape, sigma=sigma, spacing=spacing, shifted=shifted)


def apply_frequency_filter(
    image: np.ndarray,
    frequency_mask: np.ndarray,
    *,
    shifted: bool = False,
) -> np.ndarray:
    """Multiply a spectrum by a compatible mask and reconstruct its spatial response."""
    values = _validate_scalar_image(image)
    mask = _validate_filter_mask(frequency_mask, values.shape)
    spectrum = compute_fft2(values, shifted=shifted)
    filtered = spectrum * mask
    return np.real_if_close(reconstruct_from_spectrum(filtered, shifted=shifted), tol=1000)


def frequency_derivative(
    image: np.ndarray,
    *,
    axis: str,
    spacing: float | tuple[float, float] = 1.0,
) -> np.ndarray:
    """Compute a periodic Fourier derivative along x (columns) or y (rows)."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    values = _validate_scalar_image(image)
    u, v = frequency_grid(values.shape, spacing=spacing)
    multiplier = 1j * 2.0 * np.pi * (u if axis == "x" else v)
    # For a real input, the derivative response is represented by the real spatial component.
    # The single Nyquist bin for even dimensions has no distinct positive-frequency partner and
    # can otherwise leave a tiny convention-dependent imaginary residue.
    return np.real(np.fft.ifft2(np.fft.fft2(values) * multiplier))


def frequency_laplacian(
    image: np.ndarray,
    *,
    spacing: float | tuple[float, float] = 1.0,
) -> np.ndarray:
    """Compute the Fourier-domain Laplacian with transfer function ``-4 pi^2 (u^2 + v^2)``."""
    values = _validate_scalar_image(image)
    u, v = frequency_grid(values.shape, spacing=spacing)
    multiplier = -4.0 * np.pi**2 * (u * u + v * v)
    return np.real_if_close(np.fft.ifft2(np.fft.fft2(values) * multiplier), tol=1000)


def local_frequency_energy(
    image: np.ndarray,
    *,
    window_size: int | tuple[int, int],
    cutoff: float = 0.15,
) -> np.ndarray:
    """Return a same-size map of high-frequency energy measured independently per window.

    Windows are non-overlapping and edge windows may be smaller.  Each window has its mean
    removed before the FFT, so DC brightness does not dominate the selected high-frequency
    energy.  The returned map repeats each window's scalar energy over that window for display
    and simple educational thresholding; it is not a semantic segmentation result.
    """
    values = _validate_scalar_image(image)
    if isinstance(window_size, Integral) and not isinstance(window_size, bool):
        window_rows = window_columns = int(window_size)
    elif isinstance(window_size, tuple) and len(window_size) == 2:
        window_rows, window_columns = (int(value) for value in window_size)
    else:
        raise TypeError("window_size must be a positive integer or (rows, columns)")
    if window_rows <= 0 or window_columns <= 0:
        raise ValueError("window_size values must be positive")
    if not isinstance(cutoff, Real) or isinstance(cutoff, bool) or not np.isfinite(cutoff) or not 0 <= cutoff <= 0.5:
        raise ValueError("cutoff must be finite and between 0 and 0.5 cycles per pixel")

    energy_map = np.zeros(values.shape, dtype=np.float64)
    for row_start in range(0, values.shape[0], window_rows):
        row_end = min(row_start + window_rows, values.shape[0])
        for column_start in range(0, values.shape[1], window_columns):
            column_end = min(column_start + window_columns, values.shape[1])
            window = values[row_start:row_end, column_start:column_end]
            centered = window - np.mean(window)
            spectrum = np.fft.fft2(centered)
            u, v = frequency_grid(window.shape)
            high_frequency = np.hypot(u, v) >= float(cutoff)
            energy = float(np.sum(np.abs(spectrum[high_frequency]) ** 2) / (window.size**2))
            energy_map[row_start:row_end, column_start:column_end] = energy
    return energy_map
