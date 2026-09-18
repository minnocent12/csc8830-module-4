from __future__ import annotations

import numpy as np
import pytest

from module4.fourier import (
    apply_frequency_filter,
    compute_fft2,
    frequency_derivative,
    frequency_grid,
    frequency_laplacian,
    gaussian_high_pass,
    gaussian_low_pass,
    local_frequency_energy,
    magnitude_spectrum,
    reconstruct_from_spectrum,
)


def test_fft_ifft_round_trip_preserves_input_without_mutation() -> None:
    image = np.arange(48, dtype=np.float64).reshape(6, 8)
    original = image.copy()

    reconstruction = reconstruct_from_spectrum(compute_fft2(image))

    np.testing.assert_allclose(reconstruction, image, atol=1e-12)
    np.testing.assert_array_equal(image, original)


def test_shifted_spectrum_reconstructs_and_log_magnitude_is_finite() -> None:
    image = np.zeros((8, 10), dtype=np.float64)
    image[2:5, 3:7] = 1.0
    shifted = compute_fft2(image, shifted=True)

    np.testing.assert_allclose(reconstruct_from_spectrum(shifted, shifted=True), image, atol=1e-12)
    assert magnitude_spectrum(shifted).shape == image.shape
    assert np.isfinite(magnitude_spectrum(shifted)).all()


def test_constant_image_has_only_dc_and_zero_gaussian_high_pass_response() -> None:
    image = np.full((16, 16), 7.0)
    spectrum = compute_fft2(image)
    high_pass = gaussian_high_pass(image.shape, sigma=0.1)

    assert abs(spectrum[0, 0]) == pytest.approx(7.0 * image.size)
    assert np.max(np.abs(spectrum[1:, :])) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(apply_frequency_filter(image, high_pass), 0.0, atol=1e-12)


def test_impulse_has_uniform_fourier_magnitude() -> None:
    image = np.zeros((8, 10), dtype=np.float64)
    image[3, 4] = 1.0

    magnitude = magnitude_spectrum(compute_fft2(image), logarithmic=False)

    np.testing.assert_allclose(magnitude, 1.0, atol=1e-12)


def test_gaussian_high_pass_is_complement_and_masks_have_expected_dc_layout() -> None:
    low_pass = gaussian_low_pass((9, 11), sigma=0.12)
    high_pass = gaussian_high_pass((9, 11), sigma=0.12)
    shifted_low_pass = gaussian_low_pass((8, 10), sigma=0.12, shifted=True)

    np.testing.assert_allclose(high_pass, 1.0 - low_pass, atol=1e-15)
    assert low_pass[0, 0] == pytest.approx(1.0)
    assert shifted_low_pass[4, 5] == pytest.approx(1.0)
    assert low_pass.shape == high_pass.shape == (9, 11)


def test_frequency_grid_keeps_x_in_columns_and_y_in_rows() -> None:
    u, v = frequency_grid((8, 16))

    assert u.shape == v.shape == (8, 16)
    assert u[0, 1] == pytest.approx(1 / 16)
    assert v[1, 0] == pytest.approx(1 / 8)
    assert u[1, 0] == pytest.approx(0.0)
    assert v[0, 1] == pytest.approx(0.0)


def test_fourier_derivatives_match_periodic_sinusoidal_fixture() -> None:
    rows, columns = 32, 40
    y, x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    image = np.sin(2.0 * np.pi * 3.0 * x / columns) + np.sin(2.0 * np.pi * 2.0 * y / rows)
    expected_x = (2.0 * np.pi * 3.0 / columns) * np.cos(2.0 * np.pi * 3.0 * x / columns)
    expected_y = (2.0 * np.pi * 2.0 / rows) * np.cos(2.0 * np.pi * 2.0 * y / rows)

    np.testing.assert_allclose(frequency_derivative(image, axis="x"), expected_x, atol=1e-12)
    np.testing.assert_allclose(frequency_derivative(image, axis="y"), expected_y, atol=1e-12)


def test_fourier_laplacian_matches_periodic_sinusoidal_fixture_and_constant_zero() -> None:
    rows, columns = 32, 40
    y, x = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    image = np.sin(2.0 * np.pi * 3.0 * x / columns) + np.sin(2.0 * np.pi * 2.0 * y / rows)
    expected = -((2.0 * np.pi * 3.0 / columns) ** 2) * np.sin(2.0 * np.pi * 3.0 * x / columns)
    expected -= (2.0 * np.pi * 2.0 / rows) ** 2 * np.sin(2.0 * np.pi * 2.0 * y / rows)

    np.testing.assert_allclose(frequency_laplacian(image), expected, atol=1e-11)
    np.testing.assert_allclose(frequency_laplacian(np.ones((8, 8))), 0.0, atol=1e-12)


def test_local_frequency_energy_separates_synthetic_smooth_and_striped_regions() -> None:
    image = np.zeros((32, 32), dtype=np.float64)
    x = np.arange(16)
    image[:, :16] = np.linspace(0.2, 0.8, 16)[None, :]
    image[:, 16:] = (x[None, :] % 2).repeat(32, axis=0)
    original = image.copy()

    energy = local_frequency_energy(image, window_size=8, cutoff=0.15)

    assert energy.shape == image.shape
    assert np.isfinite(energy).all()
    assert energy[:, 16:].mean() > energy[:, :16].mean()
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda: compute_fft2(np.zeros((2, 2, 1))),
        lambda: compute_fft2(np.array([[0.0, np.nan]])),
        lambda: gaussian_low_pass((4, 4), sigma=0),
        lambda: apply_frequency_filter(np.zeros((4, 4)), np.zeros((3, 4))),
        lambda: frequency_derivative(np.zeros((4, 4)), axis="z"),
        lambda: local_frequency_energy(np.zeros((4, 4)), window_size=2, cutoff=0.6),
    ],
)
def test_invalid_fourier_inputs_are_rejected(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
