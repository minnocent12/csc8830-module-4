# Question 3 — Fourier-Domain Edge Detection and Region Segmentation

This document is the canonical theory response for Module 4 Question 3, Parts A–F. The
supporting NumPy functions and Streamlit page are educational demonstrations of the equations;
they are not a trained detector, a semantic segmentation system, or empirical evidence about
RGB-versus-thermal performance.

## Notation and transform conventions

`f(x, y)` denotes a scalar spatial-domain image, where `x` is the column coordinate and `y` is
the row coordinate. `F(u, v)` denotes its frequency-domain representation, where `u` is the
frequency along `x` and `v` is the frequency along `y`. `j = sqrt(-1)`. Frequencies in the code
are cycles per spatial unit, and the Fourier exponent therefore contains `2 pi`.

The continuous transform convention is:

\[
F(u,v)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
f(x,y)e^{-j2\pi(ux+vy)}\,dx\,dy.
\]

The corresponding inverse is:

\[
f(x,y)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
F(u,v)e^{j2\pi(ux+vy)}\,du\,dv.
\]

Other texts may place normalization factors in the forward transform instead of the inverse.
The physical content is unchanged when a transform pair is used consistently. NumPy's `fft2`
uses the unnormalized forward transform and `ifft2` supplies the inverse normalization.

## Part A — 2D Fourier representation

For an `M x N` digital image `f[m,n]`, the finite 2D DFT used by the implementation is:

\[
F[k,l]=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f[m,n]
e^{-j2\pi\left(\frac{km}{M}+\frac{ln}{N}\right)}.
\]

Its inverse is:

\[
f[m,n]=\frac{1}{MN}\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}F[k,l]
e^{j2\pi\left(\frac{km}{M}+\frac{ln}{N}\right)}.
\]

The continuous transform describes an ideal function over an unbounded plane; the DFT describes
finite sampled data and implicitly uses periodic extension at the image boundaries. That
periodic-boundary assumption explains why discontinuities at opposite borders can contribute
high-frequency energy. `module4.fourier.compute_fft2` converts a validated scalar image to
floating point and computes the complex DFT without changing the caller's array.

Low spatial frequencies correspond to slowly varying intensity and broad structures. High
spatial frequencies correspond to rapid variation, fine detail, sharp transitions, and noise.
For a complex coefficient:

\[
|F(u,v)|=\sqrt{\operatorname{Re}(F)^2+\operatorname{Im}(F)^2}
\]

is its magnitude, while `angle(F(u,v))` is its phase. Magnitude describes how strongly a
frequency is present. Phase carries relative positional/alignment information and is often
critical to recognizable structure. Neither component alone is generally sufficient for exact
reconstruction: the inverse transform uses both magnitude and phase.

For display, `fftshift` moves the zero-frequency coefficient from the raw array corner to the
center, making radial low/high-frequency patterns easier to inspect. It only changes array
layout; it does not create or remove frequency content. The page displays `log(1 + |F|)` to
compress the dynamic range. That logarithm is for visualization only; reconstruction uses the
original complex spectrum.

## Part B — why edges contain high frequencies

An edge is a rapid change in intensity over a short spatial distance. A constant region is
mostly a DC/zero-frequency component. A smooth gradient has primarily low-frequency energy. A
sharp step transition changes much faster and requires a broad collection of Fourier components,
including substantial high-frequency terms, to reproduce its narrow transition. An ideal
discontinuity cannot be represented exactly by only a small low-frequency band.

This is why high-pass filtering can emphasize boundaries. It is not, however, an ideal semantic
edge detector: high frequencies also arise from sensor noise, fine texture, small objects,
compression artifacts, and boundary discontinuities. A high-pass response is a frequency-
enhanced spatial signal, not automatically a clean binary human mask.

## Part C — Gaussian high-pass filtering

For any low-pass transfer function `H_LP(u,v)`, the complementary high-pass construction is:

\[
H_{HP}(u,v)=1-H_{LP}(u,v).
\]

The filtered spectrum and reconstructed response are:

\[
G(u,v)=H_{HP}(u,v)F(u,v),\qquad
g(x,y)=\mathcal{F}^{-1}\{G(u,v)\}.
\]

The implementation uses a Gaussian low-pass mask as its primary demonstration:

\[
H_{LP}(u,v)=\exp\left(-\frac{u^2+v^2}{2\sigma^2}\right),
\qquad H_{HP}=1-H_{LP}.
\]

Here `sigma` is measured in cycles per spatial unit. The Gaussian changes smoothly with radial
frequency, so it avoids the strong ringing associated with an ideal hard circular cutoff. An
ideal cutoff can produce Gibbs-like ringing because an abrupt frequency boundary corresponds to
long oscillatory spatial sidelobes. Even with a Gaussian, the high-pass output should be
interpreted as an edge/detail response rather than a final segmentation mask.

## Part D — Fourier derivative property

Using integration by parts and assuming the boundary term vanishes under the transform
conditions:

\[
\mathcal{F}\left\{\frac{\partial f}{\partial x}\right\}
=j2\pi uF(u,v),
\qquad
\mathcal{F}\left\{\frac{\partial f}{\partial y}\right\}
=j2\pi vF(u,v).
\]

For example, differentiating with respect to `x` multiplies each Fourier coefficient by
`j 2 pi u`. The magnitude of that multiplier grows with `|u|`, so rapid variation along the
columns is emphasized. The analogous `v` multiplier emphasizes rapid variation along rows.
The phase factor `j` also encodes the quadrature relationship required for the signed
derivative; taking only a magnitude would lose the derivative's sign and spatial structure.

`module4.fourier.frequency_derivative` uses `np.fft.fftfreq` to construct FFT-compatible `u`
and `v` grids, multiplies the unshifted complex spectrum, and applies `ifft2`. For real input it
returns the real spatial component; an even-sized transform's single Nyquist bin has no distinct
positive-frequency partner and can otherwise leave a convention-dependent imaginary residue.
This avoids manually guessing centered-index coordinates. The periodic sinusoidal tests provide
a fixture whose analytical derivative is known exactly under the DFT's periodic boundary
convention.

## Part E — frequency-domain Laplacian

The spatial Laplacian is:

\[
\nabla^2f=\frac{\partial^2f}{\partial x^2}+\frac{\partial^2f}{\partial y^2}.
\]

Applying the derivative property twice gives:

\[
\mathcal{F}\left\{\frac{\partial^2f}{\partial x^2}\right\}
=-(2\pi u)^2F(u,v),
\]
\[
\mathcal{F}\left\{\frac{\partial^2f}{\partial y^2}\right\}
=-(2\pi v)^2F(u,v).
\]

Adding the terms produces:

\[
\mathcal{F}\{\nabla^2f\}
=-4\pi^2(u^2+v^2)F(u,v).
\]

The radial multiplier grows quadratically with frequency, so the Laplacian strongly emphasizes
fine detail and edges. That same quadratic growth makes it highly sensitive to high-frequency
noise. `frequency_laplacian` implements this transfer function with row/column frequency axes
in the correct order. A Laplacian response is an edge/detail operator and is not, by itself, a
complete region or human segmentation method.

## Part F — frequency-domain region segmentation

Frequency-selective region segmentation uses differences in texture or periodic structure rather
than relying only on average intensity. A classical workflow can:

1. transform an image or local window to `F(u,v)`;
2. select a radial band or directional band with `H(u,v)`;
3. measure energy, for example the sum of squared selected coefficients;
4. optionally inverse-transform the selected band to obtain a spatial response;
5. threshold or classify the response into candidate regions; and
6. apply transparent classical cleanup such as morphology or connected components.

A smooth region generally concentrates energy near low frequencies. A finely striped region can
have stronger high-frequency or directional peaks. This makes selected-band energy a useful
texture descriptor. It does not establish object identity or semantic meaning.

### Global-transform limitation

A global Fourier magnitude describes what frequencies occur in the complete image, but it does
not directly say where those frequencies occur. If a texture exists only on the left side, its
global spectrum can reveal the texture's frequencies without directly producing a left-side
pixel mask. This loss of localization is the central limitation of using one global FFT for
segmentation.

### Local/windowed analysis

To retain approximate spatial localization, divide the image into windows `f_k(x,y)` and compute

\[
F_k(u,v)=\mathcal{F}\{f_k(x,y)\}
\]

for each window. Selected-band energy or directional statistics can be assigned to the window,
forming a spatial descriptor map. Overlapping windows make the map smoother but increase cost;
non-overlapping windows are simpler and expose the spatial-frequency trade-off. Small windows
localize changes better but provide coarser frequency resolution. Larger windows provide finer
frequency resolution but blur the location of transitions. Window boundaries and finite-image
padding can also create spectral leakage.

`local_frequency_energy` is intentionally a small educational implementation. It subtracts each
window mean, measures energy at or above a configurable cutoff, and fills a same-size display map
with each window's scalar value. The Streamlit synthetic example contains a smooth region and a
striped region and is labeled **synthetic educational demonstration**. It is not a measured
assignment result and is not written to the empirical-results files.

### Advantages and limitations

Advantages include mathematically defined filters, efficient FFT computation, direct control of
frequency bands, strong interpretation of periodic texture, and the ability to analyze
directional frequency content. Limitations include global loss of localization, sensitivity to
noise, ringing from hard cutoffs, parameter selection, boundary effects, spectral leakage, and
the spatial-resolution/frequency-resolution trade-off of local windows. Frequency content does
not directly encode semantic objects, so Fourier processing alone is not generally sufficient for
arbitrary human segmentation and is not claimed to be universally superior to spatial methods.
