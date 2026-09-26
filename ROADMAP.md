# Roadmap

This file contains the project’s unfinished, in-scope work. Completed work and
historical implementation notes are kept in version control history.

Items marked **Design first** require an explicit design review before public
API or implementation work. A design must cover the model and measurement
assumptions, Rust and Python surfaces, checkpoint and serialization effects,
validation, deterministic tests, documentation, and the implementation’s
material differences from its cited method.

## Model correctness and identifiability

- [x] Specify and enforce the low-resolution sampling invariant that relates
  object-plane camera pitch, wavelength, and objective NA. The design must
  distinguish coherent-field sampling from detector-intensity sampling, state
  the exact inequality and equality behavior, audit existing examples and
  presets, and use the existing parameter-validation error category unless a
  genuinely new recovery path requires another one. Document the invariant in
  `Optics` rustdoc and `docs/concepts/core-concepts.md`. The image-plane model
  follows G. Zheng, R. Horstmeyer, and C. Yang, “Wide-field, high-resolution
  Fourier ptychographic microscopy,” *Nature Photonics* **7**, 739–745 (2013),
  [https://doi.org/10.1038/nphoton.2013.187](https://doi.org/10.1038/nphoton.2013.187).
- [ ] **Design first:** Define a complete object/pupil gauge convention for
  pupil-recovering EPRY and gradient descent. Cover both magnitude scale and
  global or affine phase, choose a robust supported-pupil reference, compensate
  the object spectrum so the forward prediction is invariant, and specify how
  cadence interacts with regularization and checkpoint resume. Tests must check
  gauge normalization and forward-model invariance rather than expecting an
  arbitrary change in data-loss convergence. See A. Fannjiang and P. Chen,
  “Blind ptychography: uniqueness and ambiguities,” *Inverse Problems* **36**,
  045005 (2020),
  [https://doi.org/10.1088/1361-6420/ab6504](https://doi.org/10.1088/1361-6420/ab6504).
- [x] Add a compact table-driven convention suite covering representative
  wavelengths, pixel sizes, magnifications, and LED layouts.

## Core reconstruction

- [ ] Implement a CUDA backend with cuFFT and resident kernels for Fourier
  crops, pupil operations, intensity formation, projection, and reductions.
- [ ] Keep reconstruction state and scratch buffers device-resident across AP,
  FPIE, EPRY, ADMM, and gradient-descent updates.
- [ ] Add CPU/GPU numerical-parity, unsupported-device, performance, and memory
  tests.

## Reconstruction algorithms

- [ ] **Design first:** Adapt momentum-accelerated PIE (mPIE) to the existing
  rPIE-based `Fpie` update. Specify momentum state, friction and feedback
  parameters, batch and schedule semantics, checkpoint persistence, and whether
  the adaptation remains object-only or also supports pupil recovery. Reference:
  A. Maiden, D. Johnson, and P. Li, “Further improvements to the
  ptychographical iterative engine,” *Optica* **4**(7), 736–745 (2017),
  [https://doi.org/10.1364/OPTICA.4.000736](https://doi.org/10.1364/OPTICA.4.000736).
- [ ] **Design first:** Add an adaptive-step alternating-projection method only
  after defining its line-search or feedback state, failure behavior, scheduling
  semantics, and benchmark advantage over fixed-step AP. Reference: C. Zuo,
  J. Sun, and Q. Chen, “Adaptive step-size strategy for noise-robust Fourier
  ptychographic microscopy,” *Optics Express* **24**(18), 20724–20744 (2016),
  [https://doi.org/10.1364/OE.24.020724](https://doi.org/10.1364/OE.24.020724).
- [ ] **Design first:** Extend `GradientDescent` with the truncated-gradient or
  outlier-rejection rule of truncated Poisson Wirtinger reconstruction. Poisson
  negative log likelihood already exists; do not introduce a duplicate Poisson
  objective. Define truncation statistics for masks, gains, backgrounds,
  multiplexed frames, and mini-batches, and compare against the existing
  untruncated Poisson path. Reference: L. Bian, J. Suo, J. Chung, X. Ou,
  C. Yang, F. Chen, and Q. Dai, “Fourier ptychographic reconstruction using
  Poisson maximum likelihood and truncated Wirtinger gradient,” *Scientific
  Reports* **6**, 27384 (2016),
  [https://doi.org/10.1038/srep27384](https://doi.org/10.1038/srep27384).
- [ ] **Design first:** Evaluate a global Newton, Gauss–Newton, or practical
  quasi-Newton object solver against the existing sequential and mini-batch
  methods. Specify Hessian approximation, memory scaling, preconditioning,
  supported loss functions, and interaction with pupil and calibration
  variables before selecting an API. Reference: L.-H. Yeh, J. Dong, J. Zhong,
  L. Tian, M. Chen, G. Tang, M. Soltanolkotabi, and L. Waller, “Experimental
  robustness of Fourier ptychography phase retrieval algorithms,” *Optics
  Express* **23**(26), 33214–33240 (2015),
  [https://doi.org/10.1364/OE.23.033214](https://doi.org/10.1364/OE.23.033214).
- [ ] **Design first:** Evaluate bright-field circle detection or spectral
  correlation as an initialization or alternative calibration path for planar
  arrays. The existing `JointReconstruction` already performs bounded global
  physical pose, pitch, reference-index, selected-offset, source-power, and
  frame-gain calibration; preserve that implementation and keep it distinct
  from generic per-source Fourier-grid correction. Compare the proposed method
  with J. Sun, Q. Chen, Y. Zhang, and C. Zuo, “Efficient positional
  misalignment correction method for Fourier ptychographic microscopy,”
  *Biomedical Optics Express* **7**(4), 1336–1350 (2016),
  [https://doi.org/10.1364/BOE.7.001336](https://doi.org/10.1364/BOE.7.001336),
  and R. Eckert, Z. F. Phillips, and L. Waller, “Efficient illumination angle
  self-calibration in Fourier ptychography,” *Applied Optics* **57**(19),
  5434–5442 (2018),
  [https://doi.org/10.1364/AO.57.005434](https://doi.org/10.1364/AO.57.005434).
- [ ] **Design first:** Treat aberration-corrected, closed-form complex-field
  reconstruction (APIC) as a separate non-iterative workflow unless a design
  demonstrates that the iterative `ReconstructionAlgorithm` contract remains
  appropriate. Define and validate its NA-matching and dark-field acquisition
  requirements, output model, tiling, and failure modes. Reference: R. Cao,
  C. Shen, and C. Yang, “High-resolution, large field-of-view label-free imaging
  via aberration-corrected, closed-form complex field reconstruction,” *Nature
  Communications* **15**, 4713 (2024),
  [https://doi.org/10.1038/s41467-024-49126-y](https://doi.org/10.1038/s41467-024-49126-y).
- [ ] **Design first:** Define a multi-wavelength or spectral reconstruction
  architecture for separate or multiplexed measurements. Decide which object,
  pupil, geometry, source-power, and frame-gain parameters may be shared across
  wavelengths instead of assuming shared calibration, and preserve explicit
  wavelength-dependent sampling. Reference: S. Dong, R. Shiradkar, P. Nanda,
  and G. Zheng, “Spectral multiplexing and coherent-state decomposition in
  Fourier ptychographic imaging,” *Biomedical Optics Express* **5**(6),
  1757–1767 (2014),
  [https://doi.org/10.1364/BOE.5.001757](https://doi.org/10.1364/BOE.5.001757).
- [ ] **Design first:** Investigate mixed-state partial spatial coherence and
  finite spectral-bandwidth models. Treat this as a substantial forward-model,
  measurement, and identifiability project rather than a bounded algorithm
  option. Define the coherence representation, state-count selection, memory
  scaling, calibration gauges, and relationship to incoherent coded-source
  multiplexing before implementation. Dong et al. (2014), cited above, provides
  an initial state-decomposition reference but does not by itself define the
  crate’s general coherence model.

Every implemented reconstruction algorithm must receive rustdoc, complete
citation metadata, Rust and Python APIs, authoritative stubs, deterministic
tests, checkpoint coverage when it carries state, and an entry in the algorithm
selection guidance.

## Dataset format and loading

- [ ] Add a lazy `dataset_spec` loading path for frame folders and large TIFF
  stacks without weakening bundle validation.
- [ ] Expand generic bundle tests for malformed manifests, inconsistent frame
  counts and shapes, invalid illumination metadata, and multiplexed datasets.
- [ ] **Design first:** Register at least one redistributable experimental
  image-plane FPM dataset. Identify an immutable archive with an explicit
  license, dataset-level citation, checksum, compressed size, frame ordering,
  optical parameters, and enough provenance to produce a conforming
  `dataset_spec` bundle through an external conversion workflow. Investigate
  PtyLab-linked examples and original FPM laboratory releases, but do not treat
  the PtyLab software citation as a dataset citation: L. Loetgering et al.,
  “PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for
  conventional and Fourier ptychography,” *Optics Express* **31**(9),
  13763–13797 (2023),
  [https://doi.org/10.1364/OE.485370](https://doi.org/10.1364/OE.485370).
  Registry access may use the network explicitly, while builds, tests, and the
  local `DatasetLoader` must remain offline.

## Simulation and validation

- [ ] Record stable metric bounds for the named deterministic simulation
  presets.
- [ ] Add deterministic presets and robustness thresholds using the existing
  ownership boundaries: separate true and reconstruction geometries for
  illumination-position mismatch, `AcquisitionPlan` or acquisition errors for
  frame-gain mismatch, model or camera background as appropriate, and
  `CameraModel` for saturation, pixel sensitivity, and bad pixels. Do not
  duplicate those effects in `IlluminationAcquisitionErrors`.
- [ ] **Design first:** Add a physically documented illumination-angle
  transmission or vignetting helper that resolves through stable
  `SourceCalibration` power. State whether the model represents source radiant
  intensity, irradiance projected onto the sample, collection vignetting, or a
  selected combination; do not label an arbitrary `cos^n(theta)` curve as a
  first-principles apparatus model.
- [ ] Add a smooth phase-only synthetic object with an explicit spatial
  bandwidth.

## Metrics, registration, and resolution

- [ ] **Design first:** Add two-field Fourier ring correlation for independent
  reconstruction splits. Define input-domain semantics, complex correlation,
  annular binning, windowing, masks, sample counts, pixel-radius output,
  optional physical-frequency calibration, and sample-count-dependent half-bit
  or one-bit threshold curves. References: N. Banterle, K. H. Bui, E. A. Lemke,
  and M. Beck, “Fourier ring correlation as a resolution criterion for
  super-resolution microscopy,” *Journal of Structural Biology* **183**(3),
  363–367 (2013),
  [https://doi.org/10.1016/j.jsb.2013.05.004](https://doi.org/10.1016/j.jsb.2013.05.004),
  and M. van Heel and M. Schatz, “Fourier shell correlation threshold
  criteria,” *Journal of Structural Biology* **151**(3), 250–262 (2005),
  [https://doi.org/10.1016/j.jsb.2005.05.009](https://doi.org/10.1016/j.jsb.2005.05.009).
- [ ] **Design first:** Add explicit optional subpixel translation registration
  for complex-field comparison. Specify circular versus non-periodic boundaries,
  interpolation, mask overlap, allocation, fitted complex gain, and returned
  shift. Document that translation is not a universal ambiguity with fixed
  detector coordinates and a fixed pupil, while blind object/pupil recovery can
  retain a coupled affine-phase ambiguity. Reference the registration method to
  M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, “Efficient subpixel image
  registration algorithms,” *Optics Letters* **33**(2), 156–158 (2008),
  [https://doi.org/10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156).
- [ ] **Design first:** Replace or supplement the current custom bar pattern
  with a target that returns explicit feature metadata and physical spatial
  frequencies. Then add a documented contrast criterion and report resolved
  horizontal and vertical features or a contrast-versus-frequency curve. Do not
  report USAF group and element unless the generator implements the USAF layout
  and frequency mapping.

## Diagnostics and benchmarks

- [ ] Record resolved source frame indices, illumination associations, and
  spatial crops in benchmark records produced from dataset subsets.
- [ ] Reconsider returning Python Polars DataFrames through `pyo3-polars` once
  the Rust Polars, Python Polars, PyO3, NumPy bindings, and supported wheel
  matrix can be upgraded and tested as one compatible set. The current
  bindings use PyO3 0.29 and NumPy 0.29; adding the adapter would couple their
  conversion traits and `pyo3-ffi` link requirements to both Polars runtimes.
  Until that full matrix is verified, bundles expose ordinary Parquet paths.

## Documentation

- [ ] Add loss-selection guidance to `docs/guides/reconstruction.md`. Explain
  that amplitude loss is a robust default across the large bright-field and
  dark-field dynamic range, Poisson negative log likelihood is appropriate for
  calibrated shot-noise-dominated counts, intensity MSE is sensitive to bright
  residuals, and read noise, clipping, unmodeled background, or outliers break a
  pure Poisson assumption. Ground the guidance in Bian et al. (2016) and Yeh et
  al. (2015), cited above, and describe the crate’s gain/background handling.
- [ ] Consolidate known model limitations in
  `docs/concepts/core-concepts.md`: each source is internally spatially and
  temporally coherent, coded sources combine mutually incoherently, the object
  is a single thin complex-transmission slice, multiple scattering and
  multislice propagation are absent, defocus uses the documented paraxial
  quadratic phase, pupil-aberration coefficients are direct radian polynomial
  weights rather than Noll-normalized RMS coefficients, and no geometry-derived
  obliquity or vignetting law is applied automatically. Do not assign a universal
  paraxial NA validity cutoff; approximation error also depends on wavelength
  and propagation distance.
- [ ] Add a focused guide for implementing and validating a non-CPU resident
  backend when the first such backend is available.
