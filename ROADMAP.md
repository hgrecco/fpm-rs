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
- [x] Define and enforce a complete object/pupil gauge convention for
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

## Reconstruction algorithms

- [x] Adapt momentum-accelerated PIE (mPIE) to the existing
  rPIE-based `Fpie` update. Specify momentum state, friction and feedback
  parameters, batch and schedule semantics, checkpoint persistence, and whether
  the adaptation remains object-only or also supports pupil recovery. Reference:
  A. Maiden, D. Johnson, and P. Li, “Further improvements to the
  ptychographical iterative engine,” *Optica* **4**(7), 736–745 (2017),
  [https://doi.org/10.1364/OPTICA.4.000736](https://doi.org/10.1364/OPTICA.4.000736).

  **Implemented design:**

  - Add a separate public `Mpie` algorithm that reuses the sequential rPIE
    projection kernel used by `Fpie`. Keep `Fpie` behavior and defaults
    unchanged. `Mpie` remains object-only with a fixed pupil; pupil recovery and
    its object/pupil gauge are outside this milestone. It supports the same
    masks, frame weights, gains, backgrounds, and incoherent multiplexing as
    `Fpie` because momentum is applied after the complete measured-frame update.
  - Use `stability` for the rPIE denominator blend, `object_step` for the
    per-frame correction scale, `momentum_interval` for the number of effective
    frame updates between momentum events, `momentum_friction` for retained
    velocity, and `momentum_feedback` for the velocity added to the object. The
    defaults are `stability = 0.05`, `object_step = 0.2`,
    `momentum_interval = 30`, and `momentum_friction = momentum_feedback = 0.9`,
    which lie within the ranges reported by Maiden et al.; they are starting
    values to validate for image-plane FPM, not claimed universal optima.
  - Let `O_anchor` be the centered object spectrum immediately after the last
    momentum event, `V` the centered complex velocity spectrum, and `O_rpie` the
    object after the ordinary rPIE update for the current frame. After
    `momentum_interval` effective frame updates, apply
    `V <- momentum_friction * V + (O_rpie - O_anchor)` and then
    `O <- O_rpie + momentum_feedback * V`; store that `O` as the next anchor and
    reset the interval counter. Initialize `V` to zero and `O_anchor` to the
    initial object. The paper uses one `eta_obj` for both friction and feedback;
    the two public controls are intentionally separated here, while equal
    values reproduce its Eqs. (19) and (21). `object_step` is the adaptation of
    the paper's `gamma_obj` in Eq. (22).
  - Count one event after all source modes of a positive-weight measured frame
    have been projected and inserted. Do not count zero-weight frames. Carry a
    partial interval across batch and iteration boundaries; do not force a
    momentum event at either boundary. Consequently `batch_size` only groups
    runner calls and cannot change the numerical path. Sequential or seeded
    random acquisition order still changes the path intentionally, and a
    multiplexed frame counts once rather than once per source mode.
  - Add public `MpieAuxiliaryState` and an `Mpie` variant of
    `AlgorithmAuxiliaryState`. Persist the velocity spectrum, anchor spectrum,
    effective-frame counter, and the update parameters that determine their
    interpretation. A matching mPIE checkpoint resumes exactly, including in
    the middle of an interval. A checkpoint with no auxiliary state is accepted
    as an explicit warm start and initializes momentum from its stored object;
    a different algorithm's auxiliary variant is rejected. Keep checkpoint
    format version 2 because `algorithm_auxiliary` is already the versioned
    extension point; document that older readers cannot consume checkpoints
    containing the new enum variant.
  - Expose `Mpie` and all controls in Rust builders and the Python constructor,
    exports, and authoritative stub. Validate finite positive `object_step` and
    `epsilon`, `stability` in `[0, 1]`, `momentum_friction` in `[0, 1)`,
    `momentum_feedback` in `[0, 1]`, and nonzero `momentum_interval` and
    `batch_size`, using stable parameter names. Validate auxiliary dimensions,
    finite complex values, counter bounds, and stored-parameter agreement before
    applying an update. Reject mPIE inside `JointReconstruction` until a design
    defines whether velocity is reset or transported when physical calibration
    recompiles the forward model.
  - Refactor the shared projection loop only enough to run a post-frame momentum
    hook; do not change the forward model, amplitude projection, loss
    diagnostics, callback timing, or existing algorithms. Invalidate the
    object-domain cache after every momentum event and return `Error::Numerical`
    if an update produces a non-finite spectrum.
  - Add deterministic tests for the complex recurrence, default validation,
    stable validation errors, zero-feedback equivalence to identically
    configured `Fpie`, batch-size invariance, seeded schedule repeatability,
    zero-weight and multiplexed-frame counting, rejection of incompatible or
    malformed auxiliary state, and uninterrupted versus checkpoint-resumed
    equality when the interval crosses an iteration boundary. Add Python
    construction, validation, execution, and resume coverage.
  - Add `Mpie` to Rust and Python API documentation, the algorithm-selection
    guide, the benchmark example, and `CHANGES.md`, with the complete Maiden et
    al. citation and DOI near the method. State that the paper studies scanned
    ptychography, jointly accelerates object and probe, and explicitly leaves
    Fourier-ptychography testing as future work; this implementation instead
    accelerates only the fixed-pupil FPM object spectrum. Keep `CITATION.cff`
    unchanged because it describes how to cite this software rather than the
    scientific methods used inside it.
  - Before marking this item complete, demonstrate on a deterministic difficult
    FPM simulation that `Mpie` improves the gauge-aligned object error or reaches
    a fixed objective threshold in fewer effective frame updates than `Fpie`
    under the same schedule and update budget. Run `pixi run ci`, the Rust and
    Python API documentation checks, the MkDocs and notebook checks, citation
    checking, and the reconstruction benchmark/example.
- [x] **Design first:** Add an adaptive-step alternating-projection method only
  after defining its line-search or feedback state, failure behavior, scheduling
  semantics, and benchmark advantage over fixed-step AP. Reference: C. Zuo,
  J. Sun, and Q. Chen, “Adaptive step-size strategy for noise-robust Fourier
  ptychographic microscopy,” *Optics Express* **24**(18), 20724–20744 (2016),
  [https://doi.org/10.1364/OE.24.020724](https://doi.org/10.1364/OE.24.020724).

  **Accepted design:**

  - Add a separate public `AdaptiveAlternatingProjection` algorithm, leaving
    `AlternatingProjection` behavior and defaults unchanged. It is object-only
    with a fixed pupil, matching the main method evaluated by Zuo et al. It
    reuses the canonical amplitude-projection kernel and therefore supports
    masks, frame weights, gains, backgrounds, and incoherent multiplexing.
  - Use one object step for a complete acquisition-schedule pass. Start from
    `initial_object_step = 1`, retain it when the relative decrease between the
    two most recently completed pass objectives is greater than
    `progress_threshold = 0.01`, and otherwise multiply it by
    `reduction_factor = 0.5`, clamped to `minimum_object_step = 0.001`. The first
    pass establishes a baseline and cannot reduce the step. This is the
    feedback rule of Eq. (16), with an explicit positive floor as contemplated
    by the paper; it is not a backtracking line search and never retries or
    rolls back a pass.
  - Use the runner's existing measurement-domain amplitude-MSE summary as the
    feedback objective: the frame-weighted mean of per-frame, mask-aware pixel
    means, including the current gains and background. Accumulate it during the
    ordinary sequential projections, as suggested by Zuo et al., so adaptation
    performs no extra forward pass. Keep the diagnostic `loss_type` fixed to
    amplitude MSE for this algorithm so changing reporting cannot silently
    change feedback semantics.
  - Adapt only at complete pass boundaries. Carry partial objective sums across
    batches, making `batch_size` a runner grouping that cannot change the
    numerical path. Sequential, reverse, or seeded shuffled schedules remain
    supported and intentionally can produce different paths; each scheduled
    frame appears once per feedback cycle. Zero-weight frames contribute
    neither objective nor update. Reject missing positive feedback weight,
    non-finite objectives, skipped/repeated iteration indices, or non-finite
    updated state instead of silently resetting the controller.
  - Add public `AdaptiveAlternatingProjectionAuxiliaryState` and an
    `AdaptiveAlternatingProjection` variant of `AlgorithmAuxiliaryState`.
    Persist the active zero-based iteration, current step, previous completed
    objective, in-progress weighted objective sum and weight, and every scalar
    parameter that defines their interpretation. A matching checkpoint resumes
    exactly at the next pass. A checkpoint with no auxiliary state is accepted
    as an explicit warm start and establishes a new baseline; another
    algorithm's auxiliary state, malformed state, or changed controller
    parameter is rejected. Keep checkpoint format version 2 because the enum is
    already its versioned extension point, while noting that older readers
    cannot deserialize the new variant.
  - Reject the adaptive method inside `JointReconstruction`: physical model
    recompilation changes the feedback objective, and resetting or transporting
    controller history requires a separate design. Keep pupil recovery outside
    this milestone even though the paper's appendix explores applying the same
    step to object and pupil updates.
  - Expose builders for all controls in Rust and keyword-only constructor
    parameters in Python. Validate finite positive initial/minimum steps with
    the minimum no larger than the initial value, a finite progress threshold
    in `[0, 1)`, a finite reduction factor in `(0, 1)`, and positive
    `iterations`, `batch_size`, and `epsilon`, using stable parameter names.
    Record the effective object step as a per-iteration algorithm metric.
  - Add deterministic tests for the feedback recurrence and floor, validation,
    fixed-step equivalence before the first reduction, batch-size invariance,
    zero-weight frames, seeded scheduling, incompatible and malformed
    auxiliary state, and uninterrupted versus checkpoint-resumed equality.
    Add Python construction, validation, execution, metrics, and resume
    coverage.
  - Add the method and full reference to Rustdoc, the authoritative Python
    stub, the algorithm-selection guide, benchmark profiles/example, and
    `CHANGES.md`. State that the paper's convergence proof assumes convex
    component objectives whereas FPM phase retrieval is non-convex, and that
    this implementation uses the inexpensive accumulated objective
    approximation rather than an exact extra full-data evaluation.
  - Before marking this item complete, demonstrate on a deterministic noisy FPM
    simulation that the adaptive method reaches a lower gauge-aligned object
    error than fixed-step AP under the same schedule and update budget. Run
    `pixi run ci`, Rust and Python API documentation checks, MkDocs and notebook
    checks, citation checking, and the reconstruction benchmark/example.
- [x] **Design first:** Extend `GradientDescent` with the truncated-gradient or
  outlier-rejection rule of truncated Poisson Wirtinger reconstruction. Poisson
  negative log likelihood already exists; do not introduce a duplicate Poisson
  objective. Define truncation statistics for masks, gains, backgrounds,
  multiplexed frames, and mini-batches, and compare against the existing
  untruncated Poisson path. Reference: L. Bian, J. Suo, J. Chung, X. Ou,
  C. Yang, F. Chen, and Q. Dai, “Fourier ptychographic reconstruction using
  Poisson maximum likelihood and truncated Wirtinger gradient,” *Scientific
  Reports* **6**, 27384 (2016),
  [https://doi.org/10.1038/srep27384](https://doi.org/10.1038/srep27384).

  **Accepted design:**

  - Extend the existing public `GradientDescent` rather than adding another
    Poisson objective or algorithm type. Add an optional finite positive
    `poisson_truncation_threshold`; `None` remains the default and preserves the
    existing untruncated path exactly, while `25` reproduces the threshold
    coefficient selected by Bian et al. Exposing truncation with any other loss
    is a parameter error.
  - For one runner mini-batch and its pre-update state, predict the intrinsic
    detector intensity `p` and form the intrinsic target
    `y = max((measurement - background) / gain, 0)`. Exclude masked pixels and
    zero-weight frames. Define the batch statistic as the frame-weighted mean
    absolute residual over the remaining pixels,
    `R = sum(w * |y - p|) / sum(w)`, where each pixel carries its frame weight.
    Compute it once before any gradient in that batch is applied.
  - Let `z_rms = ||Z||_2`, where `Z` is the centered reconstruction spectrum.
    Under the library's normalized-forward FFT this is the RMS object-domain
    amplitude; it corresponds to the authors' `||fft2(z)||_2 / N` under
    MATLAB's unnormalized forward transform. Retain a detector pixel exactly when
    `|y - p| <= alpha * R * sqrt(p) / max(z_rms, sqrt(epsilon))`. This adapts
    the paper's global full-stack statistic to the solver's true mini-batches;
    consequently batch size and schedule may change the selected pixels, just
    as they already change the gradient trajectory.
  - For incoherent multiplexing, compute `p` from the sum of all weighted mode
    intensities and apply one detector-pixel gate to every contributing mode.
    Apply the same gate to object, pupil, and illumination-offset gradient and
    curvature contributions. Frame weights still scale accepted gradients.
    Continue to report the full, untruncated Poisson negative log likelihood so
    truncated and untruncated runs remain directly comparable.
  - Perform the statistic pass once for the whole batch before splitting
    parallel frame workers. Workers share that immutable statistic, and their
    accepted gradients are reduced in deterministic schedule order. Record the
    frame-weighted retained-pixel fraction as a `gradient_descent` iteration
    metric when truncation is enabled. Reject non-finite statistics rather than
    silently disabling the gate; accepting no pixels is a valid zero data step.
  - Truncation adds no evolving solver state: every gate is derived from the
    checkpointed reconstruction state and current batch. Checkpoint format and
    serialization therefore remain unchanged, and an uninterrupted run must
    match a resumed run when algorithm parameters and schedule match. The
    method remains usable inside `JointReconstruction`; a recompiled physical
    model simply defines the next batch's fresh statistic.
  - Expose the option in Rust builders, the Python keyword-only constructor,
    authoritative stubs, and reconstruction guidance. Add the complete Bian et
    al. reference to Rustdoc and the Python stub, and explicitly document the
    implementation's mini-batch statistic, calibrated intrinsic units,
    multiplexed-mode gate, fixed-step update, and optional pupil/calibration
    extensions relative to the paper's full-data, object-only presentation and
    scheduled step size.
  - Add deterministic tests for validation, disabled-path equivalence, the
    scalar gate boundary, masks, gains, backgrounds, multiplexing, batch and
    parallel semantics, retained-fraction metrics, checkpoint resume, and
    Python construction and execution. Before marking this item complete,
    demonstrate on a deterministic corrupted FPM simulation that truncation
    improves gauge-aligned object error over the identically configured
    untruncated Poisson path. Run `pixi run ci`, Rust and Python API
    documentation checks, MkDocs and notebook checks, citation checking, and
    the reconstruction benchmark/example.
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

## CUDA backend

- [ ] Implement a CUDA backend with cuFFT and resident kernels for Fourier
  crops, pupil operations, intensity formation, projection, and reductions.
- [ ] Keep reconstruction state and scratch buffers device-resident across AP,
  FPIE, EPRY, ADMM, and gradient-descent updates.
- [ ] Add CPU/GPU numerical-parity, unsupported-device, performance, and memory
  tests.
