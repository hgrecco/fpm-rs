# Roadmap

This file contains the project’s unfinished, in-scope work. Completed work and
historical implementation notes are kept in version control history.

Items marked **Design first** require an explicit design review before public
API or implementation work. A design must cover the model and measurement
assumptions, Rust and Python surfaces, checkpoint and serialization effects,
validation, deterministic tests, documentation, and the implementation’s
material differences from its cited method.

## Open work index

- **Reconstruction:** [bright-field initializer evaluation](#bright-field-initializer-evaluation),
  [APIC complex-field reconstruction](#apic-complex-field-reconstruction),
  [multi-wavelength reconstruction](#multi-wavelength-reconstruction), and
  [partial coherence and spectral bandwidth](#partial-coherence-and-spectral-bandwidth).
- **Datasets:** [Lazy large-dataset loading](#lazy-large-dataset-loading),
  [malformed-bundle coverage](#malformed-bundle-coverage), and
  [experimental dataset registration](#experimental-dataset-registration).
- **Simulation:** [Stable preset metric bounds](#stable-preset-metric-bounds),
  [mismatch robustness presets](#mismatch-robustness-presets),
  [angle-dependent illumination transmission](#angle-dependent-illumination-transmission),
  and [smooth phase object](#smooth-phase-object).
- **Metrics:** [Fourier ring correlation](#fourier-ring-correlation),
  [subpixel complex-field registration](#subpixel-complex-field-registration),
  and [resolution targets and contrast criteria](#resolution-targets-and-contrast-criteria).
- **Diagnostics:** [Benchmark source and crop records](#benchmark-source-and-crop-records)
  and [Python Polars integration](#python-polars-integration).
- **Documentation:** [Loss-selection guidance](#loss-selection-guidance),
  [consolidated model limitations](#consolidated-model-limitations), and
  [non-CPU backend guide](#non-cpu-backend-guide).
- **CUDA (requires scope change):** [CUDA kernels](#cuda-kernels),
  [device-resident reconstruction state](#device-resident-reconstruction-state),
  and [CPU and GPU validation](#cpu-and-gpu-validation).

## Reconstruction algorithms

### Bright-field initializer evaluation

The existing workflow is documented in
[bright-field planar-array initialization](docs/guides/reconstruction.md#bright-field-planar-array-initialization).
Remaining promotion work:

- [ ] Expand adverse-data coverage across amplitude, phase, and mixed specimens;
  weak texture; noise; near-cutoff illumination; conjugate ambiguity; radius
  mismatch; and known gain/background variation. Verify detection rejection,
  frame selection, acquisition permutation, and resident/lazy/backend parity
  across this matrix rather than only the existing bounded fixtures.
- [ ] Extend bounded physical recovery beyond translation to rotation, pitch or
  axial distance with the complementary scale fixed, and reference index.
  Verify rank and gauge failures, partial updates, and atomic model refresh
  while preserving pupil and intensity calibration.
- [ ] Extend the existing same-budget warm-start comparison to bright-field-rich
  and bright-field-poor acquisitions across pose, pitch, objective NA, noise,
  aberration, and specimen contrast. Compare cold `JointReconstruction`, circle
  initialization alone, and initialization followed by the same refinement
  budget.
- [ ] Benchmark capture range, detection recall/false acceptance, source-vector
  and physical-parameter error, fit rank/condition, forward-model passes,
  runtime, peak memory, downstream objective, and aligned complex-object error.
  Promotion requires a wider reliable capture range or less time to the same
  downstream error than cold joint calibration, with unsuitable data rejected.

### APIC complex-field reconstruction

- [ ] **Design first:** Treat aberration-corrected, closed-form complex-field
  reconstruction (APIC) as a separate non-iterative workflow. Define and
  validate its NA-matching and dark-field acquisition requirements, output
  model, tiling, and failure modes. Reference: R. Cao, C. Shen, and C. Yang,
  “High-resolution, large field-of-view label-free imaging via
  aberration-corrected, closed-form complex field reconstruction,” *Nature
  Communications* **15**, 4713 (2024),
  [Cao et al. (2024)](https://doi.org/10.1038/s41467-024-49126-y).

  **Proposed architecture; implementation pending:**

  - Scope the first implementation to monochromatic, scalar, coherent,
    image-plane imaging of one thin two-dimensional complex transmission field.
    Require a shift-invariant pupil and plane-wave illumination within each
    processed tile. Do not include diffraction-plane ptychography, multislice
    or multiple-scattering models, partial coherence, finite spectral
    bandwidth, multi-wavelength coupling, fluorescence, or three-dimensional
    reconstruction. APIC's Kramers–Kronig field recovery additionally relies
    on the signal-analyticity assumptions of Cao et al.; passing geometric
    validation is necessary but is not a proof that an arbitrary specimen
    satisfies those assumptions.
  - Add a public `apic` module with separate `ApicModel`, `ApicFramePlan`,
    `ApicProblem<M>`, `ApicOptions`, `ApicReconstructor`, `ApicResult`, and
    stage-diagnostic types. Do not implement `ReconstructionAlgorithm`, create
    a `ReconstructionState`, or run through `Runner`: APIC has no repeated
    global update, user-selected iteration count, schedule, convergence trace,
    early stopping, or algorithm checkpoint. Its finite dark-field sequence is
    a dependency-ordered analytical spectrum extension, not an optimization
    iteration. Reuse `MeasurementRead`, backend FFT operations, array-layout
    checks, complex-field utilities, metrics, and artifact infrastructure where
    their semantics remain valid.
  - Compile `ApicModel` from `Optics` and resolved `Illumination` at the
    experiment/model boundary. Store the vacuum wavelength, objective NA,
    object-plane sampling, physical transverse k-vectors, known pupil amplitude
    and support, single-source frame mapping, and enough information to derive
    crate-convention Fourier shifts for each tile. Do not retain LED geometry
    or let the reconstructor inspect apparatus positions. Share canonical
    sampling, pupil, FFT-shift, and crop-placement helpers with
    `ImagePlaneModel`, but do not infer objective NA back from a sampled pupil
    or duplicate the existing propagation-vector/Fourier-crop sign convention.
  - Represent active acquisition explicitly in `ApicFramePlan` as disjoint
    ordered sets of NA-matching and dark-field measurement indices. Provide an
    inspectable classifier that computes illumination NA from the compiled
    k-vector and wavelength, selects NA-matching frames using a named absolute
    dimensionless `na_matching_tolerance`, and selects dark-field frames only
    above a named non-negative `dark_field_margin_na`. Also allow a manually
    constructed plan for calibrated acquisitions. Bright-field and otherwise
    unused frames may remain in the measurement provider but do not enter the
    result; never infer roles from file order or metadata labels.
  - Require every active measurement frame to contain exactly one non-zero
    source contribution and require each active source to occur once in the
    first implementation. Reject multiplexing, repeated-source exposures,
    non-unit reconstruction weights, and per-pixel masks; APIC's Fourier-domain
    correlation equations do not give those values the same semantics as an
    iterative masked objective. Permit zero-weight frames only when they are
    absent from `ApicFramePlan`. Require finite non-negative intensity after
    configured measurement preprocessing and correct known scalar background,
    frame gain, source power, and source weight before analytical recovery.
    Do not silently clamp negative corrected samples or estimate unknown frame
    gains, source powers, or backgrounds.
  - Restrict the first grid realization to square tiles and circular pupil
    support with known, finite, positive amplitude on supported samples. Select
    the high-resolution grid through the existing `ReconstructionShape`
    policy from the union of shifted pupil supports instead of exposing the
    reference implementation's unconstrained padding factor. Require every
    active illumination shift to lie on the tile's Fourier grid within a named
    tolerance and reject fractional shifts rather than silently rounding them.
    A later design may add an analytically justified fractional-shift operator,
    rectangular tiles, or recovery of unknown pupil amplitude.
  - Make preflight validation construct the complete dependency plan before
    reading all pixels. The NA-matching overlap graph must be connected, its
    pupil-phase design matrix must have full rank after gauge constraints, and
    its shifted supports must seed a connected known spectrum. Every selected
    dark-field frame must have a non-empty unknown region and enough overlap
    with the spectrum known at its planned step to form the isolated
    cross-correlation equations. Choose a deterministic reachable dark-field
    order by greatest admissible known/unknown overlap, then illumination NA,
    then acquisition index; reject a plan if no remaining frame is reachable.
    Report azimuthal gaps, overlap fractions, expected support growth, matrix
    dimensions, and structural ranks as diagnostics rather than presenting
    them as guarantees of specimen-dependent numerical recovery.
  - Execute one tile as fixed stages: preprocess active intensities; recover
    the NA-matching complex spectra with the spatial Kramers–Kronig relation;
    solve the overlap phase-difference system for pupil aberration; correct and
    combine those spectra into the initial known spectrum; process each planned
    dark-field frame once to extend that spectrum; then inverse-transform and
    validate the final complex field. Redundant estimates of an already
    recovered Fourier sample may be combined once with deterministic
    signal-based weights, but no stage may feed the combined result back into
    an earlier stage or repeat until a loss decreases.
  - Recover pupil phase in an explicit normalized Zernike basis with documented
    OSA/ANSI indices, normalization, angular orientation, and radian-valued
    coefficients. The default basis must exclude piston and tip/tilt to fix the
    overlap system's gauge and coordinate origin. Do not reuse
    `PupilAberration`, whose coefficients are crate-specific unnormalized
    radial-polynomial weights. Return both coefficients and the sampled complex
    pupil. The sample field retains one unobservable global phase piston; tests
    and evaluations must align that piston rather than implying absolute phase.
  - For each dark-field frame, subtract the known-spectrum autocorrelation,
    isolate cross-correlation samples that do not overlap the remaining
    autocorrelation terms, and construct the complex linear operator for the
    new Fourier samples. Use a deterministic rank-revealing direct least-squares
    factorization with an explicit regularization value and singular-value or
    condition threshold. Do not label an iterative Krylov or optimizer solve as
    closed form. Record numerical rank, condition estimate, residual norm,
    known and recovered support fractions, and regularization for every frame;
    fail instead of returning finite-looking values when the declared rank or
    conditioning limit is violated.
  - Keep illumination calibration outside APIC. The first implementation uses
    the exact compiled k-vectors and must not reproduce the reference code's
    internal drift search or describe arbitrary pixel shifts as physical LED
    calibration. Users who need physical planar-array correction must calibrate
    `Optics` and `Illumination` first and then compile a new `ApicModel`; generic
    independent k-vector correction, if designed later, must remain explicitly
    non-physical. Likewise, defer paper-style unknown intensity correction;
    any later relative-scale recovery must be opt-in, mean-one constrained, and
    mutually exclusive with an unconstrained source-power/frame-gain pair.
  - Define `ApicTilePlan` with the full-frame region, a square low-resolution
    tile shape, deterministic tile origins, and real-space overlap. Each tile
    is reconstructed independently with the same compiled plane-wave model in
    the first milestone. Adjacent complex tiles must be phase-piston aligned
    from their valid real-space overlap by a weighted complex correlation;
    build a tile-overlap graph, require it to be connected, fix the first tile
    as the phase reference, solve all offsets consistently, and blend aligned
    fields with normalized deterministic windows. Reject weak or disconnected
    tile alignment instead of blending amplitude and wrapped phase separately.
    Per-tile illumination directions for finite-distance sources and spatially
    varying aberration are deferred until the experiment layer can compile
    them from physical tile coordinates.
  - Stream resident or lazy measurements one frame at a time and crop only the
    current tile. Retain the NA-matching spectra needed by the pupil solve, the
    growing high-resolution spectrum and coverage arrays, and one dark-field
    system at a time; never materialize the complete tiled measurement stack.
    Report peak sizes of the direct linear systems because their storage can
    dominate and scale poorly with tile length. Reuse the selected backend for
    supported FFT and buffer operations, but make no GPU-execution claim while
    rank-revealing linear algebra remains CPU-only.
  - Return an `ApicResult` rather than overloading `ReconstructionResult`.
    Include the global complex field, amplitude, wrapped phase, centered
    spectrum, Fourier-coverage mask, real-space blend-weight map, per-tile
    sampled pupils and Zernike coefficients, resolved frame plan and
    dark-field dependency order, stage/frame/tile diagnostic records, runtime,
    and metadata. The single-tile case uses the same representation with one
    tile. Do not synthesize an iteration trace, final objective, recovered
    physical calibration, or one global pupil when tiled recovery produced
    distinct pupil estimates. A final canonical forward-model residual may be
    reported as a diagnostic only, not as an APIC convergence criterion.
  - Use APIC-specific stage progress events such as tile start, NA-matching
    recovery, aberration recovery, dark-field frame completion, tile
    completion, alignment, and mosaic completion. Cancellation may take effect
    only at documented stage or frame boundaries. Do not send these events
    through iteration/frame callback hooks whose contexts require a mutable
    `ReconstructionState`; Python callbacks must document GIL reacquisition and
    the blocking behavior of `reconstruct` separately.
  - Define a separately tagged, versioned `ApicCheckpoint` only for stage and
    tile-boundary resume. Persist the model and measurement fingerprints,
    options, tile plan, resolved frame/dependency plan, completed tile
    artifacts, any current tile's known spectrum and coverage, recovered pupil,
    completed dark-field position, diagnostics, and elapsed time. Do not change
    or accept `ReconstructionCheckpoint`, and reject resume after any
    model-defining, acquisition, tile, or numerical option changes. An
    uninterrupted run and a boundary-resumed run must be bitwise equal under
    the deterministic CPU reduction order; mid-factorization checkpointing is
    out of scope.
  - Give APIC bundles a distinct manifest kind and schema rather than filling
    iterative result fields with sentinels. Store global arrays, per-tile pupil
    arrays and coefficients, coverage, plan, diagnostics, options, and
    provenance using the existing safe NPY/Parquet artifact conventions. The
    first implementation requires no `dataset_spec` schema change: construct
    the APIC frame plan from existing measurements plus explicit experiment
    inputs, and keep source-specific `.mat` conversion external. Revisit a
    dataset-level APIC role record only if a portable acquisition contract is
    demonstrated beyond one source format.
  - Classify failures at the closest boundary. Invalid parameters cover NA
    tolerances, grid tolerance, basis selection, regularization, condition
    limits, and tile overlap. Invalid models cover unsupported pupil/grid or
    multiplexing state. Invalid measurements cover active masks, weights,
    non-finite or negative corrected intensities, and frame-plan disagreement.
    Numerical errors identify the APIC stage, tile, and frame and include
    rank/conditioning or tile-alignment context. Unsupported features fail
    before execution. Do not skip a failed dark-field frame, publish a partial
    mosaic as success, or convert a numerical failure into a warning.
  - Expose matching Rust builders and keyword-only Python constructors for the
    model, explicit or classified frame plan, tile plan, numerical options,
    reconstructor, checkpoint, and result. Use SI units and `(row, column)` or
    `(height, width)` ordering at public boundaries. Python result arrays are
    owned NumPy values with documented complex128/float64/boolean dtypes and
    shapes; frame, tile, and Zernike metadata stay attached to typed records.
    Do not expose private `_core` names or a constructor that looks like an
    iterative algorithm with `iterations`, `batch_size`, `schedule`, or loss
    parameters.
  - Add deterministic tests for NA classification boundaries, explicit-plan
    validation, single-source enforcement, calibrated gain/background scaling,
    integer-grid tolerance, KK field recovery, overlap-graph and Zernike rank,
    pupil-phase recovery, initial stitching, one- and multi-step dark-field
    extension, direct-solve rank and condition failures, acquisition
    permutation invariance after planning, lazy/resident parity, crate sign and
    FFT conventions, single-tile parity, tile phase alignment, disconnected
    tiles, checkpoint resume, bundle round trips, Rust/Python parity, and all
    documented error mappings. Use independently derived synthetic fixtures and
    the published equations; do not copy the authors' GPL-3.0 MATLAB source
    into this MIT-licensed repository or make network/data downloads part of
    builds and tests.
  - Before promotion, compare APIC with fixed-acquisition `Epry` and
    `GradientDescent` on the same APIC-compatible deterministic simulations,
    reporting globally aligned complex-object error, pupil-phase error after
    gauge removal, Fourier coverage, forward residual, runtime, peak memory,
    and response to bounded illumination-angle, noise, and aberration changes.
    Validate an externally converted public APIC dataset when licensing and
    provenance permit, without registering source-specific conversion inside
    the crate. Document that fpm-rs uses explicit SI-unit compilation,
    deterministic rank/condition gates, separate progress/checkpoint/result
    contracts, known intensity calibration, and strict integer-grid inputs,
    whereas Cao et al. demonstrate MATLAB preprocessing, automatic
    illumination sorting, optional drift/intensity correction, and
    patch-oriented reconstruction.
  - During the later implementation change, add complete Rustdoc and the full
    Cao et al. reference to every principal public APIC entry point and the
    authoritative Python stub. Add APIC as a separate workflow, not an
    iterative-algorithm row, in `docs/guides/reconstruction.md`; define APIC,
    NA-matching illumination, analytical spectrum extension, and its gauge in
    the core concepts/glossary; document diagnostics, bundles, tiling, and
    failure handling; and update `CHANGES.md`. Run the complete Rust, Python,
    documentation, citation, checkpoint, bundle, example, and notebook checks
    with the implementation change.

### Multi-wavelength reconstruction

The implemented spectral, phase-unwrapping, and joint OPD workflows are
documented in
[reconstruct multiple wavelengths](docs/guides/reconstruction.md#reconstruct-multiple-wavelengths).
Spectral checkpoints/result bundles, the explicit spectral dataset profile,
mixing diagnostics, and the
[recovery benchmark](python/examples/benchmark_multi_wavelength.py) are implemented.
Remaining extensions:

- [ ] **Design first:** Add explicit wavelength-field registration and effective
  resolution matching before OPD mixing. Define physical coordinates,
  resampling, masks, and phase-reference transport; the current unwrapping
  workflow requires already registered fields with matched resolution.
- [ ] **Design first:** Extend branch handling to automatic or spatial
  unwrapping of the longest synthetic beat, with explicit gauge, continuity,
  branch-ambiguity, and invalid-pixel contracts.
- [ ] **Design first:** Define dispersive material/thickness parameterizations
  and their identifiability before extending the current nondispersive OPD
  constraint.
- [ ] **Design first:** Add blind spectral pupil or calibration recovery only
  after defining shared physical parameters, wavelength scaling, phase and
  amplitude gauges, source-power/frame-gain normalization, and checkpoint
  transport. Keep sampled pupils channel-specific; finite bandwidth and partial
  coherence belong to the following item.

### Partial coherence and spectral bandwidth

- [ ] **Design first:** Investigate mixed-state partial spatial coherence and
  finite spectral-bandwidth models. Treat this as a substantial forward-model,
  measurement, and identifiability project rather than a bounded algorithm
  option. Define the coherence representation, state-count selection, memory
  scaling, calibration gauges, and relationship to incoherent coded-source
  multiplexing before implementation. The following state-decomposition
  reference is a starting point rather than a specification of the crate’s
  general coherence model: S. Dong, R. Shiradkar, P. Nanda, and G. Zheng,
  [“Spectral multiplexing and coherent-state decomposition in Fourier
  ptychographic imaging”](https://doi.org/10.1364/BOE.5.001757), *Biomedical
  Optics Express* **5**(6), 1757–1767 (2014).

## Dataset format and loading

### Lazy large-dataset loading

- [ ] Add a lazy `dataset_spec` loading path for frame folders and large TIFF
  stacks without weakening bundle validation.

### Malformed-bundle coverage

- [ ] Extend existing manifest-validation tests to inconsistent frame counts
  and measurement shapes, invalid illumination metadata, and malformed
  multiplexed datasets.

### Experimental dataset registration

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
  [Loetgering et al. (2023)](https://doi.org/10.1364/OE.485370).
  Registry access may use the network explicitly, while builds, tests, and the
  local `DatasetLoader` must remain offline.

## Simulation and validation

### Stable preset metric bounds

- [ ] Record stable metric bounds for the named deterministic simulation
  presets.

### Mismatch robustness presets

- [ ] Add deterministic presets and robustness thresholds using the existing
  ownership boundaries: separate true and reconstruction geometries for
  illumination-position mismatch, `AcquisitionPlan` or acquisition errors for
  frame-gain mismatch, model or camera background as appropriate, and
  `CameraModel` for saturation, pixel sensitivity, and bad pixels. Do not
  duplicate those effects in `IlluminationAcquisitionErrors`.

### Angle-dependent illumination transmission

- [ ] **Design first:** Add a physically documented illumination-angle
  transmission or vignetting helper that resolves through stable
  `SourceCalibration` power. State whether the model represents source radiant
  intensity, irradiance projected onto the sample, collection vignetting, or a
  selected combination; do not label an arbitrary `cos^n(theta)` curve as a
  first-principles apparatus model.

### Smooth phase object

- [ ] Add a smooth phase-only synthetic object with an explicit spatial
  bandwidth.

## Metrics, registration, and resolution

### Fourier ring correlation

- [ ] **Design first:** Add two-field Fourier ring correlation for independent
  reconstruction splits. Define input-domain semantics, complex correlation,
  annular binning, windowing, masks, sample counts, pixel-radius output,
  optional physical-frequency calibration, and sample-count-dependent half-bit
  or one-bit threshold curves. References: N. Banterle, K. H. Bui, E. A. Lemke,
  and M. Beck, “Fourier ring correlation as a resolution criterion for
  super-resolution microscopy,” *Journal of Structural Biology* **183**(3),
  363–367 (2013),
  [Banterle et al. (2013)](https://doi.org/10.1016/j.jsb.2013.05.004),
  and M. van Heel and M. Schatz, “Fourier shell correlation threshold
  criteria,” *Journal of Structural Biology* **151**(3), 250–262 (2005),
  [van Heel and Schatz (2005)](https://doi.org/10.1016/j.jsb.2005.05.009).

### Subpixel complex-field registration

- [ ] **Design first:** Add explicit optional subpixel translation registration
  for complex-field comparison. Specify circular versus non-periodic boundaries,
  interpolation, mask overlap, allocation, fitted complex gain, and returned
  shift. Document that translation is not a universal ambiguity with fixed
  detector coordinates and a fixed pupil, while blind object/pupil recovery can
  retain a coupled affine-phase ambiguity. Reference the registration method to
  M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, “Efficient subpixel image
  registration algorithms,” *Optics Letters* **33**(2), 156–158 (2008),
  [Guizar-Sicairos et al. (2008)](https://doi.org/10.1364/OL.33.000156).

### Resolution targets and contrast criteria

- [ ] **Design first:** Replace or supplement the current custom bar pattern
  with a target that returns explicit feature metadata and physical spatial
  frequencies. Then add a documented contrast criterion and report resolved
  horizontal and vertical features or a contrast-versus-frequency curve. Do not
  report USAF group and element unless the generator implements the USAF layout
  and frequency mapping.

## Diagnostics and benchmarks

### Benchmark source and crop records

- [ ] Record resolved source frame indices, illumination associations, and
  spatial crops in benchmark records produced from dataset subsets.

### Python Polars integration

- [ ] Reconsider returning Python Polars DataFrames through `pyo3-polars` once
  the Rust Polars, Python Polars, PyO3, NumPy bindings, and supported wheel
  matrix can be upgraded and tested as one compatible set. The current
  bindings use PyO3 0.29 and NumPy 0.29; adding the adapter would couple their
  conversion traits and `pyo3-ffi` link requirements to both Polars runtimes.
  Until that full matrix is verified, bundles expose ordinary Parquet paths.

## Documentation

### Loss-selection guidance

- [ ] Expand loss-selection guidance in `docs/guides/reconstruction.md`. Explain
  that amplitude loss is a robust default across the large bright-field and
  dark-field dynamic range, Poisson negative log likelihood is appropriate for
  calibrated shot-noise-dominated counts, intensity MSE is sensitive to bright
  residuals, and read noise, clipping, unmodeled background, or outliers break a
  pure Poisson assumption. Describe the crate’s gain/background handling and
  ground the guidance in L. Bian, J. Suo, J. Chung, X. Ou, C. Yang, F. Chen,
  and Q. Dai, [“Fourier ptychographic reconstruction using Poisson maximum
  likelihood and truncated Wirtinger gradient”](https://doi.org/10.1038/srep27384),
  *Scientific Reports* **6**, 27384 (2016), and L.-H. Yeh, J. Dong, J. Zhong,
  L. Tian, M. Chen, G. Tang, M. Soltanolkotabi, and L. Waller,
  [“Experimental robustness of Fourier ptychography phase retrieval
  algorithms”](https://doi.org/10.1364/OE.23.033214), *Optics Express*
  **23**(26), 33214–33240 (2015).

### Consolidated model limitations

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

### Non-CPU backend guide

- [ ] Add a focused guide for implementing and validating a non-CPU resident
  backend when the first such backend is available.

## CUDA backend

These future items require an explicit project scope change before
implementation; the current library provides CPU image-plane FPM.

### CUDA kernels

- [ ] Implement a CUDA backend with cuFFT and resident kernels for Fourier
  crops, pupil operations, intensity formation, projection, and reductions.

### Device-resident reconstruction state

- [ ] Keep reconstruction state and scratch buffers device-resident across AP,
  FPIE, EPRY, ADMM, and gradient-descent updates.

### CPU and GPU validation

- [ ] Add CPU/GPU numerical-parity, unsupported-device, performance, and memory
  tests.
