# Roadmap

This file contains the project’s unfinished, in-scope work. Completed work and
historical implementation notes are kept in version control history.

Items marked **Design first** require an explicit design review before public
API or implementation work. A design must cover the model and measurement
assumptions, Rust and Python surfaces, checkpoint and serialization effects,
validation, deterministic tests, documentation, and the implementation’s
material differences from its cited method.

## Open work index

- **Reconstruction:** [Global Gauss-Newton promotion](#global-gauss-newton-promotion),
  [bright-field initializer evaluation](#bright-field-initializer-evaluation),
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
- **CUDA:** [CUDA kernels](#cuda-kernels),
  [device-resident reconstruction state](#device-resident-reconstruction-state),
  and [CPU and GPU validation](#cpu-and-gpu-validation).

## Reconstruction algorithms

### Global Gauss-Newton promotion

- [ ] **Design first:** Evaluate a global Newton, Gauss–Newton, or practical
  quasi-Newton object solver against the existing sequential and mini-batch
  methods. Specify Hessian approximation, memory scaling, preconditioning,
  supported loss functions, and interaction with pupil and calibration
  variables before selecting an API. Reference: L.-H. Yeh, J. Dong, J. Zhong,
  L. Tian, M. Chen, G. Tang, M. Soltanolkotabi, and L. Waller, “Experimental
  robustness of Fourier ptychography phase retrieval algorithms,” *Optics
  Express* **23**(26), 33214–33240 (2015),
  [https://doi.org/10.1364/OE.23.033214](https://doi.org/10.1364/OE.23.033214).

  **Implemented design; verification and promotion benchmark pending:**

  - Use a separate object-only `GlobalGaussNewton` solver. Do not form or
    store the exact augmented complex Hessian proposed by Yeh et al.: it has
    quadratic object-size storage, can be indefinite away from a solution, and
    the paper's own comparison found that its lower iteration count did not
    offset its runtime. Also defer L-BFGS: its `O(history * object_pixels)`
    storage is practical, but its curvature history introduces checkpoint and
    physical-model transport semantics without exploiting the least-squares
    structure. Select a matrix-free, damped Gauss–Newton system because it is
    positive definite after damping, can use conjugate gradients, and retains
    global frame coupling with linear object-size memory.
  - Fix the data objective to the crate's canonical intrinsic-unit amplitude
    MSE. For frame `f`, remove known background and gain to obtain
    `y = max((measurement - background) / gain, 0)`, predict
    `p = sum_s(source_weight_s * |A_s O|^2)`, and use the masked amplitude
    residual `sqrt(p) - sqrt(y)`. Scale each valid pixel by the square root of
    `frame_weight / (valid_pixels_in_frame * positive_frame_weight_sum)` so
    `||r(O)||^2` is exactly the runner's frame-weighted, mask-aware mean
    amplitude MSE. Zero-weight frames contribute neither residuals nor
    derivatives. `A_s` is the canonical compiled crop, fixed pupil, and inverse
    FFT operator, including subpixel offsets; an incoherently multiplexed frame
    has one detector residual whose derivative includes every source mode.
  - Implement analytic real-linear products `J v` and `J^T u` for the complex
    object under the real inner product `Re(a^H b)`. Stabilize the amplitude
    derivative with `max(sqrt(p), sqrt(epsilon))`, matching the existing
    gradient path's dark-pixel convention. Solve
    `(J^T J + damping * diag(C)) d = -J^T r` from zero with preconditioned
    conjugate gradients. `C` is a positive pupil-power Fourier-coverage
    approximation accumulated with frame and multiplexing weights, normalized
    and floored by `epsilon`; use `(1 + damping) * C` as the Jacobi
    preconditioner. The approximation deliberately omits field phase and the
    spatial pattern of masks, which remain present in the exact matrix-free
    `J` and `J^T` products.
  - Stream frames and, for multiplexing, retain only the modes of the current
    frame. Do not cache the measurement stack, all predicted fields, a
    Jacobian, or a Hessian. With `N` complex object pixels, `P` detector pixels,
    and at most `q` simultaneous source modes, storage is `O(N + qP)` with a
    fixed number of object-sized conjugate-gradient vectors, rather than
    `O(N^2)` for the full Hessian or `O(history * N)` for L-BFGS. This preserves
    lazy measurements but processes all frames for each normal-operator and
    line-search evaluation; the expected cost of one outer iteration is one
    global gradient pass, up to `maximum_cg_iterations` global `J^T J` passes,
    and the trial-objective passes required by line search.
  - Make one solver step atomic and global. `batch_size()` returns `usize::MAX`,
    and `step` rejects a batch that is not an exact permutation of every frame.
    Accumulate frames in canonical index order so sequential, reverse, and
    seeded schedules give the same numerical path. Initially keep reductions
    single-threaded and deterministic; use the state's backend for FFTs without
    adding a CPU-only public contract.
  - Globalize the inexact direction with Armijo backtracking on the same full
    amplitude-MSE objective. Start with step one and multiply by
    `line_search_reduction` until sufficient decrease is reached. Reaching the
    conjugate-gradient iteration limit is allowed when the finite direction is
    a descent direction; non-positive or non-finite conjugate-gradient
    curvature, a non-descent direction, or exhausting the line search returns
    `Error::Numerical` without modifying the object. A zero gradient is a
    successful no-op. Commit only an accepted trial, invalidate the object
    cache, and return post-update per-frame losses so trace and callback
    objectives describe the accepted state.
  - Limit the first milestone to a fixed pupil and fixed compiled calibration
    variables. Honor a checkpoint's pupil, gains, background, and generic
    source corrections, but do not update them; do not offer TV or pupil
    regularization. Because no curvature survives an outer step,
    `GlobalGaussNewton` may be used by Rust and Python `JointReconstruction`:
    each physical model refresh is followed by a fresh linearization. Joint
    object/pupil or object/calibration Gauss–Newton blocks remain out of scope.
    The fixed-pupil global object-phase ambiguity is left in the initialization
    convention and handled by the existing gauge-aligned evaluation metrics.
  - Add Rust fields and builders, and matching keyword-only Python parameters,
    for `iterations`, `damping`, `maximum_cg_iterations`,
    `cg_relative_tolerance`, `maximum_line_search_steps`,
    `line_search_reduction`, `line_search_sufficient_decrease`, and `epsilon`.
    Proposed starting defaults are `20`, `1e-3`, `12`, `1e-3`, `8`, `0.5`,
    `1e-4`, and `1e-10`, respectively; they are implementation starting points
    to validate, not claimed universal optima. Validate positive finite damping
    and epsilon, positive iteration limits, tolerances in `(0, 1)`, and a
    line-search reduction in `(0, 1)`, using stable parameter names. Do not
    expose `batch_size`, `loss_type`, pupil recovery, illumination recovery, or
    regularization controls on this initial API.
  - Emit `global_gauss_newton` iteration metrics for conjugate-gradient
    iterations, final linear residual ratio, line-search evaluations, accepted
    step scale, and gradient norm. The solver carries no
    `AlgorithmAuxiliaryState`: checkpoints occur only after the atomic outer
    step, same-configuration resume is exact, and checkpoint format version 2
    remains unchanged. Reject unrelated stateful algorithm auxiliary data;
    changing solver parameters on resume is an explicit stateless warm start,
    consistent with the existing stateless algorithms.
  - Before marking this item complete, compare the public candidate from the
    same initialization with fixed-pupil `Fpie` and amplitude-MSE
    `GradientDescent` on `noiseless_mixed_v1`, `poisson_gaussian_v1`, and the
    fixed-pupil model-mismatch case `aberrated_pupil_v1`. Report outer
    iterations, forward/adjoint or normal-operator passes, peak resident memory,
    median CPU runtime over repeated runs, final objective, and globally aligned
    complex-object error. Promotion requires a predeclared objective or object
    error threshold to be reached faster on at least one difficult case, or a
    lower aligned object error at the same runtime budget, without a noiseless
    regression. If neither condition holds, document the negative result,
    remove the public candidate, and close this roadmap item rather than
    keeping a maintenance-heavy algorithm on iteration-count evidence alone.
  - Before marking this item complete, add finite-difference `J v`, real-adjoint
    dot-product, and dense tiny-problem `J^T J` checks; deterministic tests for
    masks, weights, gains, backgrounds, fractional crops, multiplexing,
    coverage floors, conjugate-gradient stopping, line-search rejection without
    mutation, schedule invariance, joint-model refresh, validation, and
    uninterrupted versus checkpoint-resumed equality; and Python construction,
    validation, execution, metrics, joint reconstruction, and resume coverage.
    Add the algorithm to Rustdoc, the authoritative Python stub, algorithm
    selection guidance, benchmark profiles/example, and `CHANGES.md`.
  - Document that Yeh et al. use exact CR-calculus Hessians and report global
    Newton results for amplitude, intensity, and Poisson objectives, whereas
    this proposal uses only the positive-semidefinite Gauss–Newton part of the
    amplitude residual with explicit damping, matrix-free products, and a fixed
    pupil. Matrix-free Levenberg–Marquardt is supported as a ptychographic
    implementation precedent by S. Kandel, S. Maddali, Y. S. G. Nashed,
    S. O. Hruszkewycz, C. Jacobsen, and M. Allain, “Efficient ptychographic
    phase retrieval via a matrix-free Levenberg–Marquardt algorithm,” *Optics
    Express* **29**(15), 23019–23055 (2021),
    [https://doi.org/10.1364/OE.422768](https://doi.org/10.1364/OE.422768), but
    that work studies diffraction-plane ptychography and automatic
    differentiation rather than this crate's analytic image-plane FPM model.

### Bright-field initializer evaluation

- [ ] **Implementation and evaluation:** Evaluate bright-field circle detection or spectral
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

  **Implemented core architecture; promotion validation remains:**

  - Choose bright-field circular-edge detection as an optional physical
    planar-array initializer, not as another reconstruction algorithm or a
    replacement for `IlluminationCalibration`. Its output is a bounded,
    serializable `Illumination` and a consistently refreshed
    `ImagePlaneModel`; users may reconstruct directly from that warm start or
    pass it to the existing `JointReconstruction` for measurement-loss
    refinement. Do not automatically run either stage or hide their separate
    diagnostics and failure modes behind one convenience call.
  - Scope the first implementation to the crate's monochromatic, scalar,
    coherent, thin-sample image-plane model with a shift-invariant circular
    pupil. Circle localization assumes that selected bright-field frames have
    enough specimen DC/reference interference and spatial texture to expose
    the two pupil-radius edges in the Fourier transform of measured intensity.
    Treat this as a data-quality precondition, not a guarantee implied by
    `PlanarLedArray`. Do not claim support for diffraction-plane ptychography,
    multislice or three-dimensional models, partial coherence, non-circular
    pupil support, or spatially varying illumination within one processed
    image.
  - Add a public `illumination_initialization` module with
    `BrightfieldCircleInitializer`, `BrightfieldCircleOptions`,
    `BrightfieldCircleObservation`, `PlanarArrayInitializationDiagnostics`,
    and `PlanarArrayInitializationResult`. The initializer consumes
    `MeasurementRead`, `Optics`, the nominal `Illumination`, and the matching
    `ImagePlaneModel`; it must not inspect dataset-specific filenames or infer
    physical source identity from `FrameMetadata`. Validate that the supplied
    model was compiled from the nominal illumination, while permitting its
    complex pupil values and known background to have been updated without
    changing support, sampling, shapes, or source/frame topology.
  - Derive frame-to-source identity only from the canonical
    `AcquisitionPlan`. An active detection frame must contain exactly one
    positive source contribution and have positive known frame gain, source
    power, and measurement weight. Repeated single-source exposures may yield
    repeated observations of the same stable source. Zero-weight frames and
    multiplexed frames may remain in the acquisition but are excluded; reject
    an explicitly selected ineligible frame. Require active masks to be absent
    or all-valid because arbitrary missing pixels create global Fourier
    artifacts. Other, unused frames retain their ordinary reconstruction
    semantics.
  - Select automatic candidates conservatively from the nominal compiled
    illumination. In dimensionless NA units, require the sum of nominal
    illumination NA, `center_search_radius_na`, and `brightfield_margin_na` to
    be strictly less than `objective_na`, so every allowed center remains
    strictly bright-field.
    Also allow an explicit acquisition-frame subset subject to the same safety
    test. Do not decide bright-field membership from mean intensity, filename
    order, or a nominal LED index rectangle. Require enough distinct source
    locations and lattice-axis coverage for the requested physical fit before
    reading every frame.
  - For each active frame, subtract the model's known optical background and
    divide by its complete positive scalar intensity factor before image
    processing. Apply one documented deterministic separable apodization to
    reduce boundary leakage, transform with the selected backend, and form the
    centered magnitude spectrum. Do not silently estimate unknown background,
    gain, or source power and do not silently clamp negative corrected samples;
    validate finite input and report the negative-sample fraction as a
    diagnostic. Compute the mean magnitude spectrum in a first streaming pass,
    then divide each spectrum by the floored mean and detect it in a second
    pass. This preserves resident/lazy parity without retaining the image
    stack.
  - Express all circle geometry in the model's physical transverse
    `kx`/`ky` coordinates, using `sampling.dkx` for columns and `sampling.dky`
    for rows. A circular physical pupil is generally an ellipse in pixel-index
    coordinates for a rectangular image, so do not assume a square detector or
    one common pixel radius. Search one shared pupil cutoff in a bounded range
    around `2π * objective_na / wavelength_vacuum_m`, smooth with the
    configured `gaussian_sigma_pixels`, and evaluate the first- and
    second-radial-derivative circular-edge scores described by Eckert et al.
    Normalize each score by valid angular coverage and evaluate only arcs that
    remain inside the sampled Fourier domain and, when the centers are
    separated, outside the predicted overlap with the conjugate circle. A
    nominally on-axis frame may contribute to shared-radius estimation but not
    to a signed off-axis center observation.
  - Search each center only inside the declared physical radius around its
    nominal `KVector`, then use deterministic coarse-to-fine refinement with
    interpolation for a subpixel center. Because the magnitude spectrum of a
    real intensity image is centrosymmetric, the observations at `+k` and
    `-k` do not determine the illumination sign by themselves. Use the nominal
    physical array solely to label that branch, report the competing score,
    and reject a noncentral observation when the bounded search regions overlap
    or the two branches cannot be distinguished under the declared ambiguity
    threshold. Never resolve the sign from acquisition order.
  - Return a detection only when the first- and second-derivative candidate
    sets agree, the usable arc fraction and normalized edge contrast exceed
    explicit minima, the local peak is isolated, and the refined center lies
    inside the search bound. Record both candidate metrics, estimated center
    and covariance proxy, fitted pupil radius, arc fraction, contrast,
    conjugate ambiguity, and rejection reason for every considered frame.
    The shared fitted radius is detector state and a validation diagnostic; it
    must agree with the compiled objective NA within a named tolerance and must
    not mutate `Optics`, magnification, wavelength, sampling, or pupil support.
  - Fit accepted centers directly to the canonical `PlanarLedArray::resolve`
    mapping rather than first producing arbitrary per-source shifts or an
    unconstrained affine transform. Reuse `PlanarArrayCalibrationParameters`,
    `CalibrationParameterSpec`, absolute SI units, bounds, scales, priors, and
    the active right-handed extrinsic XYZ convention. Refactor the physical
    parameter extraction, gauge constraints, and illumination reconstruction
    into a private shared helper used by both initialization and calibration;
    this refactor must leave the existing measurement-loss optimizer's
    numerical behavior unchanged.
  - Limit circle-based fitting to global translation, rotation, pitch, and
    reference-index variables. Reject selected per-source XYZ offsets because
    one detected transverse direction supplies only two constraints, and
    reject source-power and frame-gain variables because circle centers carry
    no multiplicative information. Preserve the existing translation/reference
    index exclusions. Compute the data Jacobian of accepted `(NA_x, NA_y)`
    residuals before adding priors, require full column rank after gauge
    constraints, and reject insufficient source count, lattice span, or
    azimuthal diversity. Priors and finite bounds may regularize a noisy fit but
    must not be presented as making a data-rank-deficient physical parameter
    identifiable.
  - Use a deterministic bounded robust nonlinear least-squares fit in scaled
    parameter coordinates. Detection confidence may set fixed observation
    weights, while a named robust residual in NA units suppresses gross circle
    outliers; neither measurement frame weights nor repeated sources may alter
    the physical parameter gauge. Commit a candidate only after rebuilding the
    full `Illumination`, resolving all sources, refreshing a cloned model
    through `update_illumination_geometry`, and confirming bounds, fixed-grid
    crop validity, topology, and objective decrease. A failed trial must not
    partially mutate the nominal illumination or model.
  - Preserve source powers, sparse acquisition rows, frame gains, pupil values,
    background, Fourier sampling, and reconstruction shape byte-for-byte while
    applying the fitted geometry. Return the nominal and initialized
    illumination, the initialized model, absolute and normalized physical
    values, accepted/rejected fit history, and all detection/conditioning
    diagnostics. This result is a physical planar-array estimate; do not also
    populate generic `(row, column)` Fourier-grid corrections.
  - Do not implement Eckert et al.'s spectral-correlation stage in this
    milestone. Its per-source local grid search is object- and pupil-dependent,
    overlaps the existing generic `GradientDescent(recover_illumination=true)`
    path, and would require projecting independent shifts back onto physical
    parameters in competition with `JointReconstruction`'s direct bounded
    physical objective. If later evidence supports it, design it as another
    explicitly generic Fourier-grid correction strategy or as an internal
    proposal mechanism for the physical calibrator, with defined subpixel,
    multiplexing, loss, schedule, checkpoint, and pose-projection semantics;
    do not label independent spectral shifts as realizable apparatus
    calibration.
  - Keep initialization outside `ReconstructionAlgorithm`, `Runner`, and
    `ReconstructionState`. It has two deterministic measurement passes followed
    by a small physical fit, not reconstruction iterations. Add
    initializer-specific progress events for mean-spectrum accumulation,
    per-frame detection, physical-fit steps, and completion; cancellation may
    take effect at those boundaries. Do not send them through iteration or
    frame callbacks whose contexts require a reconstructed object. Python must
    document that the call blocks and when the GIL is reacquired for progress
    callbacks.
  - Stream one frame and its complex FFT workspace at a time. Retain the mean
    spectrum, one current spectrum, accepted observation records, and the
    small fit Jacobian; expected memory is linear in the image pixels plus the
    product of observation and active-parameter counts, independent of total
    frame count. Use the configured backend for FFTs, but keep the physical fit
    deterministic and do not claim GPU execution while its small dense rank
    calculation remains CPU-only. Preserve canonical frame order for
    reductions so explicit frame permutations with the same source
    observations produce the same fitted result after stable source/frame
    sorting.
  - Make `PlanarArrayInitializationResult` versioned and fully serializable,
    with matching Rust/Python `save_json` and `load_json`. Give it a dedicated
    verified initialization bundle containing the physical state, initialized
    model, options, per-frame observation table, fit history, conditioning,
    runtime, and optional normalized-spectrum preview; do not force it into
    `ReconstructionResult` fields or a generic-correction artifact. No
    `ReconstructionCheckpoint` format change is needed because initialization
    completes before reconstruction and restarts deterministically. A later
    reconstruction checkpoint records the initialized illumination/model as
    its normal physical starting state, while the separate initialization
    artifact retains how that state was obtained.
  - Expose matching Rust builders and keyword-only Python constructors for the
    parameter selection, circle options, explicit frame subset, initializer,
    callback, result, and bundle. Public observations report acquisition frame,
    stable source index, `(kx, ky)` in radians per metre, `(NA_x, NA_y)`,
    Fourier-grid coordinates, fitted radius, scores, confidence, and rejection
    status. Python numerical arrays are owned NumPy values with documented
    `float64` shapes and `(row, column)` ordering; do not expose private `_core`
    types or convert physical results into unnamed tuples.
  - Add deterministic detector tests from independently constructed Fourier
    spectra and canonical simulator tests with textured amplitude, phase, and
    mixed specimens. Cover zero and near-cutoff illumination, conjugate-sign
    ambiguity, rectangular images and unequal `dkx`/`dky`, radius mismatch,
    weak texture, noise, known spatial background and gains, negative corrected
    samples, masks, multiplexed/zero-weight ignored frames, repeated sources,
    explicit selection, acquisition permutation, lazy/resident parity, and
    CPU/backend FFT parity. Add bounded recovery and partial-update regressions
    for translation, rotation, pitch or axial distance with the complementary
    scale fixed, and reference index; reject rank-deficient parameter
    combinations and assert the crate's positive-k/Fourier-crop sign
    convention.
  - Test atomic model refresh, unchanged pupil/intensity calibration, JSON and
    bundle round trips, progress/cancellation boundaries, Rust/Python parity,
    and use of the returned warm start by `Fpie`, `Epry`, and
    `JointReconstruction`. A synthetic recovery test must show that accepted
    initialization reduces both source-vector error and bounded physical
    parameter error; downstream tests must compare cold joint calibration,
    circle initialization alone, and circle initialization followed by the
    same joint-calibration budget.
  - Before promotion, benchmark a declared family of bright-field-rich and
    bright-field-poor planar-array simulations across bounded pose, pitch,
    objective-NA, noise, aberration, and specimen-contrast changes. Report
    detection recall and false acceptance, source-vector and physical-parameter
    error, fit rank/condition, capture range, full forward-model passes, runtime,
    peak memory, downstream objective, and aligned complex-object error.
    Retain the public initializer only if it expands the reliable capture range
    or reduces time to the same downstream error relative to the current cold
    `JointReconstruction`, while rejecting rather than confidently fitting
    unsuitable data.
  - Document the final implementation in Rustdoc, the authoritative Python
    stub, `docs/guides/reconstruction.md`, diagnostics/bundle documentation,
    the glossary, an example, and `CHANGES.md`. State that Sun et al. search
    independent apertures with simulated annealing during reconstruction and
    regress a four-parameter planar misalignment model, whereas this proposal
    performs no reconstructed-object search and fits detected centers directly
    to the crate's bounded physical geometry. State that Eckert et al. combine
    bright-field preprocessing with iterative spectral correlation and support
    several illuminator/3D settings, whereas the first fpm-rs milestone adopts
    only their circular-edge initialization for two-dimensional planar arrays
    and delegates iterative physical refinement to the existing calibrator.
    Run the complete Rust, Python, documentation, citation, bundle, example,
    and synthetic-recovery checks with implementation changes.

  The public bright-field initializer, physical fit, callbacks, JSON and
  verified bundle persistence, Rust/Python surfaces, documentation, analytic
  rectangular-grid detector test, and deterministic translation-recovery test
  are implemented. Keep this item open for the broader adverse-data matrix,
  downstream cold-versus-warm calibration comparisons, and capture-range and
  runtime benchmarks listed above before declaring the initializer promoted.

### APIC complex-field reconstruction

- [ ] **Design first:** Treat aberration-corrected, closed-form complex-field
  reconstruction (APIC) as a separate non-iterative workflow. Define and
  validate its NA-matching and dark-field acquisition requirements, output
  model, tiling, and failure modes. Reference: R. Cao, C. Shen, and C. Yang,
  “High-resolution, large field-of-view label-free imaging via
  aberration-corrected, closed-form complex field reconstruction,” *Nature
  Communications* **15**, 4713 (2024),
  [https://doi.org/10.1038/s41467-024-49126-y](https://doi.org/10.1038/s41467-024-49126-y).

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
    `GlobalGaussNewton` on the same APIC-compatible deterministic simulations,
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
    only with that implementation change. This design change itself adds no
    public API or executable APIC code.

### Multi-wavelength reconstruction

- [ ] **Design first:** Define a multi-wavelength or spectral reconstruction
  architecture for separate or multiplexed measurements. Decide which object,
  pupil, geometry, source-power, and frame-gain parameters may be shared across
  wavelengths instead of assuming shared calibration, and preserve explicit
  wavelength-dependent sampling. Reference: S. Dong, R. Shiradkar, P. Nanda,
  and G. Zheng, “Spectral multiplexing and coherent-state decomposition in
  Fourier ptychographic imaging,” *Biomedical Optics Express* **5**(6),
  1757–1767 (2014),
  [https://doi.org/10.1364/BOE.5.001757](https://doi.org/10.1364/BOE.5.001757).

  **Proposed architecture; implementation pending:**

  - Scope the first implementation to a finite set of narrowband, mutually
    incoherent image-plane FPM channels. Channel `c` has one explicit vacuum
    wavelength and one thin-sample complex transmission field. A measured
    detector frame may contain one channel (separate acquisition) or an
    incoherent intensity sum from several channels (spectral multiplexing).
    Do not include cross-wavelength field interference, continuous spectra,
    finite emitter bandwidth, mixed-state partial coherence, RGB mosaics, or
    a dispersive material/thickness object model in this milestone. The next
    roadmap item owns finite-bandwidth and mixed-state modeling.
  - Represent one detector frame as a sparse list of non-negative intensity
    contributions `(channel, local_frame, spectral_weight)`. A local frame is
    an ordinary channel-specific `AcquisitionPlan` row and may itself contain
    mutually incoherent source contributions. After canonicalizing duplicate
    pairs and removing zero weights, predict
    `p_f = gain_f * sum_(c,l)(spectral_weight[f,c,l] * I_c,l) + background_f`,
    where `I_c,l` is computed by the existing single-wavelength forward model
    and includes that channel's source powers, source weights, and local
    illumination-frame gain. Separate data are the exact special case with one
    spectral contribution per detector frame. Apply the detector-frame gain
    and background once, after the spectral sum; never once per channel.
  - Add experiment-layer `SpectralChannel` values with a stable, unique channel
    ID, `Optics`, `SourceCalibration`, and local `AcquisitionPlan`, plus a
    `SpectralGeometry` choice that is either one shared physical
    `SourceGeometry` or explicit per-channel geometries. Construct and resolve
    one ordinary `Illumination` atomically for every channel. Shared planar or
    direction geometry is resolved separately with each channel's `Optics`, so
    the resulting transverse wave vectors retain their wavelength dependence.
    Reject `KVectorList` as shared physical geometry because its values are
    already wavelength-specific; channel-local `KVectorList` values remain
    valid. Do not add a wavelength override to any geometry type.
  - Compile those inputs into a new algorithm-facing
    `SpectralImagePlaneModel` containing ordered channel IDs, one ordinary
    `ImagePlaneModel` kernel per channel, and the sparse global spectral frame
    plan. The spectral forward evaluator must call the canonical channel
    crop/pupil/FFT operators instead of duplicating their sign, shift, or
    subpixel interpolation logic. Algorithms receive only this compiled model
    and never inspect LED geometry.
  - Require all channels to use the same detector image shape and the same
    object-plane low-resolution pixel pitch in the first implementation.
    Select one common high-resolution reconstruction shape from the union of
    every channel's wavelength-specific crop bounds, or validate one explicit
    common shape against that union. Each compiled channel keeps its own
    wavelength, pupil cutoff, synthetic NA, k-vectors, crops, and fractional
    offsets. Reject configurations that would require detector registration or
    spatial resampling; a later registration design may relax this rule.
  - Make sharing a declared model choice, never an inference from numerically
    equal inputs. Support `Independent` objects by default and an explicit
    `SharedComplex` object coupling for samples whose complex transmission may
    reasonably be treated as wavelength-independent. The independent form
    stores and updates one object spectrum per channel; the shared form stores
    one spectrum and accumulates every channel adjoint into it. Do not expose a
    partially shared amplitude/phase option until a physical dispersive object
    parameterization and its identifiability have been designed.
  - Keep sampled pupils channel-specific in every mode: aperture support,
    defocus phase, and aberration phase are evaluated on wavelength-specific
    sampling. A shared physical NA, defocus distance, or aberration
    parameterization may be used to *construct* those pupils, but must not
    alias their sampled arrays. Blind spectral pupil recovery and shared
    optical-path-difference coefficients are deferred; the first solver uses
    fixed per-channel pupils.
  - Keep source powers channel- and source-specific because illumination
    spectra and detector response need not match across wavelengths. Known
    spectral response belongs in source powers or the sparse spectral weights,
    not in a channel-specific gain applied after multiplexing. Treat those
    quantities as fixed in the first solver. If later calibration makes them
    variable, normalize active source powers to mean one within each channel
    and fix one documented spectral scale reference; otherwise each object's
    amplitude is confounded with its channel power.
  - Define frame gain and background only per physical detector frame. They are
    therefore shared by every channel contribution to a multiplexed exposure,
    while separate exposures naturally have distinct values. Keep them fixed
    in the first solver. Any later recovery must normalize positive frame gains
    to mean one and must not introduce an unidentifiable per-channel gain in a
    multiplexed frame.
  - Reuse the existing scalar `MeasurementRead` contract and its
    `(frame, row, column)` intensity shape. The spectral plan, rather than
    `FrameMetadata.illumination_index`, records channel and local-frame
    membership; multiplexed frames continue to use no single illumination
    index. Require every channel and every declared local frame to participate
    in at least one positive-weight, positive-measurement-weight detector frame.
    Report the channel mixing matrix rank and condition estimate as diagnostics
    without presenting either as a proof of nonlinear recoverability.
  - Add parallel, intentionally separate `SpectralReconstructionProblem`,
    `SpectralReconstructionAlgorithm`, state, runner, and result contracts
    instead of weakening `ReconstructionProblem<ImagePlaneModel>` or changing
    every existing algorithm. The first algorithm is an object-only
    `SpectralAlternatingProjection`: it computes all coherent modes required by
    one detector frame, forms their total predicted intensity, applies one
    measurement-domain amplitude correction, and sends the weighted adjoint
    corrections to the independent or shared object spectra. Stream one global
    frame at a time and retain only that frame's fields, so scratch memory
    scales with its active modes rather than the complete spectral data set.
  - Preserve exact reduction order from the global acquisition schedule.
    Single-channel problems must match ordinary alternating projection, and
    independent objects with entirely separate frames must match corresponding
    per-channel runs under equivalent schedules. Reject current
    `JointReconstruction`, blind-pupil algorithms, generic k-vector correction,
    and physical calibration for spectral problems until each has an explicit
    rule for shared parameters, wavelength scaling, gauge normalization, and
    checkpoint transport.
  - Return channel-ordered complex objects, amplitudes, phases, spectra, and
    pupils together with channel IDs and wavelengths. Rust should use a
    channel-record result so metadata cannot become detached from an array;
    Python should expose stacked `(channels, height, width)` copies plus
    `channel_ids`, `wavelengths_vacuum_m`, and the object-coupling mode. For a
    shared object, each Python channel plane is the same final field by
    contract. Document the independent global phase piston of every
    independent channel, or the single piston of a shared object, when
    comparing ground truth.
  - Add an explicitly tagged spectral checkpoint payload and bump the
    checkpoint format rather than forcing multiple objects through the current
    single-object auxiliary slot. Persist channel IDs and order, wavelengths,
    object coupling, common grid, sparse spectral plan, independent/shared
    object state, per-channel pupils, global gains/background, algorithm state,
    trace, and RNG/schedule state. Reject ordinary/spectral checkpoint
    interchange and any resume whose channel order or model-defining spectral
    metadata changed.
  - Extend `dataset_spec` only through a new schema version with explicit
    `spectral_channels` and sparse `spectral_acquisition` records. Keep stored
    measurements as grayscale detector frames; do not interpret RGB files as
    calibrated spectral channels. Existing single-wavelength bundles and the
    offline behavior of `DatasetLoader` remain unchanged. A converter must
    supply stable channel IDs, wavelengths, response/weight provenance, frame
    ordering, and whether each exposure is separate or multiplexed.
  - Expose matching Rust constructors/builders and keyword-only Python APIs for
    channels, the spectral plan, object coupling, model compilation, the
    problem, solver, checkpoint, and result. Provide explicit `separate` and
    sparse `multiplexed` plan constructors so the user chooses acquisition
    semantics rather than relying on shape inference. Validate unique IDs,
    finite positive and distinct wavelengths, finite non-negative weights,
    valid channel/local-frame references, non-empty canonical rows, common
    sampling/grid requirements, and structural channel coverage with stable
    parameter names.
  - Add deterministic tests for wavelength-dependent k-vector and pupil
    sampling, common-grid union sizing, single-channel parity, separate-run
    equivalence, manual incoherent spectral sums, gain/background placement,
    independent and shared object updates, channel permutations, invalid or
    uncovered plan entries, shared-`KVectorList` rejection, checkpoint resume,
    dataset round trips, and Rust/Python parity. Include a bounded synthetic
    R/G/B multiplexed recovery inspired by Dong et al., while documenting that
    this architecture generalizes their demonstrated state decomposition into
    typed channel/local-frame composition and does not model finite spectral
    bandwidth.
  - Document the implementation location and public API in Rustdoc, the
    authoritative Python stubs, `docs/guides/reconstruction.md`, the core
    concepts/glossary, dataset documentation, and `CHANGES.md`, with the full
    Dong et al. reference at the closest explanation. Before promotion,
    compare separate spectral runs with ordinary per-channel reconstruction
    and evaluate multiplexed runtime, peak memory, channel-wise gauge-aligned
    error, spectral cross-talk, and noise sensitivity against the same total
    acquisition and update budgets. Run the complete source, documentation,
    citation, Python, checkpoint, example, and notebook checks only during the
    later implementation change.

### Partial coherence and spectral bandwidth

- [ ] **Design first:** Investigate mixed-state partial spatial coherence and
  finite spectral-bandwidth models. Treat this as a substantial forward-model,
  measurement, and identifiability project rather than a bounded algorithm
  option. Define the coherence representation, state-count selection, memory
  scaling, calibration gauges, and relationship to incoherent coded-source
  multiplexing before implementation. Dong et al. (2014), cited above, provides
  an initial state-decomposition reference but does not by itself define the
  crate’s general coherence model.

## Dataset format and loading

### Lazy large-dataset loading

- [ ] Add a lazy `dataset_spec` loading path for frame folders and large TIFF
  stacks without weakening bundle validation.

### Malformed-bundle coverage

- [ ] Expand generic bundle tests for malformed manifests, inconsistent frame
  counts and shapes, invalid illumination metadata, and multiplexed datasets.

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
  [https://doi.org/10.1364/OE.485370](https://doi.org/10.1364/OE.485370).
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
  [https://doi.org/10.1016/j.jsb.2013.05.004](https://doi.org/10.1016/j.jsb.2013.05.004),
  and M. van Heel and M. Schatz, “Fourier shell correlation threshold
  criteria,” *Journal of Structural Biology* **151**(3), 250–262 (2005),
  [https://doi.org/10.1016/j.jsb.2005.05.009](https://doi.org/10.1016/j.jsb.2005.05.009).

### Subpixel complex-field registration

- [ ] **Design first:** Add explicit optional subpixel translation registration
  for complex-field comparison. Specify circular versus non-periodic boundaries,
  interpolation, mask overlap, allocation, fitted complex gain, and returned
  shift. Document that translation is not a universal ambiguity with fixed
  detector coordinates and a fixed pupil, while blind object/pupil recovery can
  retain a coupled affine-phase ambiguity. Reference the registration method to
  M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, “Efficient subpixel image
  registration algorithms,” *Optics Letters* **33**(2), 156–158 (2008),
  [https://doi.org/10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156).

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

- [ ] Add loss-selection guidance to `docs/guides/reconstruction.md`. Explain
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

### CUDA kernels

- [ ] Implement a CUDA backend with cuFFT and resident kernels for Fourier
  crops, pupil operations, intensity formation, projection, and reductions.

### Device-resident reconstruction state

- [ ] Keep reconstruction state and scratch buffers device-resident across AP,
  FPIE, EPRY, ADMM, and gradient-descent updates.

### CPU and GPU validation

- [ ] Add CPU/GPU numerical-parity, unsupported-device, performance, and memory
  tests.
