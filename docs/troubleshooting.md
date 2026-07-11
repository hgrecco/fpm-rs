# Troubleshooting

## Shape or model mismatch

Use `(frames, height, width)` for measurements and `(height, width)` everywhere
else. The stack's frame count must equal `model.frame_count`; for multiplexed
illumination this is not necessarily `model.source_count`. Keep acquisition
frame order aligned with the compiled illumination order.

## Non-finite or stalled reconstruction

Confirm measurements are finite non-negative intensities, masks and frame
weights have supported shapes, optical distances use metres, and the
reconstruction has overlapping Fourier coverage. Start with
`AlternatingProjection` defaults and a `DiagnosticRecorder("basic")` before
adding pupil recovery, calibration, or regularization.

## Python import or plotting failures

If `fpm_rs._core` is missing in a checkout, build the extension with
`pixi run -e py313 python-develop` or install the package. Install the `plot`
extra for Matplotlib helpers. The supported interpreters are CPython 3.13 and
3.14.

## Documentation build failures

Run `pixi run docs-build` from the repository root. The task first builds the
extension and verifies that `fpm_rs` imports. Strict MkDocs errors usually mean
a broken relative link, a page missing from navigation, or an API object that
is not in the public typed surface. Rustdoc failures use an isolated target at
`target/docs-rust/`.

`docs-serve` omits live rustdoc rebuilding. Use `pixi run docs-preview` when
checking the final Rust API link and Pages layout.

## Dataset access

`DatasetLoader` expects an already-converted directory containing
`dataset.json`. It never downloads or converts input. See [Datasets](datasets.md)
for the bundle layout and validation rules.
