# Rust API

Native rustdoc is generated only for the public `fpm-rs` crate. It exposes the
physical experiment descriptions, compiled image-plane model, resident and lazy
measurements, reconstruction algorithms and runner, callbacks, diagnostics,
metrics, simulation, datasets, and CPU backend boundary.

The Rust API is intended for lower-level integrations, new algorithm work,
custom measurement storage or callbacks, and performance-sensitive use. The
workspace's `fpm-rs-python` crate is a private PyO3 adapter and is deliberately
excluded from the published rustdoc site.

<a href="../../rust-api/">Open the generated <code>fpm-rs</code> Rust API.</a>

The combined documentation build runs:

```sh
cargo doc --package fpm-rs --no-deps --locked
```

The result is copied under `site/rust-api/`, so rustdoc assets and links remain
relative and work beneath a GitHub Pages repository subpath.
