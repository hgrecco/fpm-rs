# Installation

## Supported systems

The project tests stable Rust and CPython 3.12/3.13/3.14 on Linux, macOS, and
Windows. The minimum supported Rust version is 1.97. The implementation is
CPU-only; there is no CUDA, ROCm, Metal, or other GPU backend to install.

Python requires NumPy 2.1 or newer. Building the extension from source requires
a C/Rust linker suitable for the platform, Rust/Cargo, and the maturin build
backend. There are no external FFT, BLAS, or system imaging libraries to
install.

## Python package from PyPI

Create and activate a Python 3.12, 3.13, or 3.14 virtual environment, then run:

```sh
python -m pip install --upgrade pip
python -m pip install fpm-rs
```

This installs the [published `fpm-rs` package](https://pypi.org/project/fpm-rs/)
and its Rust-backed `fpm_rs._core` extension. On a supported wheel platform,
Rust, Cargo, and a native linker are not required.

To include plotting and notebook tools:

```sh
python -m pip install "fpm-rs[notebook]"
```

Use `"fpm-rs[plot]"` when only Matplotlib plotting helpers are needed.

## Build the Python package from a checkout

To build an unreleased revision or a wheel for a platform that PyPI does not
provide, install Rust 1.97 or newer and the platform's normal native build
tools. From a repository checkout, run:

```sh
python -m pip install .
```

This invokes maturin using `pyproject.toml`, compiles `python/Cargo.toml` as the
private `fpm_rs._core` extension, and installs the Python sources from
`python/fpm_rs`.

## Pixi development installation

Pixi provides the reproducible development toolchain:

```sh
pixi install -e py312
pixi run -e py312 python-test
```

`python-develop` runs `maturin develop --skip-install`; the activation
environment places `python/` on `PYTHONPATH`. Use the `py314` environment to
validate CPython 3.14.

## Rust library

Applications can use the public `fpm-rs` crate from a local checkout:

```toml
[dependencies]
fpm-rs = { path = "../fpm-rs" }
```

Then verify it with:

```sh
cargo run --example simulate_and_reconstruct
```

The `fpm-rs-python` workspace member is an implementation detail of the Python
package, not a supported Rust integration crate.

## Verify Python

```sh
python -c "import fpm_rs; print(fpm_rs.__version__)"
```

For a functional smoke test, run the example on the [Quickstart](quickstart.md).

## Common installation problems

- **Python is too old:** use CPython 3.12, 3.13, or 3.14; `pyproject.toml` rejects older
  interpreters.
- **Cargo or a linker is missing:** a supported PyPI wheel avoids a local build.
  Otherwise, install Rust and the platform's normal native build tools, then run
  `pip install .` from a checkout.
- **An in-tree import finds no `_core` module:** run `pixi run -e py312
  python-develop` or install the package into the active environment.
- **Plots fail to import:** install `"fpm-rs[plot]"` or `"fpm-rs[notebook]"`.
- **A wheel is unavailable for the platform:** install Rust and build from
  source; the supported implementation does not require a GPU SDK.

See [Troubleshooting](../troubleshooting.md) for runtime and documentation-build
issues.
