#!/usr/bin/env python3
"""Utility to build the project's Rust crates for the active Python environment.

This script provides a cross-platform interface for compiling the Rust
components used by the project. It can invoke :command:`cargo` directly or
delegate to ``maturin develop``. Multiple crates may be supplied and, by
default, both ``nlhe_eval`` and ``rs_engine`` are built and installed into the
active Python environment's ``site-packages`` directory.

Examples
--------

.. code-block:: bash

    # Build all crates using cargo (default)
    python nlhe/build_rust.py

    # Build only nlhe_eval using maturin in release mode
    python nlhe/build_rust.py --backend maturin --crate-dir nlhe_eval --release
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
import tomllib


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run *cmd* in a subprocess and echo the command to the console."""

    print("$", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=cwd)


def _ensure_maturin() -> None:
    """Ensure that ``maturin`` is installed in the current environment."""

    if importlib.util.find_spec("maturin") is None:
        print("maturin not found, installing...")
        _run([sys.executable, "-m", "pip", "install", "maturin"])


def _parse_crate_name(crate_dir: Path) -> str:
    """Return the package name from the crate's ``Cargo.toml``."""

    with open(crate_dir / "Cargo.toml", "rb") as f:
        cargo_toml = tomllib.load(f)
    return cargo_toml["package"]["name"]


def _find_artifact(crate_dir: Path, crate_name: str, release: bool) -> Path:
    """Locate the built dynamic library produced by ``cargo build``."""

    build_type = "release" if release else "debug"
    target_dir = crate_dir / "target" / build_type
    search_dirs = [target_dir, target_dir / "deps"]

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    candidates: list[Path] = []
    patterns = []
    if ext_suffix:
        patterns.append(f"{crate_name}*{ext_suffix}")
    if os.name == "nt":
        patterns.extend([f"{crate_name}*.pyd", f"{crate_name}*.dll"])
    elif sys.platform == "darwin":
        patterns.extend([f"lib{crate_name}*.dylib", f"{crate_name}*.so"])
    else:
        patterns.extend([f"lib{crate_name}*.so", f"{crate_name}*.so"])

    for directory in search_dirs:
        for pattern in patterns:
            candidates.extend(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"unable to locate built artifact for {crate_name} in {target_dir}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_with_cargo(crate_dir: Path, crate_name: str, release: bool) -> None:
    """Build the crate using :command:`cargo` and copy the artifact."""

    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    _run(cmd, cwd=crate_dir)

    artifact = _find_artifact(crate_dir, crate_name, release)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if os.name == "nt" else ".so"
    )
    site_packages = Path(sysconfig.get_paths()["platlib"])
    dest = site_packages / f"{crate_name}{ext_suffix}"
    shutil.copy2(artifact, dest)
    print(f"Copied {artifact} -> {dest}")


def _build_with_maturin(crate_dir: Path, crate_name: str, release: bool) -> None:
    """Build the crate using ``maturin develop`` and patch ``__init__``.

    On Windows ``maturin`` generates an ``__init__`` that references the
    extension module without importing it, leading to ``NameError`` during
    runtime.  After ``maturin develop`` finishes we rewrite the package's
    ``__init__`` to import the module explicitly so the attribute lookup
    succeeds consistently across platforms.
    """

    cmd = [sys.executable, "-m", "maturin", "develop"]
    if release:
        cmd.append("--release")
    cmd.extend(["-m", str(crate_dir / "Cargo.toml")])
    _run(cmd, cwd=crate_dir)

    site_packages = Path(sysconfig.get_paths()["platlib"])
    pkg_dir = site_packages / crate_name
    init_py = pkg_dir / "__init__.py"
    if init_py.exists():
        init_py.write_text(
            f"from . import {crate_name} as _lib\n"
            f"from .{crate_name} import *\n"
            f"__all__ = getattr(_lib, '__all__', [])\n"
            f"__doc__ = _lib.__doc__\n"
        )


def _verify_import(crate_name: str) -> None:
    """Verify the built crate can be imported using different mechanisms."""

    try:
        __import__(crate_name)
        importlib.import_module(crate_name)
    except Exception as exc:  # pragma: no cover - diagnostic
        suggestion = _diagnose_import_failure(exc)
        raise SystemExit(
            f"Built artifact but failed to import {crate_name}: {exc}\n{suggestion}"
        ) from exc
    else:
        print(f"Successfully built and imported {crate_name}")


def _diagnose_import_failure(exc: Exception) -> str:
    """Return a human-readable explanation for *exc* with suggestions."""

    if isinstance(exc, ModuleNotFoundError):
        return (
            "The module was not found. Ensure the artifact was copied into the "
            "active environment's site-packages and that PYTHONPATH is set correctly."
        )
    if isinstance(exc, ImportError):
        return (
            "The module exists but failed to load. Verify it was built for the "
            "current Python version and that all required dependencies are available."
        )
    if isinstance(exc, OSError):
        return (
            "A platform-specific error occurred while loading the module. "
            "Check for missing system libraries or incompatible architecture."
        )
    return "Unknown import error; inspect the traceback above for details."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Rust crates for the project",
    )
    parser.add_argument(
        "--crate-dir",
        action="append",
        help=(
            "Path to a crate directory relative to this file. "
            "May be specified multiple times. Defaults to building nlhe_eval "
            "and rs_engine."
        ),
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build in release mode",
    )
    parser.add_argument(
        "--backend",
        choices=["cargo", "maturin"],
        default="cargo",
        help="Build backend to use",
    )
    args = parser.parse_args()

    crate_dirs = args.crate_dir or ["nlhe/nlhe_eval", "nlhe/rs_engine"]

    if args.backend == "maturin":
        _ensure_maturin()

    for crate_dir_str in crate_dirs:
        crate_dir = (Path(__file__).resolve().parent / crate_dir_str).resolve()
        if not (crate_dir / "Cargo.toml").exists():
            raise SystemExit(f"No Cargo.toml found in {crate_dir}")

        crate_name = _parse_crate_name(crate_dir)

        if args.backend == "maturin":
            _build_with_maturin(crate_dir, crate_name, args.release)
        else:
            _build_with_cargo(crate_dir, crate_name, args.release)

        _verify_import(crate_name)


if __name__ == "__main__":
    main()
