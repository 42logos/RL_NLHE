#!/usr/bin/env python3
"""Build and install Rust extension crates across platforms.

This pure-Python helper mirrors the behaviour of the Windows-only
PowerShell script but works on Linux, macOS and Windows.  It can compile
both the ``nlhe_eval`` hand evaluator and the ``rs_engine`` backend (or
any similar ``cdylib`` crate) and install the resulting module into the
active Python environment using ``maturin`` or plain ``cargo``.

Examples::

    # build the hand evaluator
    python nlhe/build_rust.py --use-maturin --crate-dir nlhe_eval

    # build the engine backend
    python nlhe/build_rust.py --use-maturin --crate-dir rs_engine

Run ``python nlhe/build_rust.py --help`` for options.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

try:
    import tomllib
except ImportError:
    # For Python < 3.11
    import tomli as tomllib


def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess, echoing the command."""
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def ensure_cmd(name: str) -> bool:
    """Return True if command *name* is available on PATH."""
    return shutil.which(name) is not None


def resolve_python(venv: Path) -> str:
    """Resolve the Python interpreter within *venv* if it exists."""
    if os.name == "nt":
        candidate = venv / "Scripts" / "python.exe"
    else:
        candidate = venv / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    print(f"Warning: {candidate} not found, using current interpreter", file=sys.stderr)
    return sys.executable


def get_crate_name(crate: Path) -> str:
    """Read the Cargo package name for *crate*."""
    with (crate / "Cargo.toml").open("rb") as fh:
        data = tomllib.load(fh)
    return data["package"]["name"]


def build_with_maturin(py: str, crate: Path, module: str) -> None:
    if not ensure_cmd("maturin"):
        print("maturin not found; installing via pip", file=sys.stderr)
        run([py, "-m", "pip", "install", "maturin"])
    run([py, "-m", "maturin", "develop", "--release", "-m", str(crate / "Cargo.toml")])
    run([py, "-c", f"import {module},sys;print('{module} imported', {module}.__file__)"])


def build_with_cargo(py: str, crate: Path, module: str) -> None:
    if not ensure_cmd("cargo"):
        raise SystemExit("cargo not found; install Rust toolchain")
    run(["cargo", "build", "--release"], cwd=crate)
    ext = {
        "win32": ".dll",
        "cygwin": ".dll", 
        "msys": ".dll",
        "darwin": ".dylib",
    }.get(sys.platform, ".so")
    target_dir = crate / "target" / "release"
    artifact: Path | None = None
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.startswith(module) and f.endswith(ext):
                artifact = Path(root) / f
    if not artifact:
        raise SystemExit(f"built artifact not found in {target_dir}")
    site = sysconfig.get_paths().get("platlib", sysconfig.get_paths()["purelib"])
    # Rename .dll to .pyd for Windows Python extensions
    dest_ext = ".pyd" if sys.platform in ("win32", "cygwin", "msys") else ext
    dest = Path(site) / (module + dest_ext)
    shutil.copy2(artifact, dest)
    print(f"installed {dest}")
    run([py, "-c", f"import {module},sys;print('{module} imported', {module}.__file__)"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=".venv", help="virtual environment path")
    parser.add_argument("--crate-dir", default="both", help="Rust crate directory")
    parser.add_argument(
        "--module-name",
        help="Python module name (defaults to Cargo package name)",
    )
    parser.add_argument("--use-maturin", default=True, action="store_true", help="build via maturin develop")
    args = parser.parse_args()

    if args.crate_dir == "both":
        for dir in ("nlhe_eval", "rs_engine"):
            print(f"Building {dir}")
            args.crate_dir = dir
            repo_root = Path(__file__).resolve().parent
            crate = repo_root / 'nlhe' / args.crate_dir
            if not crate.exists():
                raise SystemExit(f"crate directory {crate!r} not found")
            module = args.module_name or get_crate_name(crate)
            py = resolve_python(repo_root / args.venv)

            if args.use_maturin:
                build_with_maturin(py, crate, module)
            else:
                build_with_cargo(py, crate, module)
        return
    
    repo_root = Path(__file__).resolve().parent
    crate = repo_root / 'nlhe' / args.crate_dir
    if not crate.exists():
        raise SystemExit(f"crate directory {crate!r} not found")
    module = args.module_name or get_crate_name(crate)
    py = resolve_python(repo_root / args.venv) 

    if args.use_maturin:
        build_with_maturin(py, crate, module)
    else:
        build_with_cargo(py, crate, module)


if __name__ == "__main__":
    main()
