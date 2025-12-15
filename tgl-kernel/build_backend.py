"""
Copyright (c) 2025 by TGL team.

Custom build backend for TGL Kernel with AOT compilation support.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import shutil
from pathlib import Path

from build_utils import get_git_version
from setuptools import build_meta as orig

_root = Path(__file__).parent.resolve()
_data_dir = _root / "tgl_kernel" / "data"


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = _root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("TGL_KERNEL_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=_root)

    # Append local version suffix if available
    local_version = os.environ.get("TGL_KERNEL_LOCAL_VERSION")
    if local_version:
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "tgl_kernel"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for tgl_kernel package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def write_if_different(path: Path, content: str) -> None:
    """Write content to file only if it differs."""
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _create_data_dir(use_symlinks=True):
    """Create data directory with CUDA sources."""
    _data_dir.mkdir(parents=True, exist_ok=True)

    def ln(source: str, target: str) -> None:
        src = _root / source
        dst = _data_dir / target
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if use_symlinks:
            dst.symlink_to(src, target_is_directory=True)
        else:
            # For wheel/sdist, copy actual files instead of symlinks
            if src.exists():
                shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)

    ln("tgl_kernel/csrc", "csrc")
    # AOT directory will be created during build if AOT compilation is enabled
    if (_root / "tgl_kernel" / "data" / "aot").exists():
        ln("tgl_kernel/data/aot", "aot")


def _prepare_for_wheel():
    """Prepare for wheel build (copy files, optionally AOT compile)."""
    # For wheel, copy actual files instead of symlinks so they are included in the wheel
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)

    # AOT compilation (optional, controlled by environment variable)
    if os.environ.get("TGL_KERNEL_AOT_COMPILE", "0") == "1":
        print("AOT compilation enabled...")
        try:
            # Import here to avoid circular dependency
            import sys

            sys.path.insert(0, str(_root))
            from tgl_kernel.aot.build import compile_all_modules

            aot_output_dir = _data_dir / "aot"
            aot_output_dir.mkdir(parents=True, exist_ok=True)

            compile_all_modules(output_dir=aot_output_dir, verbose=True)
            print("AOT compilation completed successfully")
        except Exception as e:
            print(f"Warning: AOT compilation failed: {e}")
            print("Wheel will use JIT compilation at runtime")


def _prepare_for_editable():
    """Prepare for editable install (use symlinks)."""
    # For editable install, use symlinks so changes are reflected immediately
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=True)


def _prepare_for_sdist():
    """Prepare for source distribution."""
    # For sdist, copy actual files instead of symlinks so they are included in the tarball
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)


def get_requires_for_build_wheel(config_settings=None):
    """Get requirements for building wheel."""
    print("=" * 80)
    print("TGL Kernel: Preparing for wheel build...")
    aot_enabled = os.environ.get("TGL_KERNEL_AOT_COMPILE", "0")
    print(f"TGL_KERNEL_AOT_COMPILE = {aot_enabled}")
    print("=" * 80)
    _prepare_for_wheel()
    return []


def get_requires_for_build_sdist(config_settings=None):
    """Get requirements for building sdist."""
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    """Get requirements for editable install."""
    _prepare_for_editable()
    return []


# Forward other build_meta functions
build_wheel = orig.build_wheel
build_sdist = orig.build_sdist
build_editable = orig.build_editable
prepare_metadata_for_build_wheel = orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = orig.prepare_metadata_for_build_editable
