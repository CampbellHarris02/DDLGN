# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import pybind11
import platform
import sys
import os
import torch
from pathlib import Path

# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
PACKAGE_NAME = "difflogic"
VERSION = "0.1.0"
AUTHOR = "Casper Harris"
DESCRIPTION = "Differentiable logic gate networks with CUDA and Metal (MPS) backends"

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# ------------------------------------------------------------------------------
# Determine build options
# ------------------------------------------------------------------------------
USE_CUDA = (
    torch.cuda.is_available()
    and os.getenv("FORCE_CUDA", "1") != "0"
    and os.getenv("FORCE_CPU_ONLY", "0") == "0"
)

define_macros = [
    ("TORCH_API_INCLUDE_EXTENSION_H", None),
    ("_GLIBCXX_USE_CXX11_ABI", "0" if not torch._C._GLIBCXX_USE_CXX11_ABI else "1"),
]

extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-Wall",
        "-Wno-unused-variable",
        "-fvisibility=hidden",
    ]
}

if sys.platform == "darwin":
    extra_compile_args["cxx"] += ["-mmacosx-version-min=11.0"]
    if platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]

if sys.platform == "win32":
    extra_compile_args["cxx"] = ["/std:c++17", "/permissive-", "/w"]

include_dirs = [
    pybind11.get_include(),
    *torch.utils.cpp_extension.include_paths(),
]

# ------------------------------------------------------------------------------
# Extension modules
# ------------------------------------------------------------------------------
ext_modules = []

# Metal (MPS) backend (for macOS)
ext_modules.append(
    CppExtension(
        name="difflogic_metal",
        sources=["difflogic/metal/difflogic_metal.cpp"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
)

# CUDA backend (if available)
if USE_CUDA:
    print("🔧 Building with CUDA support")
    ext_modules.append(
        CUDAExtension(
            name="difflogic_cuda",
            sources=[
                "difflogic/cuda/difflogic.cpp",
                "difflogic/cuda/difflogic_kernel.cu",
            ],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args={"nvcc": ["-lineinfo"], **extra_compile_args},
        )
    )
else:
    print("⚠️ CUDA not detected or disabled; skipping CUDA build")

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/difflogic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    package_dir={"difflogic": "difflogic"},
    packages=["difflogic"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pybind11",
    ],
    zip_safe=False,
)
