import os

import torch
from setuptools import find_packages, setup

import torch_musa  # noqa: F401
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

_ROOT = os.path.dirname(os.path.abspath(__file__))

if not (os.environ.get("MSDA_CROSS_BUILD") == "1" and os.environ.get("TORCH_MUSA_ARCH") == "31" and os.environ.get("TORCH_MUSA_ARCH_LIST") == "31") and (not hasattr(torch, "musa") or not torch.musa.is_available()):
    raise RuntimeError("MUSA is required to build msda_musa_opt2.")

setup(
    name="msda_musa_opt2",
    version="0.1.0",
    packages=find_packages(include=("msda_musa_opt2", "msda_musa_opt2.*")),
    ext_modules=[
        MUSAExtension(
            name="_msda_opt2_musa",
            # NOTE: editable/pyproject builds require relative source paths.
            sources=[os.path.join("..", "csrc_musa", "msda", "binding_musa.mu")],
            extra_include_paths=[os.path.join("..", "csrc_musa", "msda")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

