"""
Build & install:
    python setup.py build_ext --inplace     # compile C extension
    pip install -e . --no-build-isolation   # editable install

Supports both NVIDIA CUDA and Moore Threads MUSA backends.
On MUSA platform, torchada automatically translates CUDA sources to MUSA
at compile time — no manual codegen needed.
"""
import os
from setuptools import setup

# ---------------------------------------------------------------------------
# On MUSA platform, import torchada BEFORE torch.utils.cpp_extension
# so that CUDAExtension / BuildExtension are automatically patched.
# ---------------------------------------------------------------------------
_is_musa = False
try:
    import torch
    if hasattr(torch.version, "musa") and torch.version.musa is not None:
        import torchada  # noqa: F401
        _is_musa = True
except ImportError:
    pass

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# ---------------------------------------------------------------------------
# Extension module — always use standard CUDA sources (.cu / .cuh).
# torchada's BuildExtension handles CUDA→MUSA porting transparently.
# ---------------------------------------------------------------------------
ext_modules = [
    CUDAExtension(
        '_modulated_deform_conv_ext',
        sources=[
            'csrc/binding.cpp',
            'csrc/modulated_deform_conv_cuda.cu',
            'csrc/dcn_fused_nhwc_cuda.cu',
        ],
        include_dirs=[os.path.abspath('csrc')],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O2'],
            'mcc': ['-O2'],
        },
    ),
]


if _is_musa:
    _BaseBuildExt = BuildExtension

    class BuildExtensionFixPaths(_BaseBuildExt):
        """Workaround for torchada converting source paths to absolute.

        setuptools editable wheel builds reject absolute paths.
        After torchada's run() does the CUDA→MUSA porting, we convert
        all source paths back to relative (from setup.py directory).
        """

        def run(self):
            super().run()
            setup_dir = os.path.abspath(os.path.dirname(__file__))
            for ext in self.extensions:
                ext.sources = [
                    os.path.relpath(s, setup_dir) if os.path.isabs(s) else s
                    for s in ext.sources
                ]

    cmdclass = {'build_ext': BuildExtensionFixPaths}
else:
    cmdclass = {'build_ext': BuildExtension}


setup(
    name='modulated_deform_conv',
    version='0.3.0',
    packages=['modulated_deform_conv', 'modulated_deform_conv.perf'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
