from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

cuda_source = glob.glob("**/*.cu", recursive=True)
cpp_source = glob.glob("**/*.cpp", recursive=True)

setup(
    name="matris_op",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "matris_op",
            cuda_source + cpp_source,
            extra_compile_args={"nvcc": ["-O3","-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

#python setup.py build develop