import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

target_dir = 'interformer/cpp_extension/fast_moe'
os.makedirs(target_dir, exist_ok=True)

setup(
    name='fast_moe',
    ext_modules=[
        CppExtension(
            'interformer.cpp_extension.fast_moe._fast_moe_linear',
            sources=['fast_moe_linear.cpp'],
            extra_compile_args=[
                '-O3',
                '-march=native',
                '-fopenmp',
                '-DUSE_OPENMP'
            ],
            extra_link_args=['-lgomp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(
        build_lib='interformer/cpp_extension/fast_moe',
        no_python_abi_suffix=True
    )},
    packages=['interformer.cpp_extension.fast_moe'],
    package_dir={
        'interformer.cpp_extension.fast_moe': '.'
    },
    zip_safe=False,
)
