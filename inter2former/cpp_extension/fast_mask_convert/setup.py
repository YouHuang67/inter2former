import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

target_dir = 'inter2former/cpp_extension/fast_moe'
os.makedirs(target_dir, exist_ok=True)

setup(
    name='fast_mask_convert',
    ext_modules=[
        CppExtension(
            'inter2former.cpp_extension.fast_mask_convert._fast_mask_convert',
            sources=['fast_mask_convert.cpp'],
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
        build_lib='inter2former/cpp_extension/fast_mask_convert',
        no_python_abi_suffix=True
    )},
    packages=['inter2former.cpp_extension.fast_mask_convert'],
    package_dir={
        'inter2former.cpp_extension.fast_mask_convert': '.'
    },
    zip_safe=False,
)
