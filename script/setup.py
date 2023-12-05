import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def main() -> None:
    wd: str = os.path.dirname(__file__)

    shutil.rmtree(os.path.join(wd, 'build/'), True)

    torch_extension_name: str = 'pte'
    sources: list[str] = [os.path.join(wd, 'src/main.cu')]
    include_dirs: list[str] = [os.path.join(wd, 'include/')]
    compile_args: dict[str, list[str]] = {'cxx': ['-O2'], 'nvcc': ['-O2']}

    setup(
        name=torch_extension_name,
        ext_modules=[
            CUDAExtension(
                name=torch_extension_name,
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=compile_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

    shutil.copy(os.path.join(wd, f'./build/lib.linux-x86_64-cpython-310/'
                                 f'{torch_extension_name}.cpython-310-x86_64-linux-gnu.so'),
                os.path.join(wd, f'./build/'
                                 f'{torch_extension_name}.cpython-310-x86_64-linux-gnu.so'))


if __name__ == '__main__':
    main()
