# Copyright 2013-2020 The Meson development team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import re
import typing as T

from .. import mlog
from .. import mesonlib

from .base import DependencyMethods, SystemDependency
from .cmake import CMakeDependency
from .detect import packages
from .factory import DependencyFactory
from .pkgconfig import PkgConfigDependency

if T.TYPE_CHECKING:
    from ..environment import Environment

"""
TODO: how to select BLAS interface layer (LP64, ILP64)?
      LP64 is the 32-bit interface (confusingly, despite the "64")
      ILP64 is the 64-bit interface

      OpenBLAS library names possible:
          - libopenblas
          - libopenblas64_  # or nonstandard suffix (see OpenBLAS Makefile) - let's not deal with that

TODO: do we care here about whether OpenBLAS is compiled with pthreads or OpenMP?
      MKL has separate .pc files for this (_seq vs. _iomp), OpenBLAS does not

TODO: what about blas/lapack vs. cblas/clapack?
      See http://nicolas.limare.net/pro/notes/2014/10/31_cblas_clapack_lapacke/ for details

NOTE: we ignore static libraries for now

Other Notes:

- OpenBLAS can be built with NOFORTRAN, in that case it's CBLAS + f2c'd LAPACK
- OpenBLAS can be built without LAPACK support, Arch Linux currently does this
  (see https://github.com/scipy/scipy/issues/17465)
- OpenBLAS library can be renamed with an option in its Makefile
- Build options:
    - conda-forge: https://github.com/conda-forge/openblas-feedstock/blob/49ca08fc9d1ff220804aa9b894b9a6fe5db45057/recipe/conda_build_config.yaml
    - Spack: https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/openblas/package.py#L52
    - vendored into NumPy/SciPy wheels: https://github.com/MacPython/openblas-libs/blob/master/tools/build_openblas.sh#L53

Conda-forge library names and pkg-config output
-----------------------------------------------

(lp64) $ ls lib/libopenblas*
lib/libopenblas.a  lib/libopenblasp-r0.3.21.a  lib/libopenblasp-r0.3.21.so  lib/libopenblas.so  lib/libopenblas.so.0
(lp64) $ ls include/
cblas.h  f77blas.h  lapacke_config.h  lapacke.h  lapacke_mangling.h  lapacke_utils.h  lapack.h  openblas_config.h

(ilp64) $ ls lib/libopenblas*
lib/libopenblas64_.a  lib/libopenblas64_p-r0.3.21.a  lib/libopenblas64_p-r0.3.21.so  lib/libopenblas64_.so  lib/libopenblas64_.so.0
(ilp64) $ ls include/
cblas.h  f77blas.h  lapacke_config.h  lapacke.h  lapacke_mangling.h  lapacke_utils.h  lapack.h  openblas_config.h

(lp64) $ pkg-config --cflags openblas
-I/path/to/env/include
(lp64) $ pkg-config --libs openblas
-L/path/to/env/lib -lopenblas
(ilp64) $ pkg-config --cflags openblas
-I/path/to/env/include
ilp64) $ pkg-config --libs openblas
-L/path/to/envlib -lopenblas


LP64 vs ILP64 interface and symbol names
----------------------------------------

Relevant discussions:
- For OpenBLAS, see https://github.com/xianyi/OpenBLAS/issues/646 for standardized agreement on shared library name and symbol suffix.
- PRs that added support for ILP64 to `numpy.distutils`:
    - with `64_` symbol suffix: https://github.com/numpy/numpy/pull/15012
    - generalized to non-suffixed build: https://github.com/numpy/numpy/pull/15069
- PR that added support for ILP64 to SciPy: https://github.com/scipy/scipy/pull/11302
- Note to self: when SciPy uses ILP64, it also still requires LP64, because not
  all submodules support ILP64. Not for the same extensions though.

$ objdump -t lp64/lib/libopenblas.so | grep -E "scopy*"  # output cleaned up
cblas_scopy
scopy_

$ objdump -t ilp64/lib/libopenblas64_.so | grep -E "scopy*"
cblas_scopy64_
scopy_64_

What should be implemented in Meson, and what should be left to users? Thoughts:

1. Meson should support a keyword to select the desired interface (`interface :
   'ilp64'`), defaulting to `'lp64'` because that is what reference BLAS/LAPACK
   provide and what is typically expected.
2. The dependency object returned by `dependency('openblas')` or similar should
   be query-able for what the interface is.
3. For OpenBLAS, Meson should look for `libopenblas64_.so` for ILP64.
4. For Fortran, should Meson automatically set the required compile flag
   `-fdefault-integer-8`?
     - Note: this flag is specific to gfortran and Clang; For Intel compilers it is
       `-integer-size 64`(Linux/macOS), `/integer-size: 64` (Windows).
     - This is almost always the right thing to do; however for integer
       variables that reference an external non-BLAS/LAPACK interface and must
       not be changed to 64-bit, those should then be explicitly `integer*4` in
       the user code.
5. Users are responsible for implementing name mangling. I.e., appending `64_`
   to BLAS/LAPACK symbols when they are requesting ILP64, and also using
   portable integer types in their code if they want to be able to build with
   both LP64 and ILP64. This typically looks something like:

   ```C
   #ifdef HAVE_CBLAS64_
   #define CBLAS_FUNC(name) name ## 64_
   #else
   #define CBLAS_FUNC(name) name
   #endif

   CBLAS_FUNC(cblas_dgemm)(...);
   ```
6. Users are responsible for implementing a build option (e.g., in `meson_options.txt`)
   if they want to allow switching between LP64 and ILP64.
7. Meson doesn't know anything about f2py; users have to instruct f2py to use
   64-bit integers with `--f2cmap` or similar. See `get_f2py_int64_options` in
   SciPy for details.
8. When mixing C and Fortran code, the C code typically needs mangling because
   Fortran expects a trailing underscore. This is up to the user to implement.
9. TBD: `numpy.distutils` does a symbol prefix/suffix check and provides the
   result to its users, it could be helpful if Meson did this. See
   https://github.com/numpy/numpy/blob/6094eff9/numpy/distutils/system_info.py#L2271-L2278.

CBLAS
-----

Initial rough notes:

- Not all implementations provide CBLAS,
- The header is typically named `cblas.h`, however MKL calls it `mkl_cblas.h`,
- OpenBLAS can be built without a Fortran compiler, in that case it's CBLAS + f2c'd LAPACK,
- It would be useful if the dependency objects that Meson returned can be
  queried for whether CBLAS is present or not.
- numpy.distutils detects CBLAS and defines `HAVE_CBLAS` if it's found.
- BLIS doesn't build the CBLAS interface by default. To build it, define
  `BLIS_ENABLE_CBLAS`.
- NumPy requires CBLAS, it's not optional.

MKL
---

MKL docs: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-windows-developer-guide/top/linking-your-application-with-onemkl/linking-in-detail/linking-with-interface-libraries/using-the-ilp64-interface-vs-lp64-interface.html
MKL link line advisor: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html

$ cd /opt/intel/oneapi/mkl/latest/lib  # from recommended Intel installer
$ ls pkgconfig/
mkl-dynamic-ilp64-iomp.pc  mkl-dynamic-lp64-iomp.pc  mkl-static-ilp64-iomp.pc  mkl-static-lp64-iomp.pc
mkl-dynamic-ilp64-seq.pc   mkl-dynamic-lp64-seq.pc   mkl-static-ilp64-seq.pc   mkl-static-lp64-seq.pc

$ pkg-config --libs mkl-dynamic-ilp64-seq
-L/opt/intel/oneapi/mkl/latest/lib/pkgconfig/../../lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
$ pkg-config --cflags mkl-dynamic-ilp64-seq
-DMKL_ILP64 -I/opt/intel/oneapi/mkl/latest/lib/pkgconfig/../../include

$ ls intel64/libmkl*
intel64/libmkl_avx2.so.2                  intel64/libmkl_cdft_core.so.2    intel64/libmkl_intel_lp64.so.2       intel64/libmkl_sequential.a
intel64/libmkl_avx512.so.2                intel64/libmkl_core.a            intel64/libmkl_intel_thread.a        intel64/libmkl_sequential.so
intel64/libmkl_avx.so.2                   intel64/libmkl_core.so           intel64/libmkl_intel_thread.so       intel64/libmkl_sequential.so.2
intel64/libmkl_blacs_intelmpi_ilp64.a     intel64/libmkl_core.so.2         intel64/libmkl_intel_thread.so.2     intel64/libmkl_sycl.a
intel64/libmkl_blacs_intelmpi_ilp64.so    intel64/libmkl_def.so.2          intel64/libmkl_lapack95_ilp64.a      intel64/libmkl_sycl.so
intel64/libmkl_blacs_intelmpi_ilp64.so.2  intel64/libmkl_gf_ilp64.a        intel64/libmkl_lapack95_lp64.a       intel64/libmkl_sycl.so.2
intel64/libmkl_blacs_intelmpi_lp64.a      intel64/libmkl_gf_ilp64.so       intel64/libmkl_mc3.so.2              intel64/libmkl_tbb_thread.a
intel64/libmkl_blacs_intelmpi_lp64.so     intel64/libmkl_gf_ilp64.so.2     intel64/libmkl_mc.so.2               intel64/libmkl_tbb_thread.so
intel64/libmkl_blacs_intelmpi_lp64.so.2   intel64/libmkl_gf_lp64.a         intel64/libmkl_pgi_thread.a          intel64/libmkl_tbb_thread.so.2
intel64/libmkl_blacs_openmpi_ilp64.a      intel64/libmkl_gf_lp64.so        intel64/libmkl_pgi_thread.so         intel64/libmkl_vml_avx2.so.2
intel64/libmkl_blacs_openmpi_ilp64.so     intel64/libmkl_gf_lp64.so.2      intel64/libmkl_pgi_thread.so.2       intel64/libmkl_vml_avx512.so.2
intel64/libmkl_blacs_openmpi_ilp64.so.2   intel64/libmkl_gnu_thread.a      intel64/libmkl_rt.so                 intel64/libmkl_vml_avx.so.2
intel64/libmkl_blacs_openmpi_lp64.a       intel64/libmkl_gnu_thread.so     intel64/libmkl_rt.so.2               intel64/libmkl_vml_cmpt.so.2
intel64/libmkl_blacs_openmpi_lp64.so      intel64/libmkl_gnu_thread.so.2   intel64/libmkl_scalapack_ilp64.a     intel64/libmkl_vml_def.so.2
# intel64/libmkl_blacs_openmpi_lp64.so.2    intel64/libmkl_intel_ilp64.a     intel64/libmkl_scalapack_ilp64.so    intel64/libmkl_vml_mc2.so.2
# intel64/libmkl_blas95_ilp64.a             intel64/libmkl_intel_ilp64.so    intel64/libmkl_scalapack_ilp64.so.2  intel64/libmkl_vml_mc3.so.2
# intel64/libmkl_blas95_lp64.a              intel64/libmkl_intel_ilp64.so.2  intel64/libmkl_scalapack_lp64.a      intel64/libmkl_vml_mc.so.2
intel64/libmkl_cdft_core.a                intel64/libmkl_intel_lp64.a      intel64/libmkl_scalapack_lp64.so
intel64/libmkl_cdft_core.so               intel64/libmkl_intel_lp64.so     intel64/libmkl_scalapack_lp64.so.2

$ objdump -t intel64/libmkl_intel_ilp64.so | grep scopy  # cleaned up output:
0000000000000000         *UND*  0000000000000000              mkl_blas_scopy
0000000000323000 g     F .text  0000000000000030              cblas_scopy_64
00000000002cecf0 g     F .text  0000000000000030              cblas_scopy
000000000025aca0 g     F .text  00000000000001d0              mkl_blas__scopy
000000000025aca0 g     F .text  00000000000001d0              scopy_64_
000000000025aca0 g     F .text  00000000000001d0              scopy_64
000000000025aca0 g     F .text  00000000000001d0              scopy_
000000000025aca0 g     F .text  00000000000001d0              scopy

tl;dr: for MKL there's 8 pkg-config file, so you choose lp64/ilp64, dynamic/static, and pthreads/openmp;
       after that you can pick whatever symbols you like, they all exist and are aliases.

A test with SciPy & MKL:

$ # No pkg-config files for MKL in conda-forge yet, so use the Intel installer:
$ export PKG_CONFIG_PATH=/opt/intel/oneapi/mkl/latest/lib/pkgconfig/
$ meson setup build --prefix=$PWD/build-install -Dblas=mkl-dynamic-ilp64-seq -Dlapack=mkl-dynamic-ilp64-seq
$ python dev.py build
$ ldd build/scipy/linalg/_flapack.cpython-310-x86_64-linux-gnu.so
        linux-vdso.so.1 (0x00007ffe2cd14000)
        libmkl_intel_ilp64.so.2 => /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_ilp64.so.2 (0x00007f75a5000000)
        libmkl_sequential.so.2 => /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so.2 (0x00007f75a3400000)
        libmkl_core.so.2 => /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 (0x00007f759f000000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f75a62e2000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f759ee19000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f75a63e8000)
        libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f75a62db000)
        libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f75a62d6000
$ # Due to RPATH stripping on install, this doesn't actually work unless we put MKL into our conda env:
$ ldd build-install/lib/python3.10/site-packages/scipy/linalg/_flapack.cpython-310-x86_64-linux-gnu.so
        linux-vdso.so.1 (0x00007ffdf6bc6000)
        libmkl_intel_ilp64.so.2 => not found
        libmkl_sequential.so.2 => not found
        libmkl_core.so.2 => not found
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f7e35318000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f7e35131000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f7e356e5000)

Also need to remember that MKL uses a g77 ABI (Accelerate too), while OpenBLAS and most other BLAS
libraries will be using the gfortran ABI. See the `use-g77-abi` option in SciPy's meson_options.txt.

ArmPL
-----

Pkg-config file names for ArmPL to use (from https://github.com/spack/spack/pull/34979#discussion_r1073213319):

    armpl-dynamic-ilp64-omp             armpl-Fortran-static-ilp64-omp
    armpl-dynamic-ilp64-omp.pc          armpl-Fortran-static-ilp64-omp.pc
    armpl-dynamic-ilp64-seq             armpl-Fortran-static-ilp64-seq
    armpl-dynamic-ilp64-seq.pc          armpl-Fortran-static-ilp64-seq.pc
    armpl-dynamic-lp64-omp              armpl-Fortran-static-lp64-omp
    armpl-dynamic-lp64-omp.pc           armpl-Fortran-static-lp64-omp.pc
    armpl-dynamic-lp64-seq              armpl-Fortran-static-lp64-seq
    armpl-dynamic-lp64-seq.pc           armpl-Fortran-static-lp64-seq.pc
    armpl-Fortran-dynamic-ilp64-omp     armpl-static-ilp64-omp
    armpl-Fortran-dynamic-ilp64-omp.pc  armpl-static-ilp64-omp.pc
    armpl-Fortran-dynamic-ilp64-seq     armpl-static-ilp64-seq
    armpl-Fortran-dynamic-ilp64-seq.pc  armpl-static-ilp64-seq.pc
    armpl-Fortran-dynamic-lp64-omp      armpl-static-lp64-omp
    armpl-Fortran-dynamic-lp64-omp.pc   armpl-static-lp64-omp.pc
    armpl-Fortran-dynamic-lp64-seq      armpl-static-lp64-seq
    armpl-Fortran-dynamic-lp64-seq.pc   armpl-static-lp64-seq.pc

Note that as of Jan'23, ArmPL ships the files without a `.pc` extension (that
will hopefully be fixed), and Spack renames then so adds the `.pc` copies of
the original files.

Apple Accelerate
----------------

In macOS >=13.3, two LP64 and one ILP64 build of vecLib are shipped. Due to
compatibility, the legacy interfaces (providing LAPACK 3.2.1) will be used by
default.  To use the new interfaces (providing LAPACK 3.9.1), including ILP64,
it is necessary to set some #defines before including Accelerate / vecLib headers:

- `-DACCELERATE_NEW_LAPACK`: use the new interfaces
- `-DACCELERATE_LAPACK_ILP64`: use new ILP64 interfaces (note this requires
  `-DACCELERATE_NEW_LAPACK` to be set as well)

The normal F77 symbols will remain as the legacy implementation.  The newer
interfaces have separate symbols with suffixes `$NEWLAPACK` or `$NEWLAPACK$ILP64`.

Example binary symbols:

- `_dgemm_`: this is the legacy implementation
- `_dgemm$NEWLAPACK`: this is the new implementation
- `_dgemm$NEWLAPACK$ILP64`: this is the new ILP64 implementaion

If you use Accelerate / vecLib headers with the above defines, you don't need
to worry about the symbol names. They'll get aliased correctly.

For headers and linker flags, check if these directories exist before using them:

1. `-I/System/Library/Frameworks/vecLib.framework/Headers`,
   flags: ['-Wl,-framework', '-Wl,Accelerate']
2. `-I/System/Library/Frameworks/vecLib.framework/Headers`,
   flags: ['-Wl,-framework', '-Wl,vecLib']

Note that the dylib's are no longer physically present, they're provided in the
shared linker cache.
"""


class BLASLAPACKMixin():
    def parse_modules(self, kwargs: T.Dict[str, T.Any]) -> None:
        modules: T.List[str] = mesonlib.extract_as_list(kwargs, 'modules')
        valid_modules = ['interface: lp64', 'interface: ilp64', 'cblas', 'lapack', 'lapacke']
        for module in modules:
            if module not in valid_modules:
                raise mesonlib.MesonException(f'Unknown modules argument: {module}')

        self.interface = ''
        interface = [s for s in modules if s.startswith('interface')]
        if interface:
            if len(interface) > 1:
                raise mesonlib.MesonException(f'Only one interface must be specified, found: {interface}')
            self.interface = interface[0].split(' ')[1]

        self.needs_cblas = 'cblas' in modules
        self.needs_lapack = 'lapack' in modules
        self.needs_lapacke = 'lapacke' in modules

    def check_symbols(self, compile_args) -> None:
        # verify that we've found the right LP64/ILP64 interface
        symbols = ['dgemm_']
        if self.needs_cblas:
            symbols += ['cblas_dgemm']
        if self.needs_lapack:
            symbols += ['zungqr_']
        if self.needs_lapacke:
            symbols += ['LAPACKE_zungqr']

        suffix = '' if self.interface == 'lp64' else '64_'
        prototypes = "".join(f"void {symbol}{suffix}();\n" for symbol in symbols)
        calls = "  ".join(f"{symbol}{suffix}();\n" for symbol in symbols)
        code = (f"{prototypes}"
                 "int main(int argc, const char *argv[])\n"
                 "{\n"
                f"  {calls}"
                 "  return 0;\n"
                 "}"
                )
        return self.clib_compiler.links(code, self.env, extra_args=compile_args)[0]

    def get_variable(self, **kwargs: T.Dict[str, T.Any]) -> str:
        # TODO: what's going on with `get_variable`? Need to pick from
        # cmake/pkgconfig/internal/..., but not system?
        varname = kwargs['pkgconfig']
        if varname == 'interface':
            return self.interface
        return super().get_variable(**kwargs)


class OpenBLASSystemDependency(BLASLAPACKMixin, SystemDependency):
    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        self.feature_since = ('1.3.0', '')
        self.parse_modules(kwargs)

        # First, look for paths specified in a machine file
        props = self.env.properties[self.for_machine].properties
        if any(x in props for x in ['openblas_includedir', 'openblas_librarydir']):
            self.detect_openblas_machine_file(props)

        # Then look in standard directories by attempting to link
        if not self.is_found:
            extra_libdirs: T.List[str] = []
            self.detect(extra_libdirs)

        if self.is_found:
            self.version = self.detect_openblas_version()

    def detect(self, lib_dirs: T.Optional[T.List[str]] = None, inc_dirs: T.Optional[T.List[str]] = None) -> None:
        if lib_dirs is None:
            lib_dirs = []
        if inc_dirs is None:
            inc_dirs = []

        if self.interface == 'lp64':
            libnames = ['openblas']
        elif self.interface == 'ilp64':
            libnames = ['openblas64', 'openblas_ilp64', 'openblas']

        for libname in libnames:
            link_arg = self.clib_compiler.find_library(libname, self.env, lib_dirs)
            incdir_args = [f'-I{inc_dir}' for inc_dir in inc_dirs]
            found_header, _ = self.clib_compiler.has_header('openblas_config.h', '', self.env,
                                                            dependencies=[self], extra_args=incdir_args)
            if link_arg and found_header:
                if not self.check_symbols(link_arg):
                    continue
                self.is_found = True
                if lib_dirs:
                    # `link_arg` will be either `[-lopenblas]` or `[/path_to_sharedlib/libopenblas.so]`
                    # is the latter behavior expected?
                    found_libdir = Path(link_arg[0]).parent
                    self.link_args += [f'-L{found_libdir}', f'-l{libname}']
                else:
                    self.link_args += link_arg

                # has_header does not return a path with where the header was
                # found, so add all provided include directories
                self.compile_args += incdir_args
                return None

    def detect_openblas_machine_file(self, props: dict) -> None:
        # TBD: do we need to support multiple extra dirs?
        incdir = props.get('openblas_includedir')
        assert incdir is None or isinstance(incdir, str)
        libdir = props.get('openblas_librarydir')
        assert libdir is None or isinstance(libdir, str)

        if incdir and libdir:
            self.is_found = True
            if not Path(incdir).is_absolute() or not Path(libdir).is_absolute():
                raise mesonlib.MesonException('Paths given for openblas_includedir and '
                                              'openblas_librarydir in machine file must be absolute')
        elif incdir or libdir:
            raise mesonlib.MesonException('Both openblas_includedir *and* openblas_librarydir '
                                          'have to be set in your machine file (one is not enough)')
        else:
            raise mesonlib.MesonBugException('issue with openblas dependency detection, should not '
                                             'be possible to reach this else clause')

        self.detect([libdir], [incdir])

    def detect_openblas_version(self) -> str:
        v, _ = self.clib_compiler.get_define('OPENBLAS_VERSION', '#include <openblas_config.h>', self.env, [], [self])

        m = re.search(r'\d+(?:\.\d+)+', v)
        if not m:
            mlog.debug('Failed to extract openblas version information')
            return None
        return m.group(0)


class OpenBLASPkgConfigDependency(BLASLAPACKMixin, PkgConfigDependency):
    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, env, kwargs)
        self.feature_since = ('1.3.0', '')
        self.parse_modules(kwargs)

        if not self.check_symbols(self.link_args):
            self.is_found = False


class OpenBLASCMakeDependency(BLASLAPACKMixin, CMakeDependency):
    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any],
                 language: T.Optional[str] = None, force_use_global_compilers: bool = False) -> None:
        # TODO: support ILP64. Use functools.partial(PkgConfigDependency...)
        #       for the 3 possible names here?
        super().__init__('OpenBLAS', env, kwargs, language, force_use_global_compilers)
        self.feature_since = ('1.3.0', '')
        self.parse_modules(kwargs)

        if not self.check_symbols(self.link_args):
            self.is_found = False


class NetlibPkgConfigDependency(BLASLAPACKMixin, PkgConfigDependency):
    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        # TODO: add 'cblas'
        super().__init__('blas', env, kwargs)
        self.feature_since = ('1.3.0', '')
        self.parse_modules(kwargs)


class AccelerateSystemDependency(BLASLAPACKMixin, SystemDependency):
    """
    Accelerate is always installed on macOS, and not available on other OSes.
    We only support using Accelerate on macOS >=13.3, where Apple shipped a
    major update to Accelerate, fixed a lot of bugs, and bumped the LAPACK
    version from 3.2 to 3.9. The older Accelerate version is still available,
    and can be obtained as a standard Framework dependency with:

        dependency('appleframeworks', modules : 'Accelerate')
    """
    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        self.feature_since = ('1.3.0', '')
        self.parse_modules(kwargs)

        if not self.check_macOS_recent_enough():
            return None

        self.detect()

    def check_macOS_recent_enough(self) -> bool:
        macOS13_3_or_later = False
        if os.name == 'darwin':
            cmd = ['xcrun', '-sdk', 'macosx', '--show-sdk-version'],
            sdk_version = subprocess.run(cmd, capture_output=True, check=True)
            if sdk_version >= '13.3':
                macOS13_3_or_later = True
        return macOS13_3_or_later

    def detect(self) -> None:
        libname = 'Accelerate'
        link_arg = self.clib_compiler.find_framework('Accelerate', self.env)
        found_header, _ = self.clib_compiler.has_header('<Accelerate/Accelerate.h>', '',
                                                        self.env, dependencies=[self])
        if link_arg and found_header:
            self.is_found = True
            self.compile_args += ['-DACCELERATE_NEW_LAPACK']
            if self.interface == 'ilp64':
                self.compile_args += ['-DACCELERATE_LAPACK_ILP64']

        # We won't check symbols here, because Accelerate is built in a
        # consistent fashion with known symbol mangling, unlike OpenBLAS or
        # Netlib BLAS/LAPACK.


packages['openblas'] = openblas_factory = DependencyFactory(
    'openblas',
    [DependencyMethods.SYSTEM, DependencyMethods.PKGCONFIG, DependencyMethods.CMAKE],
    system_class=OpenBLASSystemDependency,
    pkgconfig_class=OpenBLASPkgConfigDependency,
    cmake_class=OpenBLASCMakeDependency,
)


packages['netlib-blas'] = netlib_factory = DependencyFactory(
    'netlib-blas',
    [DependencyMethods.PKGCONFIG],  #, DependencyMethods.SYSTEM],
    #system_class=NetlibSystemDependency,
    pkgconfig_class=NetlibPkgConfigDependency,
)


packages['accelerate'] = accelerate_factory = DependencyFactory(
    'accelerate',
    [DependencyMethods.SYSTEM],
    system_class=AccelerateSystemDependency,
)
