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

from pathlib import Path
import re
import typing as T

from .. import mlog
from .. import mesonlib

from .base import DependencyMethods, SystemDependency
from .factory import DependencyFactory

if T.TYPE_CHECKING:
    from ..environment import Environment


# TODO: how to select BLAS interface layer (LP64, ILP64)?
#       LP64 is the 32-bit interface (confusingly)
#       ILP64 is the 64-bit interface
#
#       OpenBLAS library names possible:
#           - libopenblas
#           - libopenblas64_  # are these suffixes NumPy-specific or not? If so, don't support it?
#           - libopenblas_ilp64
#
# MKL docs: https://www.intel.com/content/www/us/en/develop/documentation/onemkl-windows-developer-guide/top/linking-your-application-with-onemkl/linking-in-detail/linking-with-interface-libraries/using-the-ilp64-interface-vs-lp64-interface.html
#
# TODO: do we care here about whether OpenBLAS is compiled with pthreads or OpenMP?
#       MKL has separate .pc files for this (_seq vs. _iomp), OpenBLAS does not
#
# TODO: what about blas/lapack vs. cblas/clapack?
#       See http://nicolas.limare.net/pro/notes/2014/10/31_cblas_clapack_lapacke/ for details
#
# NOTE: we ignore static libraries for now


class OpenBLASSystemDependency(SystemDependency):
    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        self.feature_since = ('0.64.0', '')

        # First, look for paths specified in a machine file
        props = self.env.properties[self.for_machine].properties
        if any(x in props for x in ['openblas_includedir', 'openblas_librarydir']):
            self.detect_openblas_machine_file(props)

        # Then look in standard directories by attempting to link
        if not self.is_found:
            # TODO - for Windows may be, e.g.: Path('C:/opt/openblas/if_32/64/')
            extra_libdirs: T.List[str] = []
            self.detect(extra_libdirs)

        if self.is_found:
            self.version = self.detect_openblas_version()

        self.run_check()

    def detect(self, lib_dirs: T.Optional[T.List[str]] = None, inc_dirs: T.Optional[T.List[str]] = None) -> None:
        if lib_dirs is None:
            lib_dirs = []
        if inc_dirs is None:
            inc_dirs = []

        link_arg = self.clib_compiler.find_library('openblas', self.env, lib_dirs)
        incdir_args = [f'-I{inc_dir}' for inc_dir in inc_dirs]
        found_header, _ = self.clib_compiler.has_header('openblas_config.h', '', self.env,
                                                        dependencies=[self], extra_args=incdir_args)
        if link_arg and found_header:
            self.is_found = True
            if lib_dirs:
                # `link_arg` will be either `[-lopenblas]` or `[/path_to_sharedlib/libopenblas.so]`
                # is the latter behavior expected?
                found_libdir = Path(link_arg[0]).parent
                self.link_args += [f'-L{found_libdir}', '-lopenblas']
            else:
                self.link_args += link_arg

            # has_header does not return a path with where the header was
            # found, so add all provided include directories
            self.compile_args += incdir_args

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

    def run_check(self) -> None:
        # See https://github.com/numpy/numpy/blob/main/numpy/distutils/system_info.py#L2319
        # Symbols to check:
        #    for BLAS LP64: dgemm  # note that numpy.distutils checks nothing here
        #    for BLAS ILP64: dgemm_, cblas_dgemm
        #    for LAPACK LP64: zungqr_
        #    for LAPACK ILP64: zungqr_, LAPACKE_zungqr
        pass


openblas_factory = DependencyFactory(
    'openblas',
    [DependencyMethods.PKGCONFIG, DependencyMethods.SYSTEM, DependencyMethods.CMAKE],
    cmake_name='OpenBLAS',
    system_class=OpenBLASSystemDependency,
)
