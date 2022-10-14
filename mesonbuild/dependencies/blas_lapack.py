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
import functools
import os
import re
import typing as T

from .. import mlog
from .. import mesonlib

from .base import DependencyMethods, SystemDependency
from .base import DependencyException
from .cmake import CMakeDependency
from .factory import factory_methods

if T.TYPE_CHECKING:
    from ..environment import Environment, MachineChoice
    from .factory import DependencyGenerator


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


@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CMAKE, DependencyMethods.SYSTEM})
def openblas_factory(env: 'Environment', for_machine: 'MachineChoice',
                  kwargs: T.Dict[str, T.Any],
                  methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    candidates: T.List['DependencyGenerator'] = []

    if DependencyMethods.CMAKE in methods:
        candidates.append(functools.partial(
            CMakeDependency, 'OpenBLAS', env, kwargs))

    if DependencyMethods.SYSTEM in methods:
        candidates.append(functools.partial(OpenBLASDependencySystem, 'openblas', env, kwargs))

    return candidates


class OpenBLASDependencySystem(SystemDependency):
    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        self.feature_since = ('0.64.0', '')

        self.env = environment
        self.name = 'openblas'
        self.is_found = False

        # First, look for paths specified in a machine file
        props = self.env.properties[self.for_machine].properties
        if any(x in props for x in ['openblas_includedir', 'openblas_librarydir']):
            self.detect_openblas_machine_file(props)

        # Then look in standard directories by attempting to link
        if not self.is_found:
            extra_dirs = []  # TODO - for Windows may be, e.g.: Path('C:/opt/openblas/if_32/64/')
            link_arg = self.clib_compiler.find_library('openblas', self.env, extra_dirs)
            h = self.clib_compiler.has_header('openblas_config.h', '', self.env, dependencies=[self])
            if link_arg and h[0]:
                self.is_found = True
                self.link_args += link_arg

        if self.is_found:
            self.version = self.detect_openblas_version()

        self.run_check()

    def detect_openblas_machine_file(self, props: dict) -> None:
        incdir = props.get('openblas_includedir')
        assert incdir is None or isinstance(incdir, str)
        libdir = props.get('openblas_librarydir')
        assert libdir is None or isinstance(libdir, str)

        if incdir and libdir:
            self.is_found = True
            inc_dir = Path(incdir)
            lib_dir = Path(libdir)
            if not inc_dir.is_absolute() or not lib_dir.is_absolute():
                # TODO: why doesn't this fail the build? It gets caught, how to avoid?
                raise DependencyException('Paths given for openblas_includedir and openblas_librarydir in machine file must be absolute')
        elif incdir or libdir:
            raise DependencyException('Both openblas_includedir *and* openblas_librarydir have to be set in your machine file (one is not enough)')
        else:
            # We only call this method if incdir or libdir were found
            raise RuntimeError('Meson internal issue during openblas dependency detection')

        # Now we have the absolute paths we need - use them:
        self.link_args += [f'-L{lib_dir}', '-lopenblas', f'-I{inc_dir}']

    def detect_openblas_version(self) -> str:
        v, _ = self.clib_compiler.get_define('OPENBLAS_VERSION', '#include <openblas_config.h>', self.env, [], [self])

        m = re.search(r'\d+(?:\.\d+)+', v)
        if not m:
            mlog.debug(f'Failed to extract openblas version information')
            return '0.0.0'
        return m.group(0)

    def run_check(self):
        # See https://github.com/numpy/numpy/blob/main/numpy/distutils/system_info.py#L2319
        # Symbols to check:
        #    for BLAS LP64: dgemm  # note that numpy.distutils checks nothing here
        #    for BLAS ILP64: dgemm_, cblas_dgemm
        #    for LAPACK LP64: zungqr_
        #    for LAPACK ILP64: zungqr_, LAPACKE_zungqr
        pass
