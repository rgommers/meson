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
import typing as T

from ..mesonlib import OptionKey
from .base import DependencyMethods, SystemDependency
from .base import DependencyException
from .cmake import CMakeDependency
from .factory import factory_methods

if T.TYPE_CHECKING:
    from ..environment import Environment, MachineChoice
    from .factory import DependencyGenerator


@factory_methods({DependencyMethods.SYSTEM, DependencyMethods.CMAKE})
def numpy_factory(env: 'Environment', for_machine: 'MachineChoice',
                  kwargs: T.Dict[str, T.Any],
                  methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    candidates: T.List['DependencyGenerator'] = []

    # NumPy does not provide pkg-config files (yet, planned for after NumPy
    # moves to Meson as its build system)

    if DependencyMethods.SYSTEM in methods:
        candidates.append(functools.partial(NumPyDependencySystem, 'numpy', env, kwargs))

    if DependencyMethods.CMAKE in methods:
        # TODO: test/fix this (does CmakeDependency handle this case?)!
        candidates.append(functools.partial(
            CMakeDependency, 'Python3 COMPONENTS Interpreter NumPy', env, kwargs))

    return candidates


class NumPyDependencySystem(SystemDependency):
    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        self.feature_since = ('0.64.0')

        if not environment.machines.matches_build_machine(self.for_machine):
            # FIXME: can't support cross-compiling via running Python to get at
            #        `numpy.get_include()`
            return

        self.name = 'numpy'
        self.is_found = False
        include_dir = self._get_numpy_includedir()
        if include_dir is None:
            return

        self.compile_args.append(f'-I{include_dir}')
        self.version = self._get_version()

        def _run_py(self, code):
            # TODO: how do we get the python executable detected by
            #       `dependency(python)`?
            #py3 = XXX
            # Alternatively: don't run the Python interpreter (can't do for
            # cross-compiling anyway), but just hardcode the location relative
            # to site-packages; we *know* what it should be. But still need to
            # get that location somehow.
            ret = subprocess.run([py3, '-c', code],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,
                                 text=True)
            if ret.returncode == 0:
                return ret

        def _get_numpy_includedir(self) -> T.Optional(str):
            code = 'import numpy; print(numpy.get_include())'
            include_dir = self._run_py(code)
            if include_dir and include_dir.endswith(os.path.join(['numpy', 'core', 'include'])):
                self.is_found = True
                return include_dir

        def _get_version(self) -> str:
            code = 'import numpy; print(numpy.__version__)'
            return self._run_py(code)

