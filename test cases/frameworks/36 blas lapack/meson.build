project('test blas and lapack', 'c')

openblas_dep = dependency('openblas',
  modules: [
    'interface: lp64',
    'cblas',
    'lapack',
  ],
  required: false,
)
openblas64_dep = dependency('openblas', required: false, modules: ['interface: ilp64'])
if not openblas_dep.found() and not openblas64_dep.found()
  error('MESON_SKIP_TEST: OpenBLAS library not available')
endif

deps = {
  'lp64': openblas_dep,
  'ilp64': openblas64_dep,
}

foreach interface, dep : deps
  if dep.found()
    blas_c_args = []
    blas_interface = dep.get_variable('interface')
    if blas_interface == 'ilp64'
      blas_c_args += '-DBLAS_SYMBOL_SUFFIX=64_'
    elif blas_interface != 'lp64'
      error('no interface var for OpenBLAS dependency!')
    endif

    c_exe = executable('cblas_lapack_c_' + interface,
      'cblas_lapack.c',
      c_args: blas_c_args,
      dependencies: [dep]
    )
    test('openblas_dep', c_exe, timeout: 30)

    if add_languages('fortran', required: false)
      f_exe = executable('openblas_fortran_' + interface,
        'main.f90',
        dependencies: [dep]
      )
      test('openblas_fortran', f_exe, timeout: 30)
    endif
  endif
endforeach
