project('test blas and lapack', 'c')

# NOTE: this is for local testing, we probably only want a LP64 openblas
#       in the Meson CI. We need to check symbols though! We may now pick up a
#       wrong binary
openblas_dep = dependency('openblas', required: false, modules: ['interface: ilp64'])
if not openblas_dep.found()
  error('MESON_SKIP_TEST: OpenBLAS library not available')
endif

blas_c_args = []
blas_interface = openblas_dep.get_variable('interface')
if blas_interface == 'ilp64'
  blas_c_args += '-DBLAS_SYMBOL_SUFFIX=64_'
elif blas_interface != 'lp64'
  error('no interface var for OpenBLAS dependency!')
endif

c_exe = executable('cblas_lapack_c',
  'cblas_lapack.c',
  c_args: blas_c_args,
  dependencies: [openblas_dep]
)
test('openblas_dep', c_exe, timeout: 30)

if add_languages('fortran', required: false)
  f_exe = executable('openblas_fortran',
    'main.f90',
    dependencies: [openblas_dep]
  )
  test('openblas_fortran', f_exe, timeout: 30)
endif
