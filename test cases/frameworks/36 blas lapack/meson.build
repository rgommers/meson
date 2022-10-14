project('test blas and lapack', 'c')

openblas_dep = dependency('openblas', required: false)
if not openblas_dep.found()
  error('MESON_SKIP_TEST: OpenBLAS library not available')
endif

#c_exe = executable('blas_c', 'main.c',
#  dependencies: [openblas_dep])
#test('openblas_dep', c_exe, timeout: 30)

if add_languages('fortran', required: false)
  f_exe = executable('openblas_fortran', 'main.f90',
    dependencies: [openblas_dep])
  test('openblas_fortran', f_exe, timeout: 30)
endif
