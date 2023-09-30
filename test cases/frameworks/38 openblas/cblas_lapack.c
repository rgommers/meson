// Basic BLAS/LAPACK example adapted from a test in Spack for OpenBLAS
// Name mangling adapted from NumPy's npy_cblas.h

// hacky - maybe we should get these headers from the meson.build file based on dependency
#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef ACCELERATE_NEW_LAPACK
    #if __MAC_OS_X_VERSION_MAX_ALLOWED < 130300
        #ifdef HAVE_BLAS_ILP64
            #error "Accelerate ILP64 support is only available with macOS 13.3 SDK or later"
        #endif
    #else
        #define NO_APPEND_FORTRAN
        #ifdef HAVE_BLAS_ILP64
            #define BLAS_SYMBOL_SUFFIX $NEWLAPACK$ILP64
        #else
            #define BLAS_SYMBOL_SUFFIX $NEWLAPACK
        #endif
    #endif
#endif

#ifdef NO_APPEND_FORTRAN
#define BLAS_FORTRAN_SUFFIX
#else
#define BLAS_FORTRAN_SUFFIX _
#endif

#ifndef BLAS_SYMBOL_SUFFIX
#define BLAS_SYMBOL_SUFFIX
#endif

#define BLAS_FUNC_CONCAT(name,suffix,suffix2) name ## suffix ## suffix2
#define BLAS_FUNC_EXPAND(name,suffix,suffix2) BLAS_FUNC_CONCAT(name,suffix,suffix2)

#define CBLAS_FUNC(name) BLAS_FUNC_EXPAND(name,,BLAS_SYMBOL_SUFFIX)
#define BLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_FORTRAN_SUFFIX,BLAS_SYMBOL_SUFFIX)

#ifdef HAVE_BLAS_ILP64
#define blas_int long
#else
#define blas_int int
#endif

#ifdef __cplusplus
extern "C" {
#endif

void BLAS_FUNC(dgesv)(blas_int *n, blas_int *nrhs, double *a, blas_int *lda, blas_int *ipivot, double *b,
                      blas_int *ldb, blas_int *info);

#ifdef __cplusplus
}
#endif

int main(void) {
    // CBLAS:
    blas_int incx = 1;
    double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5};
    blas_int n_elem = 9;
    double norm;

    // CBLAS_FUNC(cblas_dgemm)(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3,
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3,
                2, C, 3);
    norm = cblas_dnrm2(n_elem, C, incx) - 28.017851;

    if (fabs(norm) < 1e-5) {
        printf("OK: CBLAS result using dgemm and dnrm2 as expected\n");
    } else {
        fprintf(stderr, "CBLAS result using dgemm and dnrm2 incorrect: %f\n", norm);
        exit(EXIT_FAILURE);
    }

    // LAPACK:
    double m[] = {3, 1, 3, 1, 5, 9, 2, 6, 5};
    double x[] = {-1, 3, -3};
    blas_int ipiv[3];
    blas_int info;
    blas_int n = 1;
    blas_int nrhs = 1;
    blas_int lda = 3;
    blas_int ldb = 3;

    BLAS_FUNC(dgesv)(&n, &nrhs, &m[0], &lda, ipiv, &x[0], &ldb, &info);
    n_elem = 3;
    norm = cblas_dnrm2(n_elem, x, incx) - 4.255715;

    if (fabs(norm) < 1e-5) {
        printf("OK: LAPACK result using dgesv as expected\n");
    } else {
        fprintf(stderr, "LAPACK result using dgesv incorrect: %f\n", norm);
        exit(EXIT_FAILURE);
    }

    return 0;
}
