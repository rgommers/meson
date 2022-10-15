// Adapted from a test in Spack for OpenBLAS

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipivot, double *b,
            int *ldb, int *info);

#ifdef __cplusplus
}
#endif

int main(void) {
    // CBLAS:
    int incx = 1;
    double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5};
    int n_elem = 9;
    double norm;

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
    int ipiv[3];
    int info;
    int n = 1;
    int nrhs = 1;
    int lda = 3;
    int ldb = 3;

    dgesv_(&n, &nrhs, &m[0], &lda, ipiv, &x[0], &ldb, &info);
    n_elem = 3;
    norm = cblas_dnrm2(n_elem, x, incx) - 4.255715;

    if (fabs(norm) < 1e-5) {
        printf("OK: LAPACK result using dgesv_ as expected\n");
    } else {
        fprintf(stderr, "LAPACK result using dgesv_ incorrect: %f\n", norm);
        exit(EXIT_FAILURE);
    }

    return 0;
}
