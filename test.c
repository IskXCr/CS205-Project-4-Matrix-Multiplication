/* Source file for matrix tests, containing all the built-in test rountines for the matrix library */
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

/* Definition for ease of use */

#define FLOAT_ERR 1e-7

/* Global variables */

static const size_t SZT_MAX = (size_t)-1;

/* Function prototypes */

struct timespec diff(struct timespec start, struct timespec end);

void print_test_init(const char *s);
void print_test_end(const char *s);

void test_matrix_struct(void);
void test_matrix_read(void);
void test_matrix_write(void);
void test_emma_on_matrices(void);
void test_emma_on_matrix_and_scalar(void);
void test_min_max(void);
void test_matrix_mul_and_transpose(void);
void test_matrix_mul_performance(void);

void (*test_funcs[])(void) = {&test_matrix_struct,
                              &test_matrix_read,
                              &test_matrix_write,
                              &test_emma_on_matrices,
                              &test_emma_on_matrix_and_scalar,
                              &test_min_max,
                              &test_matrix_mul_and_transpose,
                              &test_matrix_mul_performance};

int main()
{
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    size_t test_sz = sizeof(test_funcs) / sizeof(void (*)(void));
    for (size_t i = 0; i < test_sz; ++i)
        (*test_funcs[i])();

    clock_gettime(CLOCK_REALTIME, &end);
    printf("Evaluation result: all tests passed.\nTime elapsed: %ld unit\n", diff(start, end).tv_nsec / 1000000L);
    return 0;
}

/* Get the difference between two time.
   Reference: https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance */
struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

void print_test_init(const char *s)
{
    printf(">>>>>>>>>>Init Test: %s\n", s);
}

void print_test_end(const char *s)
{
    printf("<<<<<<<<<<Ending Test: %s\n\n", s);
}

void test_matrix_struct(void)
{
    print_test_init("struct test");

    matrix m0 = NULL, m1 = NULL, m2 = NULL;
    matrix_errno err;
    size_t x = 233, y = 233;

    /* Test parameters */
    assert((m0 = create_matrix(0, 0)) == NULL);
    assert((m0 = create_matrix(x, 0)) == NULL);
    assert((m0 = create_matrix(0, y)) == NULL);
    assert((m0 = create_matrix(SZT_MAX - 1, SZT_MAX - 1)) == NULL); /* Extreme case */
    assert((m0 = create_matrix(x, y)) != NULL);
    printf("parameter test ok.\n");

    assert(m0->refs == 1);
    assert(m0->rows == x);
    assert(m0->cols == y);
    printf("property test ok.\n");

    /* Copying */
    assert((err = copy_matrix(&m1, m0)) == COMPLETED);
    assert(m1 != m0);
    assert(m1->refs == 1);
    delete_matrix(&m1);
    assert(m1 == NULL);
    assert(m0->refs == 1);
    printf("copy test ok.\n");

    /* Reference and deletion */
    m2 = m1 = ref_matrix(m0);
    assert(m1 == m0);
    delete_matrix(&m0);
    assert(m0 == NULL);
    assert(m1->refs == 1);
    delete_matrix(&m1);
    assert(m1 == NULL);
    assert(m2->refs == 0);
    printf("reference and deletion test ok.\n");

    /* Test recycling */
    m0 = create_matrix(x, y);
    assert(m0 == m2);
    m2 = m0;
    m0->refs = SZT_MAX;
    assert((m1 = ref_matrix(m0)) == NULL); /* Test upper bound exceeded scenario */
    m0->refs = 1;
    assert(m2->refs == 1);
    delete_matrix(&m0);
    assert(m0 == NULL);
    assert(m2->refs == 0);
    printf("recycling test ok.\n");

    print_test_end("struct test");
}

void test_matrix_read(void)
{
    print_test_init("read test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n";

    matrix tmp = NULL;
    matrix_errno err;

    /* Test string read */
    assert((err = read_matrix_from_string(NULL, &tmp)) == OP_NULL_PTR);
    assert((err = read_matrix_from_string(s1, &tmp)) == COMPLETED);
    assert(tmp != NULL);
    printf("simple string read ok.\n");

    assert(tmp->rows == 8);
    assert(tmp->cols == 9);
    assert(tmp->refs == 1);
    printf("string read property test ok.\n");

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? j + 1 : 9 - j)) <= FLOAT_ERR);
    delete_matrix(&tmp);
    printf("string read parameter test ok.\n");

    /* Test stream read */
    FILE *p = fopen("__matrix_lib_test.1", "w+");
    assert(p != NULL);
    fprintf(p, "%s", s1);
    fclose(p);
    printf("stream write ok.\n");

    p = fopen("__matrix_lib_test.1", "r");
    assert(p != NULL);
    assert((err = read_matrix_from_stream(p, &tmp)) == COMPLETED);
    printf("stream read parameter test ok.\n");

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? j + 1 : 9 - j)) <= FLOAT_ERR);
    delete_matrix(&tmp);
    printf("stream read element test ok.\n");

    print_test_end("read test");
}

void test_matrix_write(void)
{
    print_test_init("write test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n";

    matrix tmp = NULL;
    matrix_errno err;
    assert((err = read_matrix_from_string(s1, &tmp)) == COMPLETED);
    FILE *p = fopen("__matrix_lib_test.1", "w+");
    assert((err = print_matrix(p, tmp)) == COMPLETED);
    fflush(p);
    fclose(p);
    printf("write sample ok.\n");

    /* After writing to file, test reading */
    p = fopen("__matrix_lib_test.1", "r+");
    int q = 0;
    float c;
    size_t m, n;
    assert(fscanf(p, "%zu %zu", &m, &n) == 2);
    assert(m == tmp->rows && n == tmp->cols);
    while (q < 72 && fscanf(p, "%f", &c) > 0)
        assert(fabsf(c - tmp->arr[q++]) <= FLOAT_ERR);
    delete_matrix(&tmp);
    fclose(p);
    printf("write verification ok.\n");

    print_test_end("write test");
}

void test_emma_on_matrices(void)
{
    print_test_init("EMMA on matrices test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n";

    char *s2 = "8 9\n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n";

    matrix tmp0 = NULL, tmp1 = NULL, tmp2 = NULL;
    matrix_errno err;
    assert((err = read_matrix_from_string(s1, &tmp0)) == COMPLETED);
    assert((err = read_matrix_from_string(s2, &tmp1)) == COMPLETED);
    assert((err = add_matrix(tmp0, tmp1, &tmp2)) == COMPLETED);
    assert(tmp2 != tmp1 && tmp2 != tmp0);
    for (int i = 0; i < 72; ++i)
        assert(fabs(tmp2->arr[i] - 10.0f) <= FLOAT_ERR);
    delete_matrix(&tmp1);
    delete_matrix(&tmp2);
    printf("matrix addition test ok.\n");

    assert((err = read_matrix_from_string(s1, &tmp1)) == COMPLETED);
    assert((err = subtract_matrix(tmp0, tmp1, &tmp2)) == COMPLETED);
    for (int i = 0; i < 72; ++i)
        assert(fabs(tmp2->arr[i] - 0.0f) <= FLOAT_ERR);
    delete_matrix(&tmp0);
    delete_matrix(&tmp1);
    delete_matrix(&tmp2);
    printf("matrix subtraction test ok.\n");

    print_test_end("EMMA on matrices test");
}

void test_emma_on_matrix_and_scalar(void)
{
    print_test_init("EMMA on matrix and scalar test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n";
    matrix tmp = NULL;
    matrix_errno err;
    assert((err = read_matrix_from_string(s1, &tmp)) == COMPLETED);

    assert((err = multiply_scalar(tmp, &tmp, 9.0f)) == COMPLETED);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? 9 * (j + 1) : 9 * (9 - j))) <= FLOAT_ERR);
    printf("scalar multiplication test ok.\n");

    matrix tmp2 = NULL;
    assert((err = multiply_scalar(tmp, &tmp2, 9.0f)) == COMPLETED);
    for (int i = 0; i < 8; ++i) /* Check result */
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp2->arr[i * tmp2->cols + j] - (float)((i % 2 == 0) ? 81 * (j + 1) : 81 * (9 - j))) <= FLOAT_ERR);
    for (int i = 0; i < 8; ++i) /* Check original one */
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? 9 * (j + 1) : 9 * (9 - j))) <= FLOAT_ERR);
    printf("result writing test ok.\n");

    assert((err = add_scalar(tmp, &tmp, 1.0f)) == COMPLETED);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? 9 * (j + 1) + 1 : 9 * (9 - j) + 1)) <= FLOAT_ERR);
    printf("scalar addition test ok.\n");

    assert((err = subtract_scalar(tmp, &tmp, 3.0f)) == COMPLETED);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(fabsf(tmp->arr[i * tmp->cols + j] - (float)((i % 2 == 0) ? 9 * (j + 1) - 2 : 9 * (9 - j) - 2)) <= FLOAT_ERR);
    printf("scalar subtraction test ok.\n");

    delete_matrix(&tmp);
    delete_matrix(&tmp2);

    print_test_end("EMMA on matrix and scalar test completed. ");
}

void test_min_max(void)
{
    print_test_init("MIN-MAX test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7e14 8 9 \n"
               "9 8 7 6 5 4 3 2 -1 \n"
               "1 2 1657 3 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 -7768 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3.789 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 -3e12 2 1 \n";

    matrix tmp = NULL;
    matrix_errno err;
    assert((err = read_matrix_from_string(s1, &tmp)) == COMPLETED);

    assert(fabsf(matrix_min(tmp) + 3e12) <= FLOAT_ERR * 1e13);
    printf("minimum test ok.\n");

    assert(fabsf(matrix_max(tmp) - 7e14) <= FLOAT_ERR * 1e15);
    printf("maximum test ok.\n");

    delete_matrix(&tmp);

    print_test_end("MIN-MAX test");
}

void test_matrix_mul_and_transpose(void)
{
    print_test_init("Matrix multiplication test");

    char *s1 = "8 9\n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n"
               "1 2 3 4 5 6 7 8 9 \n"
               "9 8 7 6 5 4 3 2 1 \n";

    matrix tmp1 = NULL, tmp2 = NULL, tmp3 = NULL;
    matrix_errno err;
    assert((err = read_matrix_from_string(s1, &tmp1)) == COMPLETED);
    assert((err = read_matrix_from_string(s1, &tmp2)) == COMPLETED);
    assert((err = transpose_matrix(tmp2, &tmp2) == COMPLETED));
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            assert(tmp1->arr[i * 9 + j] == tmp2->arr[j * 8 + i]);
    printf("transpose test ok.\n");

    assert(tmp2->rows == 9 && tmp2->cols == 8);
    assert((err = multiply_matrix(tmp1, tmp2, &tmp3)) == COMPLETED);
    assert(tmp3 != NULL);
    assert(tmp3 != tmp1 && tmp3 != tmp2);

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            assert(fabsf(tmp3->arr[i * 8 + j] - (float)(((i + j) % 2 == 0) ? 285 : 165)) <= FLOAT_ERR);

    delete_matrix(&tmp1);
    delete_matrix(&tmp2);
    delete_matrix(&tmp3);
    printf("muliplcation test ok.\n");

    print_test_end("Matrix multiplication test");
}

void test_matrix_mul_performance(void)
{
    print_test_init("Randomized matrix multiplication performance test");

    matrix op1 = NULL, op2 = NULL, op3 = NULL;
    matrix_errno err;

    int sz = 2048;
    assert((op1 = create_matrix(sz, sz)) != NULL);
    assert((op2 = create_matrix(sz, sz)) != NULL);

    srand(189231273); // Pick a whatever value
    for (int i = 0; i < sz; ++i)
        for (int j = 0; j < sz; ++j)
        {
            op1->arr[i * sz + j] = (float)rand();
            op2->arr[i * sz + j] = (float)rand();
        }

    assert((err = multiply_matrix(op1, op2, &op3)) == COMPLETED);

    delete_matrix(&op1);
    delete_matrix(&op2);
    delete_matrix(&op3);

    print_test_end("Randomized matrix multiplication performance test");
}
