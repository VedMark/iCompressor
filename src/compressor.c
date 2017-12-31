#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "include/compressor.h"

#define C_MAX (255)

#define PRINT_MATRIX(m1) {                      \
    printf("NxM = %dx%d\n", (m1)->n, (m1)->m);  \
    for(int i = 0; i < (m1)->n; ++i) {          \
        for(int j = 0; j < (m1)->m; ++j) {      \
            printf("%f ", (m1)->values[i][j]);  \
        }                                       \
        printf("\n");                           \
    }                                           \
    printf("\n");                               \
}                                               \


double adaptive_step(matrix_type *m);
unsigned char *at(image_type *image, int i, int j);
double deviation(const matrix_type *delta_X);
int diff_vectors(const matrix_type *m1, int i1,
                 const matrix_type *m2, int i2,
                 matrix_type *m_res, int i3);
void free_matrix(matrix_type *m);
int mult_matrixes(const matrix_type *m1,
                  const matrix_type *m2,
                  matrix_type *m_res);
int new_matrix(matrix_type *matrix1, int n, int m);
void normalize(matrix_type *m);
double normalize_value(double value);
image_type restore_image(matrix_type *matrix, int rectM, int n, int m);
double restore_value(double value);
void split_on_blocks(ICMPR_model *model);
int train_step(ICMPR_model *model, int row);
void transpose(const matrix_type *m, matrix_type *m_res);
int update_W(ICMPR_model *model, matrix_type *X, matrix_type *delta_X);
int update_W_astric(ICMPR_model *model,
                    matrix_type *Y,
                    matrix_type *delta_X);


unsigned char *at(image_type *image, int i, int j) {
    return image->img_data + i * image->w * 3 + j;
}

void init_uniform_dist(matrix_type *matrix1) {
    srand((unsigned int) time(NULL));

    for(int i = 0; i < matrix1->n; ++i) {
        for(int j = 0; j < matrix1->m; ++j) {
            matrix1->values[i][j] = (1. * rand() / RAND_MAX * 2 - 1 ) * 0.1;
        }
    }
}

int diff_vectors(const matrix_type *const m1, int i1,
                 const matrix_type *const m2, int i2,
                 matrix_type *m_res, int i3) {

    if(m1->m != m2->m) return ALG_ERR;

    for(int j = 0; j < m1->m; ++j) {
        m_res->values[i3][j] = m1->values[i1][j] - m2->values[i2][j];
    }
    return SUCCESS;
}

int mult_matrixes(const matrix_type *const m1,
                  const matrix_type *const m2,
                  matrix_type *m_res) {
    if(m1->m != m2->n) return ALG_ERR;
    double sum = 0;

    for(int i = 0; i < m1->n; ++i) {
        for(int j = 0; j < m2->m; ++j) {
            sum = 0;
            for(int r = 0; r < m1->m; ++r) {
                sum += m1->values[i][r] * m2->values[r][j];
            }
            m_res->values[i][j] = sum;
        }
    }
    return SUCCESS;
}

void transpose(const matrix_type *const m, matrix_type *m_res) {
    m_res->n = m->m;
    m_res->m = m->n;

    for(int i = 0; i < m_res->n; ++i) {
        for(int j = 0; j < m_res->m; ++j) {
            m_res->values[i][j] = m->values[j][i];
        }
    }
}

int new_matrix(matrix_type *matrix1, int n, int m) {
    matrix1->n = n;
    matrix1->m = m;

    matrix1->values = malloc(matrix1->n * sizeof(double *));
    if(matrix1->values == NULL) return MEM_ERR;

    for(int i = 0; i < matrix1->n; ++i) {
        matrix1->values[i] = malloc(matrix1->m * sizeof(double));
        if(matrix1->values[i] == NULL) return MEM_ERR;
    }
    return SUCCESS;
}

void free_matrix(matrix_type *m) {
    for(int i = 0; i < m->n; ++i) {
        free(m->values[i]);
    }
    free(m->values);
}

ICMPR_model *ICMPR_load(char *file_name, int n, int m, int p, double E_max) {
    ICMPR_model *model = malloc(sizeof(ICMPR_model));
    if(model == NULL) return NULL;

    model->image.img_data = SOIL_load_image
            (
                    file_name,
                    &model->image.w, &model->image.h, &model->image.channels,
                    SOIL_LOAD_RGB
            );
    if(NULL == model->image.img_data) {
        return NULL;
    }

    model->n = n;
    model->m = 3 * m;
    model->p = p;
    model->E_max = E_max;
    model->rectM = 3 * model->image.w / model->m;

    model->X.m = 3 * n * m;
    model->X.n = 3 * model->image.w * model->image.h / model->X.m;

    if(0 != new_matrix(&model->X, model->X.n, model->X.m)) return NULL;
    init_uniform_dist(&model->X);
    if(0 != new_matrix(&model->W, 3 * n * m, p)) return NULL;
    init_uniform_dist(&model->W);
    if(0 != new_matrix(&model->W_astric, p, 3 * n * m)) return NULL;

    for(int i = 0; i < model->W_astric.n; ++i) {
        for(int j = 0; j < model->W_astric.m; ++j) {
            model->W_astric.values[i][j] = model->W.values[j][i];
        }
    }

    split_on_blocks(model);

    fprintf(stdout, "-- model initialized with image: %s\n", file_name);

    return model;
}

void split_on_blocks(ICMPR_model *model) {
    const int N = model->image.h;
    const int M = model->image.w * 3;

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            model->X.values
            [(i / model->n) * model->rectM + j / model->m]
            [(i * model->m + j % model->m) % model->X.m] =
                    normalize_value(*at(&model->image, i, j));
        }
    }
}

image_type restore_image(matrix_type *matrix, int rectM, int n, int m) {
    image_type img;
    img.w = matrix->n * matrix->m / (3 * n * rectM);
    img.h = matrix->n * matrix->m / (3 * img.w);
    img.channels = SOIL_LOAD_RGB;

    img.img_data = malloc(3 * img.w * img.h * sizeof(char));

    for(int i = 0; i < img.h; ++i) {
        for(int j = 0; j < 3 * img.w; ++j) {
            *at(&img, i, j) = (unsigned char)
                    restore_value(matrix->values
                                  [(i / n) * rectM + j / m]
                                  [(i * m + j % m) % m]);
        }
    }

    return img;
}

double normalize_value(double value) {
    return (2 * value / C_MAX) - 1;
}

double restore_value(double value) {
    return C_MAX * (value + 1) / 2;
}

void ICMPR_destroy(ICMPR_model *model) {
    SOIL_free_image_data(model->image.img_data);
    free_matrix(&model->X);
    free_matrix(&model->W);
    free_matrix(&model->W_astric);
    free(model);
}

double summary_deviation(ICMPR_model *model) {
    matrix_type Y;
    matrix_type X_astric;
    double dev = 0;

    if(MEM_ERR == new_matrix(&Y, model->X.n, model->W.m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&X_astric, Y.n, model->W_astric.m))
        return MEM_ERR;

    if(ALG_ERR == mult_matrixes(&model->X, &model->W, &Y))
        return ALG_ERR;

    if(ALG_ERR == mult_matrixes(&Y, &model->W_astric, &X_astric))
        return ALG_ERR;

    for(int i = 0; i < model->X.n; ++i) {
        for(int j = 0; j < model->X.m; ++j) {
            dev += pow(X_astric.values[i][j] - model->X.values[i][j], 2);
        }
    }
    free_matrix(&Y);
    free_matrix(&X_astric);
    return dev;
}

int ICMPR_train(ICMPR_model *model) {
    int ret_val = 0;
    double dev = 0;
    int step = 1;

    printf("-- training model\n");

    do {
        for(int i = 0; i < model->X.n; ++i) {
            ret_val = train_step(model, i);
            if(MEM_ERR == ret_val) return MEM_ERR;
            if(ALG_ERR == ret_val) return ALG_ERR;
        }

        dev = summary_deviation(model);

        printf("traing step: %d; deviation: %.lf\n", step++, dev);
    } while(dev >= model->E_max);

    return SUCCESS;
}

int train_step(ICMPR_model *model, int row) {
    matrix_type X;
    matrix_type Y;
    matrix_type X_astric;
    matrix_type delta_X;

    if(MEM_ERR == new_matrix(&X, 1, model->X.m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&Y, X.n, model->W.m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&X_astric, Y.n, model->W_astric.m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&delta_X, X.n, X.m))
        return MEM_ERR;

    for(int j = 0; j < X.m; ++j) {
        X.values[0][j] = model->X.values[row][j];
    }

//    PRINT_MATRIX(&X);
//    PRINT_MATRIX(&model->W);
//    PRINT_MATRIX(&model->W_astric);
    if(ALG_ERR == mult_matrixes(&X, &model->W, &Y))
        return ALG_ERR;

//    PRINT_MATRIX(&Y);
    if(ALG_ERR == mult_matrixes(&Y, &model->W_astric, &X_astric))
        return ALG_ERR;

//    PRINT_MATRIX(&X_astric);
    if(ALG_ERR == diff_vectors(&X_astric, 0, &X, 0, &delta_X, 0))
        return ALG_ERR;

//    PRINT_MATRIX(&delta_X);

    if(MEM_ERR == update_W(model, &X, &delta_X)) return MEM_ERR;
    if(MEM_ERR == update_W_astric(model, &Y, &delta_X)) return MEM_ERR;

//    PRINT_MATRIX(&model->W);
//    PRINT_MATRIX(&model->W_astric);

//    normalize(&model->W);
//    normalize(&model->W_astric);

//    PRINT_MATRIX(&model->W);
//    PRINT_MATRIX(&model->W_astric);

    free_matrix(&X);
    free_matrix(&Y);
    free_matrix(&X_astric);
    free_matrix(&delta_X);

    return SUCCESS;
}

double adaptive_step(matrix_type *m) {
    double sum = 0;

    for(int j = 0; j < m->m; ++j) {
        sum += m->values[0][j] * m->values[0][j];
    }

   // printf("%f\n", 1 / (sum + 10));
    return 1. / (sum + 10);
}

int update_W(ICMPR_model *model, matrix_type *X, matrix_type *delta_X) {
    matrix_type X_transpose;
    matrix_type W_astric_transpose;
    matrix_type X_deltaX;
    matrix_type X_deltaX_W;

    if(MEM_ERR == new_matrix(&X_transpose, X->m, X->n))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&W_astric_transpose,
                             model->W_astric.m,
                             model->W_astric.n))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&X_deltaX, X_transpose.n, delta_X->m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&X_deltaX_W, X_deltaX.n, W_astric_transpose.m))
        return MEM_ERR;

    transpose(X, &X_transpose);
    transpose(&model->W_astric, &W_astric_transpose);

    mult_matrixes(&X_transpose, delta_X, &X_deltaX);
    mult_matrixes(&X_deltaX, &W_astric_transpose, &X_deltaX_W);

    double alpha = 0.001;

    for(int i = 0; i < X_deltaX_W.n; ++i) {
        for(int j = 0; j < X_deltaX_W.m; ++j) {
            model->W.values[i][j] -= alpha * X_deltaX_W.values[i][j];
        }
    }

    free_matrix(&X_transpose);
    free_matrix(&W_astric_transpose);
    free_matrix(&X_deltaX);
    free_matrix(&X_deltaX_W);

    return SUCCESS;
}

int update_W_astric(ICMPR_model *model,
                    matrix_type *Y,
                    matrix_type *delta_X) {

    matrix_type Y_transpose;
    matrix_type Y_deltaX;

    if(MEM_ERR == new_matrix(&Y_transpose, Y->m, Y->n))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&Y_deltaX, Y_transpose.n, delta_X->m))
        return MEM_ERR;

    transpose(Y, &Y_transpose);

    mult_matrixes(&Y_transpose, delta_X, &Y_deltaX);

    double alpha = 0.001;

    for(int i = 0; i < Y_deltaX.n; ++i) {
        for(int j = 0; j < Y_deltaX.m; ++j) {
            model->W_astric.values[i][j] -= alpha * Y_deltaX.values[i][j];
        }
    }

    free_matrix(&Y_transpose);
    free_matrix(&Y_deltaX);

    return SUCCESS;
}

void normalize(matrix_type *m) {
    double sum = 0;

    for(int j = 0; j < m->m; ++j) {
        sum = 0;
        for(int i = 0; i < m->n; ++i) {
            sum += m->values[i][j] * m->values[i][j];
        }
        sum = sqrt(sum);
        for(int i = 0; i < m->n; ++i) {
            m->values[i][j] /= sum;
        }
    }
}

int ICMPR_restore(ICMPR_model *model, char *file_name) {
    matrix_type Y;
    matrix_type X_astric;

    if(MEM_ERR == new_matrix(&Y, model->X.n, model->W.m))
        return MEM_ERR;
    if(MEM_ERR == new_matrix(&X_astric, Y.n, model->W_astric.m))
        return MEM_ERR;

    if(ALG_ERR == mult_matrixes(&model->X, &model->W, &Y))
        return ALG_ERR;
    if(ALG_ERR == mult_matrixes(&Y, &model->W_astric, &X_astric))
        return ALG_ERR;

    image_type cmpr_img = restore_image(&X_astric,
                                        model->rectM,
                                        model->n,
                                        model->m);

    printf("size: %d %d\n", cmpr_img.w, cmpr_img.h);
    SOIL_save_image
            (
                    file_name,
                    SOIL_SAVE_TYPE_BMP,
                    cmpr_img.w, cmpr_img.h, cmpr_img.channels,
                    cmpr_img.img_data
            );

    printf("-- image restored to file: %s\n", file_name);

    free_matrix(&Y);
    free_matrix(&X_astric);
    free(cmpr_img.img_data);

    return SUCCESS;
}
