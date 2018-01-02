#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include "include/compressor.h"

#define C_MAX (255)

double adaptive_step(gsl_vector *vector);
unsigned char *at(image_type *image, int i, int j);

void init_normal_dist(gsl_matrix *matrix);
void init_uniform_dist(gsl_matrix *matrix);

void normalize(gsl_matrix *matrix);
double normalize_value(unsigned char value);
image_type restore_image(gsl_matrix *matrix, unsigned long rectM,
                         unsigned long n, unsigned long m);
unsigned char restore_value(double value);
void split_on_blocks(ICMPR_model *model);
int train_step(ICMPR_model *model, gsl_vector *X, double *alpha, double *alpha_);
int update_W(ICMPR_model *model,
             gsl_matrix *X,
             gsl_matrix *delta_X,
             double *alpha);
int update_W_astric(ICMPR_model *model,
                    gsl_matrix *Y,
                    gsl_matrix *delta_X,
                    double *alpha);


unsigned char *at(image_type *image, int i, int j) {
    return image->img_data + i * image->w * 3 + j;
}

void init_normal_dist(gsl_matrix *matrix) {
    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

    for(size_t i = 0; i < matrix->size1; ++i) {
        for(size_t j = 0; j < matrix->size2; ++j) {
            gsl_matrix_set(matrix, i, j, gsl_ran_gaussian(r, .1));
        }
    }
    gsl_rng_free(r);
}

void init_uniform_dist(gsl_matrix *matrix) {
    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

    for(size_t i = 0; i < matrix->size1; ++i) {
        for(size_t j = 0; j < matrix->size2; ++j) {
            gsl_matrix_set(matrix, i, j, gsl_ran_flat(r, -.2, .2));
        }
    }
    gsl_rng_free(r);
}

ICMPR_model *ICMPR_load(char *file_name, unsigned long n, unsigned long m,
                        unsigned long p, double E_max) {
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

    model->X = gsl_matrix_alloc(model->image.w * model->image.h / (n * m),
                                3 * n * m);
    if(NULL == model->X) return NULL;

    model->W = gsl_matrix_alloc(3 * n * m, p);
    if(NULL == model->W) return NULL;
    init_uniform_dist(model->W);

    model->W_astric = gsl_matrix_alloc(p, 3 * n * m);
    if(NULL == model->W_astric) return NULL;

    for(size_t i = 0; i < model->W_astric->size1; ++i) {
        for(size_t j = 0; j < model->W_astric->size2; ++j) {
            gsl_matrix_set(model->W_astric, i, j,
                           gsl_matrix_get(model->W, j, i));
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
            gsl_matrix_set(model->X,
                           (i / model->n) * model->rectM + j / model->m,
                           (i * model->m + j % model->m) % model->X->size2,
                           normalize_value(*at(&model->image, i, j)));
        }
    }
}

image_type restore_image(gsl_matrix *matrix, unsigned long rectM,
                         unsigned long n, unsigned long m) {
    image_type img;
    img.w = (int) (matrix->size1 * matrix->size2 / (3 * n * rectM));
    img.h = (int) (matrix->size1 * matrix->size2 / (3 * img.w));
    img.channels = SOIL_LOAD_RGB;

    img.img_data = malloc(3 * img.w * img.h * sizeof(char));

    for(int i = 0; i < img.h; ++i) {
        for(int j = 0; j < 3 * img.w; ++j) {
            *at(&img, i, j) = restore_value(gsl_matrix_get(
                    matrix,
                    (i / n) * rectM + j / m,
                    (i * m + j % m) % matrix->size2));
        }
    }

    return img;
}

double normalize_value(unsigned char value) {
    return (2.0 * value / C_MAX) - 1;
}

unsigned char restore_value(double value) {
    double result = (C_MAX * (value + 1) / 2);
    return (unsigned char) (result < 0 ? 0 : result > 255 ? 255 : result);
}

void ICMPR_destroy(ICMPR_model *model) {
    SOIL_free_image_data(model->image.img_data);
    gsl_matrix_free(model->X);
    gsl_matrix_free(model->W);
    gsl_matrix_free(model->W_astric);
    free(model);
}

double summary_deviation(ICMPR_model *model) {
    gsl_matrix *Y;
    gsl_matrix *X_astric;
    double dev = 0;

    Y = gsl_matrix_alloc(model->X->size1, model->W->size2);
    X_astric = gsl_matrix_alloc(Y->size1, model->W_astric->size2);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                                 1, model->X, model->W, 0, Y);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                                 1, Y, model->W_astric, 0, X_astric);

    for(size_t i = 0; i < model->X->size1; ++i) {
        for(size_t j = 0; j < model->X->size2; ++j) {
            dev += pow(gsl_matrix_get(X_astric, i, j)
                       - gsl_matrix_get(model->X, i, j), 2);
        }
    }
    gsl_matrix_free(Y);
    gsl_matrix_free(X_astric);
    return dev;
}

int ICMPR_train(ICMPR_model *model) {
    int ret_val = 0;
    double dev = 0;
    gsl_vector *row = NULL;
    double alpha = 0;
    double alpha_ = 0;
    int step = 1;

    row = gsl_vector_alloc(model->X->size2);

    printf("-- training model\n");

    normalize(model->W);
    normalize(model->W_astric);

    do {
        for(size_t i = 0; i < model->X->size1; ++i) {
            gsl_matrix_get_row(row, model->X, i);
            ret_val = train_step(model, row, &alpha, &alpha_);
            if(MEM_ERR == ret_val) return MEM_ERR;
            if(ALG_ERR == ret_val) return ALG_ERR;
        }

        dev = summary_deviation(model);

        printf("step: %d; alpha: %.6lf; alpha': %.6lf; deviation: %.6lf\n",
               step++, alpha, alpha_, dev);
    } while(dev >= model->E_max);

    gsl_vector_free(row);
    return SUCCESS;
}

int train_step(register ICMPR_model *model,
               gsl_vector *v,
               double *alpha,
               double *alpha_) {

    gsl_matrix *X = NULL;
    gsl_matrix *Y = NULL;
    gsl_matrix *X_ = NULL;
    int ret_val = 0;

    X = gsl_matrix_alloc(1, model->X->size2);
    if(NULL == X) return MEM_ERR;
    gsl_matrix_set_row(X, 0, v);

    Y = gsl_matrix_alloc(X->size1, model->W->size2);
    if(NULL == Y) return MEM_ERR;
    X_ = gsl_matrix_alloc(Y->size1, model->W_astric->size2);
    if(NULL == X_) return MEM_ERR;

    ret_val = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                             1, X, model->W, 0, Y);
    if(0 != ret_val) return ALG_ERR;

    ret_val = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                             1, Y, model->W_astric, 0, X_);
    if(0 != ret_val) return ALG_ERR;

    ret_val = gsl_matrix_sub(X_, X);
    if(0 != ret_val) return ALG_ERR;

    if(MEM_ERR == update_W(model, X, X_, alpha)) return MEM_ERR;
    if(MEM_ERR == update_W_astric(model, Y, X_, alpha_)) return MEM_ERR;

//    normalize(model->W);
//    normalize(model->W_astric);

    gsl_matrix_free(X);
    gsl_matrix_free(Y);
    gsl_matrix_free(X_);

    return SUCCESS;
}

double adaptive_step(gsl_vector *vector) {
    float sum = 0;

    for(size_t j = 0; j < vector->size; ++j) {
        sum += gsl_vector_get(vector, j);
    }

    return 1. / (sum + 100);
}

int update_W(ICMPR_model *model,
             gsl_matrix *X,
             gsl_matrix *delta_X,
             double *alpha) {

    gsl_matrix *X_deltaX = NULL;
    gsl_matrix *X_deltaX_W = NULL;
    int ret_val = 0;

    X_deltaX = gsl_matrix_alloc(X->size2, delta_X->size2);
    if(NULL == X_deltaX) return MEM_ERR;
    X_deltaX_W = gsl_matrix_alloc(X_deltaX->size1, model->W_astric->size1);
    if(NULL == X_deltaX_W) return MEM_ERR;

    ret_val = gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                             1, X, delta_X, 0, X_deltaX);
    if(0 != ret_val) return ALG_ERR;

    ret_val = gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                             1, X_deltaX, model->W_astric, 0, X_deltaX_W);
    if(0 != ret_val) return ALG_ERR;

    gsl_vector *row = NULL;
    row = gsl_vector_alloc(X->size2);
    gsl_matrix_get_row(row, X, 0);
    *alpha = adaptive_step(row);

    ret_val = gsl_matrix_scale(X_deltaX_W, *alpha);
    if(0 != ret_val) return ALG_ERR;

    ret_val = gsl_matrix_sub(model->W, X_deltaX_W);
    if(0 != ret_val) return ALG_ERR;

    gsl_vector_free(row);
    gsl_matrix_free(X_deltaX);
    gsl_matrix_free(X_deltaX_W);

    return SUCCESS;
}

int update_W_astric(ICMPR_model *model,
                    gsl_matrix *Y,
                    gsl_matrix *delta_X,
                    double *alpha) {


    gsl_matrix *Y_deltaX = NULL;
    int ret_val = 0;

    Y_deltaX = gsl_matrix_alloc(Y->size2, delta_X->size2);
    if(NULL == Y_deltaX) return MEM_ERR;

    ret_val = gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                             1, Y, delta_X, 0, Y_deltaX);
    if(0 != ret_val) return ALG_ERR;

    gsl_vector *row = NULL;
    row = gsl_vector_alloc(Y->size2);
    gsl_matrix_get_row(row, Y, 0);
    *alpha = adaptive_step(row);

    ret_val = gsl_matrix_scale(Y_deltaX, *alpha);
    if(0 != ret_val) return ALG_ERR;
    ret_val = gsl_matrix_sub(model->W_astric, Y_deltaX);
    if(0 != ret_val) return ALG_ERR;

    gsl_vector_free(row);
    gsl_matrix_free(Y_deltaX);

    return SUCCESS;
}

void normalize(gsl_matrix *matrix) {
    float sum = 0;

    for(size_t j = 0; j < matrix->size2; ++j) {
        sum = 0;
        for(size_t i = 0; i < matrix->size1; ++i) {
            sum += pow(gsl_matrix_get(matrix, i, j), 2);
        }
        sum = sqrtf(sum);
        for(size_t i = 0; i < matrix->size1; ++i) {
            gsl_matrix_set(matrix, i, j, gsl_matrix_get(matrix, i, j) / sum);
        }
    }
}

int ICMPR_restore(ICMPR_model *model, char *file_name) {
    gsl_matrix *Y;
    gsl_matrix *X_astric;
    int ret_val = 0;

    Y = gsl_matrix_alloc(model->X->size1, model->W->size2);
    X_astric = gsl_matrix_alloc(Y->size1, model->W_astric->size2);

    ret_val = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                             1, model->X, model->W, 0, Y);
    if(0 != ret_val) return ALG_ERR;

    ret_val = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                             1, Y, model->W_astric, 0, X_astric);
    if(0 != ret_val) return ALG_ERR;

    image_type cmpr_img = restore_image(X_astric,
                                        model->rectM,
                                        model->n,
                                        model->m);

    SOIL_save_image
            (
                    file_name,
                    SOIL_SAVE_TYPE_BMP,
                    cmpr_img.w, cmpr_img.h, cmpr_img.channels,
                    cmpr_img.img_data
            );

    printf("-- image restored to file: %s\n", file_name);

    gsl_matrix_free(Y);
    gsl_matrix_free(X_astric);
    free(cmpr_img.img_data);

    return SUCCESS;
}
