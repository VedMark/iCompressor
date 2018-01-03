#ifndef ICOMPRESSOR_COMPRESSOR_H
#define ICOMPRESSOR_COMPRESSOR_H

#include "../libsoil/include/SOIL.h"
#include <gsl/gsl_blas.h>

#define MEM_ERR (1)
#define ALG_ERR (-1)
#define SUCCESS (0)

typedef struct image_type {
    unsigned char *img_data;
    int w;
    int h;
    int channels;
} image_type;

typedef struct ICMPR_model {
    image_type image;
    gsl_matrix *X;
    gsl_matrix *W;
    gsl_matrix *W_astric;
    unsigned long rectM;
    unsigned long n;
    unsigned long m;
    unsigned long p;
    double E_max;
} ICMPR_model;


ICMPR_model * ICMPR_load(char *file_name, unsigned long n, unsigned long m, unsigned long p, double E_max);
void ICMPR_destroy(ICMPR_model *model);
int ICMPR_train(ICMPR_model *model);
int ICMPR_restore(ICMPR_model *model, char *file_name);

#endif //ICOMPRESSOR_COMPRESSOR_H
