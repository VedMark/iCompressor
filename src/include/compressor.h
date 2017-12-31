#ifndef ICOMPRESSOR_COMPRESSOR_H
#define ICOMPRESSOR_COMPRESSOR_H

#include "../libsoil/include/SOIL.h"

#define MEM_ERR (1)
#define ALG_ERR (-1)
#define SUCCESS (0)

typedef struct image_type {
    unsigned char *img_data;
    int w;
    int h;
    int channels;
} image_type;

typedef struct matrix_type {
    double **values;
    int n;
    int m;
} matrix_type;

typedef struct ICMPR_model {
    image_type image;
    matrix_type X;
    matrix_type W;
    matrix_type W_astric;
    int rectM;
    int n;
    int m;
    int p;
    double E_max;
} ICMPR_model;


ICMPR_model * ICMPR_load(char *file_name, int n, int m, int p, double E_max);
void ICMPR_destroy(ICMPR_model *model);
int ICMPR_train(ICMPR_model *model);
int ICMPR_restore(ICMPR_model *model, char *file_name);

#endif //ICOMPRESSOR_COMPRESSOR_H
