#ifndef ICOMPRESSOR_COMPRESSOR_H
#define ICOMPRESSOR_COMPRESSOR_H

#include "../libsoil/include/SOIL.h"

typedef struct image_type {
    unsigned char *img_data;
    int w;
    int h;
    int channels;
} image_type;

typedef struct matrix {
    float **values;
    int L;
    int N;
} matrix;

typedef struct ICMPR_model {
    image_type image;
    matrix X;
    int n;
    int m;
    int p;
    int E_max;
} ICMPR_model;


ICMPR_model * ICMPR_load(char *file_name, int n, int m, int p, int E_max);
void ICMPR_destroy(ICMPR_model *model);
void ICMPR_compress(ICMPR_model *model);
void ICMPR_decompress(ICMPR_model *model);

#endif //ICOMPRESSOR_COMPRESSOR_H
