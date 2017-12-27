#include <stdlib.h>
#include <stdio.h>
#include "include/compressor.h"

#define C_MAX (255)


unsigned char *at(image_type *image, int i, int j);
float normalize_value(float value);
void split_on_blocks(ICMPR_model *model);


unsigned char *at(image_type *image, int i, int j) {
    return image->img_data + i * image->h + j;
}

ICMPR_model * ICMPR_load(char *file_name, int n, int m, int p, int E_max) {
    ICMPR_model *model = malloc(sizeof(ICMPR_model));
    if(model == NULL) return NULL;

    model->image.img_data = SOIL_load_image
            (
                    file_name,
                    &model->image.w, &model->image.h, &model->image.channels,
                    SOIL_LOAD_RGB
            );

    model->n = n;
    model->m = m;
    model->p = p;
    model->E_max = E_max;

    model->X.N = n * m;
    model->X.L = 3 * model->image.w * model->image.h / model->X.N;

    model->X.values = malloc(model->X.L * sizeof(float *));
    if(model->X.values == NULL) return NULL;

    for(int i = 0; i < model->X.L; ++i) {
        model->X.values[i] = malloc(model->X.N * sizeof(float));
        if(model->X.values[i] == NULL) return NULL;
    }

    split_on_blocks(model);

    printf("model initialized with image: %s\n", file_name);

    return model;
}

void split_on_blocks(ICMPR_model *model) {
    const int N = model->image.h;
    const int M = model->image.w * 3;

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            model->X.values
            [i / model->n + j / model->m]
            [(i * model->m + j) % model->X.N] =
                    normalize_value(*at(&model->image, i, j));
        }
    }
}

float normalize_value(float value) {
    return (2 * value / C_MAX) - 1;
}

void ICMPR_destroy(ICMPR_model *model) {
    SOIL_free_image_data(model->image.img_data);
    for(int i = 0; i < model->image.h; ++i) {
        free(model->X.values[i]);
    }
    free(model->X.values);
    free(model);
}


void ICMPR_compress(ICMPR_model *model) {

}

void ICMPR_decompress(ICMPR_model *model) {

}
