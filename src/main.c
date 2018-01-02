#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include "include/compressor.h"


int main(int argc, char **argv) {
    if(argc < 2){
        printf("%s: usage: %s <filename> <filename>\n", argv[0], argv[0]);
        return 1;
    }

    int ret_value = 0;
    ICMPR_model *pICMPR_model = NULL;

    pICMPR_model = ICMPR_load(argv[1], 8, 8, 48, 256);
    if(NULL == pICMPR_model) {
        fprintf(stderr, "could not load image!\n");
        return 1;
    }

    ret_value = ICMPR_train(pICMPR_model);
    if(MEM_ERR == ret_value) {
        fprintf(stderr, "internal memory error!\n");
        return 1;
    }
    if(ALG_ERR == ret_value) {
        fprintf(stderr, "linear algebra error!\n");
        return 1;
    }

    ret_value = ICMPR_restore(pICMPR_model, "imgg_out_restored.png");
    if(MEM_ERR == ret_value) {
        fprintf(stderr, "internal memory error!\n");
        return 1;
    }
    if(ALG_ERR == ret_value) {
        fprintf(stderr, "linear algebra error!\n");
        return 1;
    }

    double cmpr_ratio = (double)(pICMPR_model->X->size2 * pICMPR_model->X->size1)
                        / ((pICMPR_model->X->size2 * sizeof(float)
                            + pICMPR_model->X->size1)
                          * pICMPR_model->p + 2 * sizeof(unsigned long));
    fprintf(stdout, "-- compression ratio: %.6f\n", cmpr_ratio);

    ICMPR_destroy(pICMPR_model);

    return 0;
}