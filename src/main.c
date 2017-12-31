#include <stdio.h>
#include "include/compressor.h"


int main(int argc, char **argv) {
    if(argc < 2){
        printf("%s: usage: %s <filename> <filename>\n", argv[0], argv[0]);
        return 1;
    }

    int ret_value = 0;
    ICMPR_model *pICMPR_model = NULL;

    pICMPR_model = ICMPR_load(argv[1], 1, 1, 2, 0.0001);
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

    double cmpr_ratio = (double)(pICMPR_model->X.m * pICMPR_model->X.n) /
                       ((pICMPR_model->X.m + pICMPR_model->X.n) * pICMPR_model->p + 2);
    fprintf(stdout, "-- compression ratio: %.4f\n", cmpr_ratio);

    ICMPR_destroy(pICMPR_model);

    return 0;
}