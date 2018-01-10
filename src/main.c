#include <stdio.h>
#include <string.h>
#include "include/compressor.h"


void print_help() {
    printf("Usage: iCompressor <input> <output> <n> <m> <p> <E_max> <F>\n"
                   "   or: iCompressor --help\n\n"
                   "  input\t\tan image file name to be compressed\n"
                   "  output\tan image file name for restored image.\n"
                   "\t\t  Has .bmp format\n"
                   "  n\t\theight of the rectangle that splits the\n"
                   "\t\t  image into input vectors. The height must be\n"
                   "\t\t  divided by the number (0 < n <= h)\n"
                   "  m\t\tthe width of the rectangle that splits the\n"
                   "\t\t  image into input vectors. The number must must be\n"
                   "\t\t  divided by the number (0 < m <= w)\n"
                   "  p\t\tnumber of neurones in the hidden layer\n"
                   "\t\t  (0 < p <= 6*m*n)\n"
                   "  F\t\tvalue to scale adaptive step\n"
                   "  E_max\t\tmaximum standard error. Training ends when\n"
                   "\t\t  standard error on an epoch becomes less than E_max\n"
                   "\t\t  (0 < E_max <= 0.1*p\n");
}

int main(int argc, char **argv) {
    const char *usage = "%s: usage: %s <input> <output> <n> <m> <p> <E_max> <F>\n";
    if(!(argc == 2 || argc == 8)){
        printf(usage, argv[0], argv[0]);
        return 1;
    }

    if (argc == 2) {
        if (!strcmp(argv[1], "--help")) {
            print_help();
            return 0;
        } else {
            printf(usage, argv[0], argv[0]);
            return 1;
        }
    }

    ICMPR_model *pICMPR_model = NULL;
    unsigned long n = 0;
    unsigned long m = 0;
    unsigned long p = 0;
    unsigned long F = 0;
    double E_max = 0;
    int ret_value = 0;

    n = strtoul(argv[3], NULL, 10);
    m = strtoul(argv[4], NULL, 10);
    p = strtoul(argv[5], NULL, 10);
    E_max = strtod(argv[6], NULL);
    F = strtoul(argv[7], NULL, 10);

    pICMPR_model = malloc(sizeof(ICMPR_model));
    ret_value = ICMPR_load(pICMPR_model, argv[1], n, m, p, E_max, F);
    if(MEM_ERR == ret_value) {
        fprintf(stderr, "internal memory error!\n");
        return 1;
    }
    if(IMG_ERR == ret_value) {
        fprintf(stderr, "could not load the image!\n");
        return 1;
    }
    if(PAR_ERR == ret_value) {
        fprintf(stderr, "parameter(s) has wrong value!\n");
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

    ret_value = ICMPR_restore(pICMPR_model, argv[2]);
    if(MEM_ERR == ret_value) {
        fprintf(stderr, "internal memory error!\n");
        return 1;
    }
    if(IMG_ERR == ret_value) {
        fprintf(stderr, "could not save the image!\n");
        return 1;
    }
    if(ALG_ERR == ret_value) {
        fprintf(stderr, "linear algebra error!\n");
        return 1;
    }

    double cmpr_ratio = (double)(pICMPR_model->X->size2 * pICMPR_model->X->size1)
                        / ((pICMPR_model->X->size2 * sizeof(double)
                            + pICMPR_model->X->size1)
                          * pICMPR_model->p + 2 * sizeof(unsigned long));
    fprintf(stdout, "-- compression ratio: %.6f\n", cmpr_ratio);

    ICMPR_destroy(pICMPR_model);

    return 0;
}
