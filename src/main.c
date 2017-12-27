#include <stdio.h>
#include "include/compressor.h"

int main(int argc, char **argv) {
    if(argc < 3){
        printf("%s: usage: %s <filename> <filename>\n", argv[0], argv[0]);
        return 1;
    }

    ICMPR_model *pICMPR_model = ICMPR_load(argv[1], 5, 3, 128, 50);

    ICMPR_destroy(pICMPR_model);

    return 0;
}