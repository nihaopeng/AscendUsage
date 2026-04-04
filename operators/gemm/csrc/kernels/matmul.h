//csrc/kernels/add.h

#include "acl/acl.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    void launch_matmul(aclrtStream stream, int numBlocks, uint8_t *A, uint8_t *B, uint8_t *C, uint32_t m, uint32_t k, uint32_t n);
}