//csrc/kernels/add.h

#include "acl/acl.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    void launch_scalar_vadd(aclrtStream stream, int numBlocks, uint8_t *A, uint8_t *B, uint8_t *C, int len);
    void launch_vector_vadd(aclrtStream stream, int numBlocks, uint8_t *A, uint8_t *B, uint8_t *C, int len);
}