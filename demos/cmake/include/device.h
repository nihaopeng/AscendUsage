#include "acl/acl.h"

extern "C" void launch_foo(int numBlocks,aclrtStream stream,uint8_t *Out,int Stride);

extern "C" void launch_hello_world(int numBlocks, aclrtStream stream,uint8_t *Out, int Stride);