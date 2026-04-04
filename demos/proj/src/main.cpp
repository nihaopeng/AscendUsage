// 一个Host侧文件，通过<<<>>>异构调用Kernel代码
#include "interface.h"
#include "cstdio"
#include "cstdlib"
#include "cstdint"

int main(int argc, char *argv[])
{
    int block_num = atoi(argv[1]);
    int cachelin_sz = atoi(argv[2]);

    uint8_t *res = new uint8_t[block_num];

    int num_arg = 3;
    char* param[] = {(char*)&block_num,(char*)&cachelin_sz,(char*)res};

    foo_and_hello(num_arg,param);

    for (int i = 0 ;i < block_num; i++){
        printf("expected:%d,res:%d\n",i,res[i]);
    }
    return 0;
}