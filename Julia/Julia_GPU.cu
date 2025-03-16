#include <cuda_runtime.h>
#include <stdio.h>

int main(){
    CPUBitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    bitmap.display_and_exit();
}