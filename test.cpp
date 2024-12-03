#include <iostream>
#include <mkl.h>

int main() {
    char version[198];  // 根据文档，198 是推荐的缓冲区大小
    MKL_Get_Version_String(version, sizeof(version));
    std::cout << "MKL Version: " << version << std::endl;
    return 0;
}
