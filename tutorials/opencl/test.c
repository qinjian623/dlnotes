#include <OpenCL/opencl.h>
#include <CL/cl.h>

int main(int argc, char *argv[])
{
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0,0,&num);
    if (err != CL_SUCCESS){
        std::cerr << "asdf" << std::endl;
        return -1;
    }
    return 0;
}



