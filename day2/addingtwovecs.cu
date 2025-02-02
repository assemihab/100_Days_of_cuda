#include <iostream>

using namespace std;


__global__ void addvecs(float *A_d,float*B_d,float*C_d, int n)
{
    int i=blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n)
    {
        C_d[i]=A_d[i]+B_d[i];
    }
}


int main()
{
    int n=100;
    float *A_h,*B_h,*C_h;
    A_h=new float[n];
    B_h=new float[n];
    C_h=new float[n];
    for(int i=0;i<n;i++)
    {
        A_h[i]=1;
        B_h[i]=2;
        C_h[i]=8;
    }
    float *A_d,*B_d,*C_d;
    cudaError_t err;

    err = cudaMalloc(&A_d, n * sizeof(float));
    if (err != cudaSuccess) { cout << "CUDA malloc failed for A_d: " << cudaGetErrorString(err) << endl; return -1; }

    err = cudaMalloc(&B_d, n * sizeof(float));
    if (err != cudaSuccess) { cout << "CUDA malloc failed for B_d: " << cudaGetErrorString(err) << endl; return -1; }

    err = cudaMalloc(&C_d, n * sizeof(float));
    if (err != cudaSuccess) { cout << "CUDA malloc failed for C_d: " << cudaGetErrorString(err) << endl; return -1; }
    cudaMemcpy(A_d,A_h,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,n*sizeof(float),cudaMemcpyHostToDevice);

    int threads=256;
    // cout<<ceil(n/256.0)<<endl;
    //return 0;
    addvecs<<<ceil(n/256.0),threads>>>(A_d,B_d,C_d,n);
    cudaMemcpy(C_h,C_d,n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t errr = cudaGetLastError();
    if (errr != cudaSuccess) {
        cout << "CUDA Kernel Error: " << cudaGetErrorString(errr) << endl;
    }

    cout<<C_h[1]<<endl;
    cout<<A_h[1]<<endl;
    cout<<B_h[1]<<endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete [] A_h;
    delete [] B_h;
    delete [] C_h;


}