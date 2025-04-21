extern "C" __global__
void VectorAdd(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}