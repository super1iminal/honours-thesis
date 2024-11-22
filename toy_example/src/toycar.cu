#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include "tc_util.h"

/*
p0 - the distribution of initial states
N - the number of Monte Carlo simulations
y -  a user defined confidence level
T - the time horizon of interest
M - the system model under test

F_0:T = {F_0, F_1, ..., F_T} - the user provided failure regions over the time horizon T of interest
X_0:T <- N Monte Carlo simulated trajectories with model M with initial states drawn from p0
R_0:T = {R(0, L_1), R(1, L_2), ..., R(T, L_T)} - the set of regions defined by X_0:T+1 and F_0:T in equation 11*
    recall that regions change between every time step

t <- 0


equations:
11: 

*/

__global__ void generate_random_numbers(float *random_numbers, int n, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        random_numbers[idx] = curand_normal(&state); // curand_normal works too
    }
}

int main(int argc, char* argv[]) {
    int n;
    if (argc < 2) {
        n = 10;
    } else {
        n = atoi(argv[1]);
    }
    
    float *d_random_numbers, *h_random_numbers;

    h_random_numbers = (float*)malloc(n * sizeof(float));
    cudaTry(cudaMalloc((void**)&d_random_numbers, n * sizeof(float)));

    unsigned long long seed = (unsigned long long)time(NULL);
    generate_random_numbers<<<1, n>>>(d_random_numbers, n, seed);
    // 1 block, n threads, all in x dimension

    cudaTry(cudaMemcpy(h_random_numbers, d_random_numbers, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%f\n", h_random_numbers[i]);
    }

    free(h_random_numbers);
    cudaTry(cudaFree(d_random_numbers));

    return 0;
}