#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Forward declarations
typedef struct Samples Samples;
typedef struct Sim_Metadata Sim_Metadata;
typedef struct Iteration_Info Iteration_Info;
typedef struct Logger Logger;

// Typedefs and function prototypes
typedef unsigned long long uint64;

// Utility function prototypes
extern void rand_vector(void *v, uint n, uint elem_size);
extern void usage(int i, char **opt_names);
extern uint get_uint(int argc, char **argv, const char **optnames, int i, uint v0);
extern double get_double(int argc, char **argv, const char **optnames, int i, double v0);
extern double mean(uint n, double sum);
extern double stdev(uint n, double sum, double sum_sq);
extern void _cudaTry(cudaError_t cuda_return, const char *FileName, int line);

// ==================== STRUCTS ====================
// Ellipse structure
typedef struct Ellipse {
    float a;
    float b;
    float theta;
} Ellipse;

// ==================== UTIL FUNCTIONS ====================
// Macro for CUDA error checking
#define cudaTry(cudaStatus) _cudaTry(cudaStatus, __FILE__, __LINE__)

// Function prototypes
float find_min(float arr[], int n);
float find_max(float arr[], int n);
void linspace(float a, float b, float *r, int x);
int eq_arrays(float* arr1, float* arr2, int n, float threshold);

// ==================== SIM-RELATED UTIL FUNCTIONS ====================
// CUDA kernels
__global__ void generate_random_numbers(float *data, int size, unsigned long long seed);
__global__ void set_to_zero(int *arr, int n);

void generate_points(Samples* samples, Ellipse* h_ellipse, int k, float kp, float kd, Logger*logger);
// new stuff
__device__ float2 step_toy(float2 x, float2 disturbance, float kp, float kd);
__device__ float2 generate_candidate_point(int idx, unsigned long long seed, float kp, float kd, int k);
__device__ int check_acceptance(float2 point, Ellipse* ellipse);
__global__ void generate_points_kernel(float2* global_output, Ellipse* ellipse, unsigned long long seed, int max_threads, unsigned int* total_accepted_global, float kp, float kd, int k);