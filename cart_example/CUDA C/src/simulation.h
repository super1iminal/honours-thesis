#pragma once

#include <cuda_runtime.h>
#include "utility.h"
#include <chrono>
#include "logger.h"

// ==================== STRUCTS ====================
// Simulation Metadata
typedef struct Sim_Metadata {
    int n;      // Number of samples
    int t;      // Number of time steps
    int t_curr; // Current time step, 1 is the first
    float kp;   // Proportional gain
    float kd;   // Derivative gain
    float2 nom; // Nominal position
} Sim_Metadata;

// Samples Struct
typedef struct Samples {
    int n;
    int stride;
    float **h_x;           // Host array of pointers to samples
    float **d_x;           // Device array of pointers to samples
    float *h_x_values;     // Contiguous host memory for sample values
    float *d_x_values;     // Contiguous device memory for sample values
} Samples;

typedef struct Matrix {
    int rows;
    int cols;
    float **h_data;         // Host array of pointers to rows
    float **d_data;         // Device array of pointers to rows
    float *h_data_values;   // Contiguous host data
    float *d_data_values;   // Contiguous device data
} Matrix;

// d dimensions (size of vector), output is a d-dimensional vector
// if output is null, should modify the input vector
typedef void (*NNFunction)(float* input, int d, float* output);

// struct of functions
typedef struct Functions {
    NNFunction* functions;
    int n;
} Functions;

// ==================== INIT FUNCTIONS ====================
// Initializers and destructors for Samples
Samples* samples_init_random(int n, int stride);
Samples* samples_init(int n, int stride);
Samples* samples_init_deterministic(float* samples, int n, int stride, Logger* logger);
Matrix* matrix_init(int rows, int cols, float fill = 0.f);

// ==================== FREE FUNCTIONS ====================
void samples_free(Samples* samples);
void matrix_free(Matrix* matrix);


// ==================== NN FUNCTIONS ====================
// Host function to launch the ReLU kernel
void launch_relu(float* input, int d, float* output);
// ReLU kernel
__global__ void relu_kernel(float* input, int size, float* output);


// ==================== SIMULATION FUNCTIONS ====================
// Step function using Samples, Disturbances, and Sim_Metadata
void step(Samples *samples, Sim_Metadata *meta, Logger *logger, Functions* functions);
// Kernel function to update positions
__global__ void update_positions(float* x, int n, float kp, float kd, unsigned long long seed);

