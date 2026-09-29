#pragma once

#include <cuda_runtime.h>
#include "utility.h"
#include <chrono>
#include "logger.h"

// Simulation Metadata
typedef struct Sim_Metadata {
    int n;      // Number of samples
    int t;      // Number of time steps
    int t_curr; // Current time step, 1 is the first
    float kp;   // Proportional gain
    float kd;   // Derivative gain
} Sim_Metadata;

// Samples Struct
typedef struct Samples {
    int n;
    float2 *d_x; // Device samples
    float2 *h_x; // Host samples
} Samples;


// Initializers and destructors for Samples
Samples* samples_init_random(int n);
Samples* samples_init(int n);
Samples* samples_init_deterministic(float2* samples, int n, Logger *logger);
void samples_free(Samples* samples);

// Step function using Samples, Disturbances, and Sim_Metadata
void step(Samples *samples, Sim_Metadata *meta, Logger *logger);

// Kernel function to update positions
__global__ void update_positions(float2 *x, int n, float kp, float kd, unsigned long long seed);
