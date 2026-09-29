#pragma once

// Replace these with user-chosen values eventually
#define EPSILON_BISECT 1e-15f // sqrt of smallest value for float32
#define TARGET_PERCENTAGE 0.9f
#define MAX_ITERATIONS_BISECT 25
#define NUM_ELLIPSES 10

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "utility.h"

// Forward declarations to prevent circular dependencies
typedef struct Samples Samples;
typedef struct Sim_Metadata Sim_Metadata;
typedef struct Logger Logger;

// Function declarations
int bisection_search(Samples *samples, Sim_Metadata *meta, Logger *logger, Ellipse *ellipse);

typedef struct Bisection {
    int num_ellipses;
    Ellipse *d_ellipses; // Device memory for ellipses
    Ellipse *h_ellipses; // Host memory for ellipses
    int *d_num_within_ellipse;
    int *h_num_within_ellipse;
    float *d_distances;
    float *h_distances;
    Ellipse low;
    Ellipse high;
} Bisection;

Bisection* bisection_init(Samples* samples, Logger *logger, int num_ellipses, Ellipse initial_ellipse);
void bisection_free(Bisection* bisection);
void generate_ellipses(Bisection* bisection, Logger* logger);
