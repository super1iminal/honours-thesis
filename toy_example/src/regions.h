#pragma once

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

typedef struct Gradient Gradient;
typedef struct Logger Logger;

typedef struct Iteration_Info {
    Ellipse *h_ellipse;
    Samples *samples;
} Iteration_Info;

// Updated function prototypes
void find_regions(Sim_Metadata *meta, Logger *logger, int num_regions);

Iteration_Info* iteration_init(int n);
void iteration_free(Iteration_Info* info);
