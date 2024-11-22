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

// Forward declarations
typedef struct Samples Samples;
typedef struct Sim_Metadata Sim_Metadata;
typedef struct Logger Logger;
typedef struct Bisection Bisection;

// Hyperparameters
#define PERTURBATION 9.478947e-3f
#define LEARNING_RATE 4.268421e-2f
#define EPSILON_GRAD 2e-4f
#define MAX_ITERATIONS_GRAD 100
#define MAX_ITERATIONS_SINCE_BEST 15

typedef struct Gradient {
    Ellipse* ellipse;
    Ellipse* perturbed_ellipse;
    float* grad;
    float perturbation;
    float initial_learning_rate;
    float threshold;
    int max_iterations;
} Gradient;

Gradient* gradient_init(float perturbation, float initial_learning_rate);
void gradient_free(Gradient* gradient);
void hyperparameter_tuning(Samples *samples, Sim_Metadata *meta, Logger* logger);
float gradient_descent(Samples *samples, Sim_Metadata *meta, Logger *logger, float perturbation, float initial_learning_rate, Ellipse* best_ellipse);
float calculate_area(const Ellipse *ellipse);
int check_valid_ellipse(Ellipse *ellipse, Logger *logger);
