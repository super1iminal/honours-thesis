#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>
#include "utility.h"

// Log levels
#define LOG_ERROR 0
#define LOG_BASIC 1
#define LOG_ADVANCED 2
#define LOG_DEBUG 3

// Forward declarations
typedef struct Samples Samples;
typedef struct Sim_Metadata Sim_Metadata;
typedef struct Logger Logger;

// Logger structure
typedef struct Logger {
    FILE *log_file;
    int log_level;
    int output;
} Logger;

// Function prototypes
void log_states(Logger *logger, Samples *samples, Sim_Metadata *meta);
void log_ellipse(Logger *logger, const Ellipse *ellipse, const int r_id, const int t_curr);
void log(Logger *logger, int msg_log_level, const char *format, ...);
Logger* logger_init(int log_level, int output); // output: -1: plot info, 0 = stdout and file, 1 = file, 2 = stdout
void logger_free(Logger* logger);
