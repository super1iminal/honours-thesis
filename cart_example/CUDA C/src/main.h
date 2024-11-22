#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>       // For strcmp
#include <sys/resource.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h>
#include <time.h>

// Include updated headers
#include "simulation.h"   // For Samples, Sim_Metadata, and Disturbances
#include "logger.h"
#include "regions.h"
#include "utility.h"
