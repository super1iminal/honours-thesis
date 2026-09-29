#include "utility.h"
#include "simulation.h"       // For definitions of Samples and Sim_Metadata
#include "bisection.h"        // For full definition of Bisection
#include "regions.h"          // For full definition of Iteration_Info
#include "logger.h"           // For logging functions
#include <vector_types.h>

void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s in %s line %d\n",
                cudaGetErrorString(cudaStatus), fileName, lineNumber);
        exit(1);
    }
}

float find_min(float arr[], int n) {
    float min = arr[0];  // Start with the first element
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

float find_max(float arr[], int n) {
    float max = arr[0];  // Start with the first element
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void linspace(float a, float b, float *r, int x) {
    // Calculate the step size
    float step = (b - a) / (x - 1);

    // Generate the numbers
    for (int i = 0; i < x; i++) {
        r[i] = a + i * step;
    }
    // Ensures the last element is exactly 'b'
    r[x - 1] = b;
}

int eq_arrays(float* arr1, float* arr2, int n, float threshold) {
    if (arr1 == NULL || arr2 == NULL || n <= 0 || threshold < 0.0f) {
        // Invalid input parameters
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (fabsf(arr1[i] - arr2[i]) > threshold) {
            return 0; // Arrays are not equal within the threshold
        }
    }
    return 1; // Arrays are equal within the threshold
}

__global__ void calculate_distances(float2 *x, float *distances, int n, float2 nom) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        distances[idx] = sqrtf(powf(x[idx].x - nom.x, 2.0f) + powf(x[idx].y - nom.y, 2.0f));
    }
}

__global__ void generate_random_numbers(float2 *d_x, int n, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_x[idx] = curand_normal2(&state);
    }
}

__global__ void set_to_zero(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = 0;
    }
}

__global__ void set_to_zero_f2(float2 *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx].x = 0.0f;
        arr[idx].y = 0.0f;
    }
}

void within_ellipses(Bisection* bisection, Samples* samples) {
    set_to_zero<<<((bisection->num_ellipses + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(bisection->d_num_within_ellipse, bisection->num_ellipses);
    within_ellipses_d<<<(samples->n + 255) / 256, 256>>>(samples->d_x, bisection->d_ellipses, bisection->d_num_within_ellipse, bisection->num_ellipses, samples->n);
    cudaTry(cudaDeviceSynchronize());
    cudaTry(cudaMemcpy(bisection->h_num_within_ellipse, bisection->d_num_within_ellipse, bisection->num_ellipses * sizeof(int), cudaMemcpyDeviceToHost));
}

// Checks points within multiple ellipses
__global__ void within_ellipses_d(float2 *x, Ellipse* ellipses, int* num_within_ellipse, int num_ellipses, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float x_val = x[idx].x;
        float y_val = x[idx].y;
        for (int i = 0; i < num_ellipses; i++) {
            // Retrieve ellipse parameters
            float a = ellipses[i].a;
            float b = ellipses[i].b;
            float theta = ellipses[i].theta;

            // Precompute cosine and sine of theta
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            // Rotate point (x_val, y_val) by -theta to align with ellipse axes
            float x_rot = x_val * cos_theta + y_val * sin_theta;
            float y_rot = -x_val * sin_theta + y_val * cos_theta;

            // Compute normalized distance within the ellipse
            float distance = (x_rot / a) * (x_rot / a) + (y_rot / b) * (y_rot / b);

            // Check if the point lies within the ellipse (distance <= 1.0)
            if (distance <= 1.0f) {
                atomicAdd(&num_within_ellipse[i], 1);
            }
        }
    }
}


// // Device function to perform one simulation step
__device__ float2 step_toy(float2 x, float2 disturbance, float kp, float kd) {
    // Calculate velocity correction
    float correction = kp * x.x + kd * x.y;

    // Apply velocity to position
    x.x += x.y * 1.0f; // Time step of 1.0

    // Apply correction
    x.y += correction;

    // Apply disturbance
    x.x += disturbance.x;
    x.y += disturbance.y;

    return x;
}
// Device function to generate a candidate point
__device__ float2 generate_candidate_point(int idx, unsigned long long seed, float kp, float kd, int k) {
    // Initialize RNG state once per thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Generate initial position and velocity
    float2 x = curand_normal2(&state);

    // Simulation loop
    for (int i = 0; i < k; i++) {
        // Generate disturbance
        float2 disturbance = curand_normal2(&state);

        // Perform simulation step
        x = step_toy(x, disturbance, kp, kd);
    }
    return x;
}

// Device function to check if the point is outside the ellipse
__device__ int check_acceptance(float2 point, Ellipse* ellipse) {
    // Check if the point lies outside the ellipse
    float x_val = point.x;
    float y_val = point.y;

    float a = ellipse->a;
    float b = ellipse->b;
    float theta = ellipse->theta;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float x_rot = x_val * cos_theta + y_val * sin_theta;
    float y_rot = -x_val * sin_theta + y_val * cos_theta;

    float distance = (x_rot / a) * (x_rot / a) + (y_rot / b) * (y_rot / b);

    return (distance > 1.0f) ? 1 : 0;
}

// Kernel function with an additional parameter for the accepted count
__global__ void generate_points_kernel(float2* global_output, Ellipse* ellipse, unsigned long long seed, int max_threads, unsigned int* total_accepted_global, float kp, float kd, int k) {
    extern __shared__ int shared_mem[];
    int* flags = shared_mem;                          // Acceptance flags
    int* positions = &shared_mem[blockDim.x];         // Positions after scan

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int accepted = 0;
    float2 candidate_point;

    if (idx < max_threads) {
        // Generate candidate point
        candidate_point = generate_candidate_point(idx, seed, kp, kd, k);

        // Check acceptance
        accepted = check_acceptance(candidate_point, ellipse);
    }

    flags[threadIdx.x] = accepted;
    __syncthreads();

    // Per-block exclusive scan
    int temp;
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        temp = (threadIdx.x >= offset) ? flags[threadIdx.x - offset] : 0;
        __syncthreads();
        flags[threadIdx.x] += temp;
        __syncthreads();
    }

    positions[threadIdx.x] = flags[threadIdx.x] - accepted;
    __syncthreads();

    // flags holds an inclusive scan, so its last entry is already the block's total
    int accepted_in_block = flags[blockDim.x - 1];

    __shared__ unsigned int global_offset;
    if (threadIdx.x == 0) {
        // Atomically update the global accepted count and get the offset
        global_offset = atomicAdd(total_accepted_global, accepted_in_block);
    }
    __syncthreads();

    if (accepted) {
        unsigned int position = global_offset + positions[threadIdx.x];
        global_output[position] = candidate_point;
    }
}

/*
n is number of points needed
k is number of steps that the simulator will take
kp is proportional gain
kd is derivative gain
h_ellipse is the ellipse we're working with
h_output is the output array of points (should be size n)
*/
// Host function
void generate_points(Samples* samples, Ellipse* h_ellipse, int k, float kp, float kd, Logger* logger) {
    int n = samples->n;
    unsigned int total_accepted_host = 0;
    unsigned int prev_total_accepted_host = 0;
    const int MAX_THREADS = n * 2; // Over-generate to ensure enough points

    // Allocate device memory
    float2* d_output;
    float2* h_output = samples->h_x;
    Ellipse* d_ellipse;
    unsigned int* d_total_accepted_global;
    
    // Adjust the size of d_output to be large enough
    // Let's assume we'll need up to 10 times n to be safe
    const int OUTPUT_SIZE = n * 10;
    cudaMalloc((void**)&d_ellipse, sizeof(Ellipse));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float2));
    cudaMalloc((void**)&d_total_accepted_global, sizeof(unsigned int));
    cudaMemcpy(d_ellipse, h_ellipse, sizeof(Ellipse), cudaMemcpyHostToDevice);

    // Initialize total_accepted_global to zero BEFORE the loop
    cudaMemset(d_total_accepted_global, 0, sizeof(unsigned int));

    // Loop until we have at least n accepted points
    int counter = 0;
    while (total_accepted_host < n) {
        auto now = std::chrono::high_resolution_clock::now();
        unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

        int num_blocks = (MAX_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t shared_mem_size = 2 * BLOCK_SIZE * sizeof(int);

        generate_points_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_output, d_ellipse, seed, MAX_THREADS, d_total_accepted_global, kp, kd, k);
        cudaDeviceSynchronize();

        // Copy total_accepted_global to host
        cudaMemcpy(&total_accepted_host, d_total_accepted_global, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Calculate accepted points in this iteration
        unsigned int accepted_in_iteration = total_accepted_host - prev_total_accepted_host;
        prev_total_accepted_host = total_accepted_host;

        if (counter % 250 == 0) {
            log(logger, LOG_ADVANCED, "Accepted in iteration %d: %u\n", counter, accepted_in_iteration);
            log(logger, LOG_ADVANCED, "Total accepted points so far: %u\n", total_accepted_host);
        }
        
        counter++;
    }

    // Ensure we don't copy more data than we have accepted
    unsigned int points_to_copy = (total_accepted_host < n) ? total_accepted_host : n;

    // Copy the first n accepted points to host
    cudaMemcpy(h_output, d_output, points_to_copy * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(samples->d_x, h_output, points_to_copy * sizeof(float2), cudaMemcpyHostToDevice);

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_ellipse);
    cudaFree(d_total_accepted_global);
}
