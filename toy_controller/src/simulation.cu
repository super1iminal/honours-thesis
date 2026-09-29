#include "simulation.h"
#include "logger.h"
#include "utility.h" // For cudaTry and generate_random_numbers

// Initialize Samples with random data
Samples* samples_init_random(int n) {
    Samples* samples = (Samples*)malloc(sizeof(Samples));
    if (samples == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Samples.\n");
        return NULL;
    }
    samples->n = n;
    cudaTry(cudaMalloc((void**)&samples->d_x, n * sizeof(float2)));
    samples->h_x = (float2*)malloc(n * sizeof(float2));
    if (samples->h_x == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for Samples.\n");
        cudaTry(cudaFree(samples->d_x));
        free(samples);
        return NULL;
    }

    // Generate initial random states
    auto now = std::chrono::high_resolution_clock::now();
    unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    cudaTry(cudaDeviceSynchronize());
    generate_random_numbers<<<(n + 255) / 256, 256>>>(samples->d_x, n, seed);
    cudaTry(cudaMemcpy(samples->h_x, samples->d_x, n * sizeof(float2), cudaMemcpyDeviceToHost));
    return samples;
}

Samples* samples_init(int n) {
    Samples* samples = (Samples*)malloc(sizeof(Samples));
    if (samples == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Samples.\n");
        return NULL;
    }
    samples->n = n;
    cudaTry(cudaMalloc((void**)&samples->d_x, n * sizeof(float2)));
    samples->h_x = (float2*)malloc(n * sizeof(float2));
    if (samples->h_x == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for Samples.\n");
        cudaTry(cudaFree(samples->d_x));
        free(samples);
        return NULL;
    }
    return samples;
}

// Initialize Samples with deterministic data
Samples* samples_init_deterministic(float2* initial_samples, int n, Logger *logger) {
    Samples* samples = (Samples*)malloc(sizeof(Samples));
    if (samples == NULL) {
        log(logger, LOG_ERROR, "ERROR: Unable to allocate memory for Samples.\n");
        return NULL;
    }
    samples->n = n;
    cudaTry(cudaMalloc((void**)&samples->d_x, n * sizeof(float2)));
    samples->h_x = (float2*)malloc(n * sizeof(float2));
    if (samples->h_x == NULL) {
        log(logger, LOG_ERROR, "ERROR: Unable to allocate host memory for Samples.\n");
        cudaTry(cudaFree(samples->d_x));
        free(samples);
        return NULL;
    }

    // Set initial states
    memcpy(samples->h_x, initial_samples, n * sizeof(float2));
    cudaTry(cudaMemcpy(samples->d_x, samples->h_x, n * sizeof(float2), cudaMemcpyHostToDevice));
    // log_states(logger, samples, NULL); // Uncomment if needed
    return samples;
}

// Free Samples
void samples_free(Samples* samples) {
    if (samples != NULL) {
        free(samples->h_x);
        cudaTry(cudaFree(samples->d_x));
        free(samples);
    }
}

// Step function
void step(Samples* samples, Sim_Metadata* meta, Logger* logger) {
    auto now = std::chrono::high_resolution_clock::now();
    unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    cudaTry(cudaDeviceSynchronize());
    update_positions<<<(samples->n + 255) / 256, 256>>>(samples->d_x, samples->n, meta->kp, meta->kd, seed);
    cudaTry(cudaDeviceSynchronize());
    cudaTry(cudaMemcpy(samples->h_x, samples->d_x, samples->n * sizeof(float2), cudaMemcpyDeviceToHost));
    return;
}

// Kernel to update positions
__global__ void update_positions(float2 *x, int n, float kp, float kd, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Initialize RNG state once per thread
        curandState state;
        curand_init(seed + idx, idx, 0, &state);

        // Generate initial position and velocity
        float2 disturbance = curand_normal2(&state);
        // Calculate velocity correction
        float correction = kp * x[idx].x + kd * x[idx].y;

        // Apply velocity to position
        x[idx].x += x[idx].y * 1.0f; // Time step of 1.0

        // Apply correction
        x[idx].y += correction;

        // Apply disturbance
        x[idx].x += disturbance.x;
        x[idx].y += disturbance.y;
    }
}