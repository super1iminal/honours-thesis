#include "simulation.h"
#include "logger.h"
#include "utility.h" // For cudaTry and generate_random_numbers


// ==================== INIT FUNCTIONS ====================
// Initialize Samples with random data
Samples* samples_init_random(int n, int stride) {
    Samples* samples = (Samples*)malloc(sizeof(Samples));
    if (samples == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Samples.\n");
        return NULL;
    }
    samples->n = n;
    samples->stride = stride;

    // Allocate host memory for h_x pointers
    samples->h_x = (float**)malloc(n * sizeof(float*));
    if (samples->h_x == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_x pointers.\n");
        free(samples);
        return NULL;
    }

    // Allocate contiguous host memory for h_x_values
    samples->h_x_values = (float*)malloc(n * stride * sizeof(float));
    if (samples->h_x_values == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_x values.\n");
        free(samples->h_x);
        free(samples);
        return NULL;
    }

    // Initialize h_x pointers to point into h_x_values
    for (int i = 0; i < n; i++) {
        samples->h_x[i] = samples->h_x_values + i * stride;
    }

    // Allocate contiguous device memory for d_x_values
    cudaTry(cudaMalloc((void**)&(samples->d_x_values), n * stride * sizeof(float)));

    // Allocate host memory for temporary array of device sample pointers
    float** d_x_host_pointers = (float**)malloc(n * sizeof(float*));
    if (d_x_host_pointers == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for d_x_host_pointers.\n");
        cudaTry(cudaFree(samples->d_x_values));
        free(samples->h_x_values);
        free(samples->h_x);
        free(samples);
        return NULL;
    }

    // Initialize the host array of device sample pointers
    for (int i = 0; i < n; i++) {
        d_x_host_pointers[i] = samples->d_x_values + i * stride;
    }

    // Allocate device memory for d_x pointers (array of pointers to device samples)
    cudaTry(cudaMalloc((void**)&(samples->d_x), n * sizeof(float*)));

    // Copy the array of device sample pointers to device memory
    cudaTry(cudaMemcpy(samples->d_x, d_x_host_pointers, n * sizeof(float*), cudaMemcpyHostToDevice));

    // Free temporary host pointers
    free(d_x_host_pointers);

    // Generate random data on the device
    int total_elements = n * stride;
    auto now = std::chrono::high_resolution_clock::now();
    unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    // Launch kernel to fill d_x_values with random numbers
    int gridSize = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_random_numbers<<<gridSize, BLOCK_SIZE>>>(samples->d_x_values, total_elements, seed);
    cudaTry(cudaDeviceSynchronize());

    // Copy the random data from device to host
    cudaTry(cudaMemcpy(samples->h_x_values, samples->d_x_values, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    return samples;
}

// Initialize Samples without data
Samples* samples_init(int n, int stride) {
    Samples* samples = (Samples*)malloc(sizeof(Samples));
    if (samples == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Samples.\n");
        return NULL;
    }
    samples->n = n;
    samples->stride = stride;

    // Allocate host memory for h_x pointers
    samples->h_x = (float**)malloc(n * sizeof(float*));
    if (samples->h_x == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_x pointers.\n");
        free(samples);
        return NULL;
    }

    // Allocate contiguous host memory for h_x_values
    samples->h_x_values = (float*)malloc(n * stride * sizeof(float));
    if (samples->h_x_values == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_x values.\n");
        free(samples->h_x);
        free(samples);
        return NULL;
    }

    // Initialize h_x pointers to point into h_x_values
    for (int i = 0; i < n; i++) {
        samples->h_x[i] = samples->h_x_values + i * stride;
    }

    // Allocate contiguous device memory for d_x_values
    cudaTry(cudaMalloc((void**)&(samples->d_x_values), n * stride * sizeof(float)));

    // Allocate host memory for temporary array of device sample pointers
    float** d_x_host_pointers = (float**)malloc(n * sizeof(float*));
    if (d_x_host_pointers == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for d_x_host_pointers.\n");
        cudaTry(cudaFree(samples->d_x_values));
        free(samples->h_x_values);
        free(samples->h_x);
        free(samples);
        return NULL;
    }

    // Initialize the host array of device sample pointers
    for (int i = 0; i < n; i++) {
        d_x_host_pointers[i] = samples->d_x_values + i * stride;
    }

    // Allocate device memory for d_x pointers (array of pointers to device samples)
    cudaTry(cudaMalloc((void**)&(samples->d_x), n * sizeof(float*)));

    // Copy the array of device sample pointers to device memory
    cudaTry(cudaMemcpy(samples->d_x, d_x_host_pointers, n * sizeof(float*), cudaMemcpyHostToDevice));

    // Free temporary host pointers
    free(d_x_host_pointers);

    return samples;
}

// Initialize Samples with deterministic data
Samples* samples_init_deterministic(float* initial_samples, int n, int stride) {
    Samples* samples = samples_init(n, stride);
    if (samples == NULL) {
        return NULL;
    }

    int total_elements = n * stride;

    // Copy initial samples into h_x_values
    memcpy(samples->h_x_values, initial_samples, total_elements * sizeof(float));

    // Copy data from host to device
    cudaTry(cudaMemcpy(samples->d_x_values, samples->h_x_values, total_elements * sizeof(float), cudaMemcpyHostToDevice));

    return samples;
}

Matrix* matrix_init(int rows, int cols, float fill) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Matrix.\n");
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;

    // Allocate host memory for h_data pointers
    matrix->h_data = (float**)malloc(rows * sizeof(float*));
    if (matrix->h_data == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_data pointers.\n");
        free(matrix);
        return NULL;
    }

    // Allocate contiguous host memory for h_data_values
    matrix->h_data_values = (float*)malloc(rows * cols * sizeof(float));
    if (matrix->h_data_values == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for h_data values.\n");
        free(matrix->h_data);
        free(matrix);
        return NULL;
    }

    // Initialize h_data pointers to point into h_data_values
    for (int i = 0; i < rows; i++) {
        matrix->h_data[i] = matrix->h_data_values + i * cols;
    }

    // Fill h_data_values with the specified fill value
    for (int i = 0; i < rows * cols; i++) {
        matrix->h_data_values[i] = fill;
    }

    // Allocate contiguous device memory for d_data_values
    cudaTry(cudaMalloc((void**)&(matrix->d_data_values), rows * cols * sizeof(float)));

    // Copy h_data_values to d_data_values
    cudaTry(cudaMemcpy(matrix->d_data_values, matrix->h_data_values, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate host memory for temporary array of device row pointers
    float** d_data_host_pointers = (float**)malloc(rows * sizeof(float*));
    if (d_data_host_pointers == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate host memory for d_data_host_pointers.\n");
        cudaTry(cudaFree(matrix->d_data_values));
        free(matrix->h_data_values);
        free(matrix->h_data);
        free(matrix);
        return NULL;
    }

    // Initialize the host array of device row pointers
    for (int i = 0; i < rows; i++) {
        d_data_host_pointers[i] = matrix->d_data_values + i * cols;
    }

    // Allocate device memory for d_data pointers (array of pointers to device rows)
    cudaTry(cudaMalloc((void**)&(matrix->d_data), rows * sizeof(float*)));

    // Copy the array of device row pointers to device memory
    cudaTry(cudaMemcpy(matrix->d_data, d_data_host_pointers, rows * sizeof(float*), cudaMemcpyHostToDevice));

    // Free temporary host pointers
    free(d_data_host_pointers);

    return matrix;
}

// ==================== FREE FUNCTIONS ====================

// Free Samples
void samples_free(Samples* samples) {
    if (samples != NULL) {
        // Free device memory
        if (samples->d_x != NULL) cudaTry(cudaFree(samples->d_x));
        if (samples->d_x_values != NULL) cudaTry(cudaFree(samples->d_x_values));

        // Free host memory
        if (samples->h_x_values != NULL) free(samples->h_x_values);
        if (samples->h_x != NULL) free(samples->h_x);

        // Free the Samples struct
        free(samples);
    }
}

void matrix_free(Matrix* matrix) {
    if (matrix != NULL) {
        // Free device memory
        if (matrix->d_data != NULL) cudaTry(cudaFree(matrix->d_data));
        if (matrix->d_data_values != NULL) cudaTry(cudaFree(matrix->d_data_values));

        // Free host memory
        if (matrix->h_data_values != NULL) free(matrix->h_data_values);
        if (matrix->h_data != NULL) free(matrix->h_data);

        // Free the Matrix struct
        free(matrix);
    }
}


// ==================== NN FUNCTIONS ====================
void launch_relu(float* input, int d, float* output) {
    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (d + blockSize - 1) / blockSize;

    // Launch the ReLU kernel
    relu_kernel<<<numBlocks, blockSize>>>(input, d, output);
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}

__global__ void relu_kernel(float* input, int size, float* output)  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float val = input[idx];
        if (val < 0) val = 0;
        if (output != NULL) {
            output[idx] = val;
        } else {
            input[idx] = val;
        }
    }
}

// ==================== SIM FUNCTIONS ====================

// Step function
void step(Samples* samples, Sim_Metadata* meta, Logger* logger, Functions* functions) {
    // auto now = std::chrono::high_resolution_clock::now();
    // unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    // cudaTry(cudaDeviceSynchronize());
    // update_positions<<<(samples->n*samples->stride + 255) / 256, 256>>>(samples->d_x, samples->n, samples->stride, meta->kp, meta->kd, seed);
    // cudaTry(cudaDeviceSynchronize());
    // cudaTry(cudaMemcpy(samples->h_x, samples->d_x, samples->n * samples->stride * sizeof(float), cudaMemcpyDeviceToHost));
    // return;
}

// Kernel to update positions
__global__ void update_positions(float *x, int n, int stride, float kp, float kd, unsigned long long seed) {
    // int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride;
    // if (idx < n * stride) {
    //     // Initialize RNG state once per thread
    //     curandState state;
    //     curand_init(seed + idx, idx, 0, &state);

    //     // Generate initial position and velocity
    //     float disturbance = curand_normal(&state);
    //     // Calculate velocity correction
    //     float correction = kp * x[idx].x + kd * x[idx].y;

    //     // Apply velocity to position
    //     x[idx].x += x[idx].y * 1.0f; // Time step of 1.0

    //     // Apply correction
    //     x[idx].y += correction;

    //     // Apply disturbance
    //     x[idx].x += disturbance.x;
    //     x[idx].y += disturbance.y;
    // }
}



