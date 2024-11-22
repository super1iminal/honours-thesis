// tests.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "simulation.h"  // Include your header files
#include "utility.h"
#include "logger.h"

// Function to test Matrix allocation and deallocation
void test_matrix_allocation() {
    printf("Testing Matrix allocation and deallocation...\n");

    int rows = 5;
    int cols = 5;
    float fill_value = 1.0f;

    Matrix* matrix = matrix_init(rows, cols, fill_value);
    if (matrix == NULL) {
        fprintf(stderr, "Matrix allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Verify that the matrix is filled with the fill_value
    bool success = true;
    for (int i = 0; i < rows * cols; i++) {
        if (matrix->h_data_values[i] != fill_value) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Matrix allocation and fill value verification succeeded.\n");
    } else {
        fprintf(stderr, "Matrix fill value verification failed.\n");
    }

    // Modify some values
    matrix->h_data_values[0] = 42.0f;
    matrix->h_data[1][1] = 24.0f;

    // Copy modified host data to device
    cudaMemcpy(matrix->d_data_values, matrix->h_data_values, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data back from device to host to verify
    float* temp = (float*)malloc(rows * cols * sizeof(float));
    cudaMemcpy(temp, matrix->d_data_values, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    if (temp[0] == 42.0f && temp[cols + 1] == 24.0f) {
        printf("Matrix value modification and device synchronization succeeded.\n");
    } else {
        fprintf(stderr, "Matrix value modification failed.\n");
    }

    free(temp);
    matrix_free(matrix);
    printf("Matrix deallocation succeeded.\n");
}

// Function to test Samples allocation and deallocation
void test_samples_allocation() {
    printf("Testing Samples allocation and deallocation...\n");

    int n = 5;
    int stride = 3;

    Samples* samples = samples_init_random(n, stride);
    if (samples == NULL) {
        fprintf(stderr, "Samples allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Verify that the samples contain data from a normal distribution
    bool success = true;
    for (int i = 0; i < n * stride; i++) {
        float value = samples->h_x_values[i];
        // For normal distribution, most values lie within [-3, 3]
        if (value < -10.0f || value > 10.0f) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Samples allocation and random data generation succeeded.\n");
    } else {
        fprintf(stderr, "Samples data verification failed.\n");
    }

    // Modify some values
    samples->h_x_values[0] = 0.5f;
    samples->h_x[2][1] = 0.75f;

    // Copy modified host data to device
    cudaMemcpy(samples->d_x_values, samples->h_x_values, n * stride * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data back from device to host to verify
    float* temp = (float*)malloc(n * stride * sizeof(float));
    cudaMemcpy(temp, samples->d_x_values, n * stride * sizeof(float), cudaMemcpyDeviceToHost);

    if (temp[0] == 0.5f && temp[2 * stride + 1] == 0.75f) {
        printf("Samples value modification and device synchronization succeeded.\n");
    } else {
        fprintf(stderr, "Samples value modification failed.\n");
    }

    free(temp);
    samples_free(samples);
    printf("Samples deallocation succeeded.\n");
}


// Main function to run tests
int main() {
    printf("Starting tests...\n");

    test_matrix_allocation();
    test_samples_allocation();

    printf("All tests completed successfully.\n");
    return 0;
}
