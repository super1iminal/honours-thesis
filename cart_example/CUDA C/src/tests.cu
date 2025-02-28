// tests.cu
#include <cuda_runtime.h>
#include <iostream>
#include "simulation.cuh"  // Include your header files
#include "utility.cuh"

//------------------------------------------------------------------------------
// Kernel to copy the matrix values into an output array on the device
//------------------------------------------------------------------------------
__global__
void copy_matrix_kernel(const Matrix* m, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m->size()) {
        // Convert linear index to row and column indices
        int row = idx / m->cols();
        int col = idx % m->cols();
        output[idx] = (*m)(row, col);
    }
}

//------------------------------------------------------------------------------
// Host test: Create a host matrix with a specific fill value and verify 
// the dimensions and element values via both operator() and operator[]
//------------------------------------------------------------------------------
void testHostMatrixWithFill() {
    const int rows = 4, cols = 5;
    const float fill = 2.5f;
    Matrix m(rows, cols, fill, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Check all elements using operator()
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = m(i, j);
            assert(val == fill);
        }
    }

    // Check all elements using operator[]
    for (int i = 0; i < rows; ++i) {
        float* rowPtr = m[i];
        for (int j = 0; j < cols; ++j) {
            assert(rowPtr[j] == fill);
        }
    }

    std::cout << "testHostMatrixWithFill passed.\n";
}

//------------------------------------------------------------------------------
// Host test: Create a host matrix using the delegating constructor (default fill 0.f)
//------------------------------------------------------------------------------
void testHostMatrixDefault() {
    const int rows = 3, cols = 3;
    Matrix m(rows, cols, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Check that every element is 0.0
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            assert(m(i, j) == 0.0f);
        }
    }

    std::cout << "testHostMatrixDefault passed.\n";
}

//------------------------------------------------------------------------------
// Helper function to verify that a device matrix contains the expected fill value.
// It launches a kernel that copies the matrix values to an output array, then
// copies that data back to the host for verification.
//------------------------------------------------------------------------------
void verifyDeviceMatrix(const Matrix &m, float expectedFill) {
    int size = m.size();
    float *d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, size * sizeof(float));
    assert(err == cudaSuccess);

    const int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocks, threadsPerBlock>>>(&m, d_out);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    float *h_out = new float[size];
    err = cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    // Verify each element is as expected (using a small epsilon for floating-point comparison)
    for (int i = 0; i < size; ++i) {
        float diff = fabs(h_out[i] - expectedFill);
        assert(diff < 1e-5);
    }

    delete[] h_out;
    cudaFree(d_out);
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix with a specific (non-zero) fill value and verify
// the dimensions and element values via a CUDA kernel.
//------------------------------------------------------------------------------
void testDeviceMatrixWithFill() {
    const int rows = 6, cols = 7;
    const float fill = 3.14f;
    Matrix m(rows, cols, fill, PROCESSOR::DEVICE);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory
    verifyDeviceMatrix(m, fill);

    std::cout << "testDeviceMatrixWithFill passed.\n";
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix using the delegating constructor (default fill 0.f)
//------------------------------------------------------------------------------
void testDeviceMatrixDefault() {
    const int rows = 5, cols = 5;
    Matrix m(rows, cols, PROCESSOR::DEVICE);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory (all should be 0.0)
    verifyDeviceMatrix(m, 0.0f);

    std::cout << "testDeviceMatrixDefault passed.\n";
}

//------------------------------------------------------------------------------
// Main function: Runs all the tests
//------------------------------------------------------------------------------
int main() {
    // Run tests on host-based matrices
    testHostMatrixWithFill();
    testHostMatrixDefault();

    // Run tests on device-based matrices
    testDeviceMatrixWithFill();
    testDeviceMatrixDefault();

    std::cout << "All tests passed.\n";
    return 0;
}
