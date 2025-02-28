// tests.cu
#include "simulation.cuh"  // Include your header files
#include "utility.cuh"

// =============================================================================
// Matrix Tests
//
// =============================================================================

// HOST TESTS ==================================================================
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

void testHostMatrixWithData() {
    const int rows = 3, cols = 3;
    float **data = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new float[cols];
        for (int j = 0; j < cols; ++j) {
            data[i][j] = i * cols + j;
        }
    }

    Matrix m(rows, cols, data, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the host memory
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            assert(m(i, j) == data[i][j]);
        }
    }

    for (int i = 0; i < rows; ++i) {
        delete[] data[i];
    }
    delete[] data;

    std::cout << "testHostMatrixWithData passed.\n";
}



// DEVICE TESTS ==================================================================
//------------------------------------------------------------------------------
// Helper function to verify that a device matrix contains the expected fill value.
// It launches a kernel that copies the matrix values to an output array, then
// copies that data back to the host for verification.
//------------------------------------------------------------------------------
void verifyDeviceMatrix(const Matrix &m, float* expected_data) {
    float *h_out = new float[m.size()];
    cudaTry(cudaMemcpy(h_out, m.data(), m.size() * sizeof(float), cudaMemcpyDeviceToHost));

    assert(within_threshold(h_out, expected_data, m.size(), 1e-5));

    delete[] h_out;
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix with a specific (non-zero) fill value and verify
// the dimensions and element values via a CUDA kernel.
//------------------------------------------------------------------------------
void testDeviceMatrixWithFill() {
    const int rows = 6, cols = 9;
    const float fill = 1.17;
    Matrix m(rows, cols, fill, PROCESSOR::DEVICE);

    // setup expected fill list
    float *expected_data = new float[m.size()];
    for (int i = 0; i < m.size(); ++i) {
        expected_data[i] = fill;
    }

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory
    verifyDeviceMatrix(m, expected_data);

    delete[] expected_data;

    std::cout << "testDeviceMatrixWithFill passed.\n";
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix using the delegating constructor (default fill 0.f)
//------------------------------------------------------------------------------
void testDeviceMatrixDefault() {
    const int rows = 5, cols = 5;
    Matrix m(rows, cols, PROCESSOR::DEVICE);

    // setup expected fill list
    float *expected_data = new float[m.size()];
    for (int i = 0; i < m.size(); ++i) {
        expected_data[i] = m.default_fill;
    }

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory (all should be 0.0)
    verifyDeviceMatrix(m, expected_data);

    delete[] expected_data;

    std::cout << "testDeviceMatrixDefault passed.\n";
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix using a pre-defined list of data
//------------------------------------------------------------------------------
void testDeviceMatrixWithData() {
    const int rows = 3, cols = 3;
    float **data = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        data[i] = new float[cols];
        for (int j = 0; j < cols; ++j) {
            data[i][j] = i * cols + j;
        }
    }

    Matrix m(rows, cols, data, PROCESSOR::DEVICE);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory
    float *h_out = new float[m.size()];
    cudaTry(cudaMemcpy(h_out, m.data(), m.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify each element is as expected
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            assert(data[i][j] == h_out[i * cols + j]);
        }
    }

    delete[] h_out;
    for (int i = 0; i < rows; ++i) {
        delete[] data[i];
    }
    delete[] data;

    std::cout << "testDeviceMatrixWithData passed.\n";
}

//------------------------------------------------------------------------------
// Main function: Runs all the tests
//------------------------------------------------------------------------------
int main() {
    // Run tests on host-based matrices
    testHostMatrixWithFill();
    testHostMatrixDefault();
    testHostMatrixWithData();

    // Run tests on device-based matrices
    testDeviceMatrixWithFill();
    testDeviceMatrixDefault();
    testDeviceMatrixWithData();

    std::cout << "All tests passed.\n";
    return 0;
}
