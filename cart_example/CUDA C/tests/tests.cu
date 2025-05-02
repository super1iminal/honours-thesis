// tests.cu
#include "../src/utility.cuh"
#include "../src/simulation.cuh"
#include "../src/cubic_spline.cuh"

////////////////////////////////////////////////////////////////////////////////
// MATRIX TESTS
////////////////////////////////////////////////////////////////////////////////

// Host Tests ==================================================================
//------------------------------------------------------------------------------
// Host test: Create a host matrix with a specific fill value and verify
// the dimensions and element values via both operator() and operator[]
//------------------------------------------------------------------------------
void testHostMatrixWithFill()
{
    const int rows = 4, cols = 5;
    const float fill = 2.5f;
    Matrix m(rows, cols, fill, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Check all elements using operator()
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float val = m(i, j);
            assert(val == fill);
        }
    }

    // Check all elements using operator[]
    for (int i = 0; i < rows; ++i)
    {
        float *rowPtr = m[i];
        for (int j = 0; j < cols; ++j)
        {
            assert(rowPtr[j] == fill);
        }
    }
}

//------------------------------------------------------------------------------
// Host test: Create a host matrix using the delegating constructor (default fill 0.f)
//------------------------------------------------------------------------------
void testHostMatrixDefault()
{
    const int rows = 3, cols = 3;
    Matrix m(rows, cols, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Check that every element is 0.0
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            assert(m(i, j) == 0.0f);
        }
    }
}

void testHostMatrixWithData()
{
    const int rows = 3, cols = 3;
    float **data = new float *[rows];
    for (int i = 0; i < rows; ++i)
    {
        data[i] = new float[cols];
        for (int j = 0; j < cols; ++j)
        {
            data[i][j] = i * cols + j;
        }
    }

    Matrix m(rows, cols, data, PROCESSOR::HOST);

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the host memory
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            assert(m(i, j) == data[i][j]);
        }
    }

    for (int i = 0; i < rows; ++i)
    {
        delete[] data[i];
    }
    delete[] data;
}

// Device Tests ==================================================================
//------------------------------------------------------------------------------
// Helper function to verify that a device matrix contains the expected fill value.
// It launches a kernel that copies the matrix values to an output array, then
// copies that data back to the host for verification.
//------------------------------------------------------------------------------
void verifyDeviceMatrix(const Matrix &m, float *expected_data)
{
    float *h_out = new float[m.size()];
    cudaTry(cudaMemcpy(h_out, m.data(), m.size() * sizeof(float), cudaMemcpyDeviceToHost));

    assert(within_threshold(h_out, expected_data, m.size(), 1e-5));

    delete[] h_out;
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix with a specific (non-zero) fill value and verify
// the dimensions and element values via a CUDA kernel.
//------------------------------------------------------------------------------
void testDeviceMatrixWithFill()
{
    const int rows = 6, cols = 9;
    const float fill = 1.17;
    Matrix m(rows, cols, fill, PROCESSOR::DEVICE);

    // setup expected fill list
    float *expected_data = new float[m.size()];
    for (int i = 0; i < m.size(); ++i)
    {
        expected_data[i] = fill;
    }

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory
    verifyDeviceMatrix(m, expected_data);

    delete[] expected_data;
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix using the delegating constructor (default fill 0.f)
//------------------------------------------------------------------------------
void testDeviceMatrixDefault()
{
    const int rows = 5, cols = 5;
    Matrix m(rows, cols, PROCESSOR::DEVICE);

    // setup expected fill list
    float *expected_data = new float[m.size()];
    for (int i = 0; i < m.size(); ++i)
    {
        expected_data[i] = m.default_fill;
    }

    // Verify dimensions
    assert(m.rows() == rows);
    assert(m.cols() == cols);
    assert(m.size() == rows * cols);

    // Verify the contents of the device memory (all should be 0.0)
    verifyDeviceMatrix(m, expected_data);

    delete[] expected_data;
}

//------------------------------------------------------------------------------
// Device test: Create a device matrix using a pre-defined list of data
//------------------------------------------------------------------------------
void testDeviceMatrixWithData()
{
    const int rows = 3, cols = 3;
    float **data = new float *[rows];
    for (int i = 0; i < rows; ++i)
    {
        data[i] = new float[cols];
        for (int j = 0; j < cols; ++j)
        {
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
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            assert(data[i][j] == h_out[i * cols + j]);
        }
    }

    delete[] h_out;
    for (int i = 0; i < rows; ++i)
    {
        delete[] data[i];
    }
    delete[] data;
}

////////////////////////////////////////////////////////////////////////////////
// CUBIC SPLINE TESTS
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// Utility function: Compare two float values for approximate equality
//------------------------------------------------------------------------------
bool closeEnough(float a, float b, float tol = 1e-4f)
{
    float diff = fabsf(a - b);
    if (diff <= tol)
        return true;
    // relative check in case numbers are large or small
    float maxAB = fmaxf(fabsf(a), fabsf(b));
    return (diff <= tol * maxAB);
}

//------------------------------------------------------------------------------
// Launch wrapper for kernelNotAKnotMin
//------------------------------------------------------------------------------
void runNotAKnotKernel(const Matrix &m_in_host, Matrix &m_out_host)
{
    // Create device copies
    //   m_in_host -> m_in_dev
    //   m_out_host -> m_out_dev
    // We'll assume m_in_host has shape (m, n) and m_out_host has shape (m, 1).
    assert(m_in_host.proc() == PROCESSOR::HOST && "Input must be on host for this test function.");
    assert(m_out_host.proc() == PROCESSOR::HOST && "Output must be on host for this test function.");
    assert(m_out_host.rows() == m_in_host.rows() && m_out_host.cols() == 1 && "Output must be shape (m,1).");

    int m = m_in_host.rows();
    int n = m_in_host.cols();

    // 1) Copy input matrix to device
    Matrix m_in_dev(m, n, PROCESSOR::DEVICE); // defaults to fill=0, we will copy data next
    cudaTry(cudaMemcpy(m_in_dev.data(), m_in_host.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

    // 2) Create device output matrix
    Matrix m_out_dev(m, 1, PROCESSOR::DEVICE); // fill=0

    // 3) Launch the kernel
    dim3 block(128);
    dim3 grid((m + block.x - 1) / block.x);
    kernelNotAKnotMin<<<grid, block>>>(m_in_dev, m_out_dev);
    cudaTry(cudaGetLastError());
    cudaTry(cudaDeviceSynchronize());

    // 4) Copy device output back to host
    cudaTry(cudaMemcpy(m_out_host.data(), m_out_dev.data(), m * sizeof(float), cudaMemcpyDeviceToHost));
}

//------------------------------------------------------------------------------
// Test 1: Basic constant function => minimal value is constant everywhere
//------------------------------------------------------------------------------
void test_constant_function()
{

    // Suppose we have M=3 splines, each with length N=5, all values=7.f
    int M = 3;
    int N = 5;
    Matrix m_in_host(M, N, 7.0f, PROCESSOR::HOST); // fill with 7.0

    // Output shape is (M, 1)
    Matrix m_out_host(M, 1, PROCESSOR::HOST);

    // Run the kernel
    runNotAKnotKernel(m_in_host, m_out_host);

    // Check results: the min of a constant function of 7 is obviously 7
    for (int i = 0; i < M; ++i)
    {
        float val = m_out_host[i][0];
        assert(closeEnough(val, 7.0f, 1e-6f) && "Min of constant function should be 7.0");
    }
}

//------------------------------------------------------------------------------
// Test 2: A single spline with a known shape
//         We'll create a shape with N=5 points (0..4) => y = [1, 3, 2, 0, 2]
//         The minimum (of the natural or not-a-knot spline) should be near x=3.
//         We won't do a fully rigorous manual solution, but we can compare
//         to a reference run from a CPU version or from known partial calculations.
//         For simplicity, we just compare to an approximate known min ~ 0.0.
//------------------------------------------------------------------------------
void test_known_shape()
{

    int M = 1;
    int N = 5;
    Matrix m_in_host(M, N, PROCESSOR::HOST, false); // We'll fill data manually
    // Fill row 0: 1, 3, 2, 0, 2
    m_in_host[0][0] = 1.f;
    m_in_host[0][1] = 3.f;
    m_in_host[0][2] = 2.f;
    m_in_host[0][3] = 0.f;
    m_in_host[0][4] = 2.f;

    // Create output
    Matrix m_out_host(M, 1, PROCESSOR::HOST);

    // Launch
    runNotAKnotKernel(m_in_host, m_out_host);

    // We expect something near 0.0 as the min. The actual spline's local minimum
    // is around index ~3, but let's see if the interpolation dips slightly below 0
    // or not. Let's just check that it is around 0. We'll require val <= 0.1
    float val = m_out_host[0][0];
    // We just check it's near zero (or possibly negative if the spline dips below data[3]=0).
    // In not-a-knot, it might be exactly at or just below 0, typically in [ -0.05, 0.05].
    assert(closeEnough(val, -0.104182f, 0.02f) && "Min is expected to be near -0.104182f for this shape.");
    // from 
        // https://tools.timodenk.com/cubic-spline-interpolation
        // and 
        // https://www.desmos.com/calculator

}

//------------------------------------------------------------------------------
// Test 3: Multiple different splines with easy shapes
//         E.g., row 0 is ascending, row 1 is descending, row 2 is sinusoidal, etc.
//------------------------------------------------------------------------------
void test_multiple_splines()
{

    int M = 3; // number of splines
    int N = 5; // length of each spline

    Matrix m_in_host(M, N, PROCESSOR::HOST, false);
    // Spline 1 (row 0): strictly increasing => min at left
    //   10, 11, 12, 13, 14
    m_in_host[0][0] = 10.f;
    m_in_host[0][1] = 11.f;
    m_in_host[0][2] = 12.f;
    m_in_host[0][3] = 13.f;
    m_in_host[0][4] = 14.f;

    // Spline 2 (row 1): strictly decreasing => min at right
    //   14, 13, 12, 11, 10
    m_in_host[1][0] = 14.f;
    m_in_host[1][1] = 13.f;
    m_in_host[1][2] = 12.f;
    m_in_host[1][3] = 11.f;
    m_in_host[1][4] = 10.f;

    // Spline 3 (row 2): symmetrical shape => min in the middle
    //   5, 3, 2, 3, 5
    m_in_host[2][0] = 5.f;
    m_in_host[2][1] = 3.f;
    m_in_host[2][2] = 2.f;
    m_in_host[2][3] = 3.f;
    m_in_host[2][4] = 5.f;


    // Output shape is (M, 1)
    Matrix m_out_host(M, 1, PROCESSOR::HOST);

    // Launch the kernel
    runNotAKnotKernel(m_in_host, m_out_host);

    // Check results
    // Row 0 => strictly increasing => min ~ 10
    assert(closeEnough(m_out_host[0][0], 10.f, 0.2f));

    // Row 1 => strictly decreasing => min ~ 10
    assert(closeEnough(m_out_host[1][0], 10.f, 0.2f));

    // Row 2 => shape 5,3,2,3,5 => min around 2
    // The spline might dip slightly below 2 due to the curvature,
    // but it should be close to 2.
    assert(m_out_host[2][0] < 3.f && "Should be near or below 2-2.2 range for the third row.");
}

//------------------------------------------------------------------------------
// Test 4: Stress test with 100,000 rows and 100 columns
//------------------------------------------------------------------------------
void test_stress_increasing()
{

    // 1) Dimensions
    int M = 10000000; // number of rows (splines)
    int N = 100;    // length of each spline

    // 2) Allocate a host Matrix for input
    Matrix m_in_host(M, N, PROCESSOR::HOST, false); // 'false' => no random fill; we fill manually

    // 3) Fill each row with an ascending sequence:
    //    For row i, we can do: row i =>  (i*0.01 + 0), (i*0.01 + 1), (i*0.01 + 2), ...
    //    or simply (0,1,2,...,N-1). The actual offset doesn't matter for the min.
    //    We'll just do (0,1,2,...,N-1).
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            m_in_host[i][j] = static_cast<float>(j);
        }
    }

    // 4) Allocate a host Matrix for output (M x 1)
    Matrix m_out_host(M, 1, PROCESSOR::HOST);

    // 5) Launch the kernel
    runNotAKnotKernel(m_in_host, m_out_host);

    // 6) Verify a subset of results.
    //    The sequence is strictly increasing, so the minimum should be the
    //    first element in each row, i.e. 0.0f. We'll do a partial check to
    //    avoid scanning all 100k if we want to keep time down.
    for (int i = 0; i < M; i += 10000) // check every 10k-th row
    {
        float val = m_out_host[i][0];
        // Should be near zero. We'll allow some tolerance for floating math/curvature.
        if (!closeEnough(val, 0.0f, 0.02f))
        {
            printf("    Failure at row %d: got %f, expected ~0.0\n", i, val);
            assert(false);
        }
    }
}

//------------------------------------------------------------------------------
// Test 5: A single spline with a known shape
//         We'll create a shape with N=4 points (0..3) => y = [100, -5, -5, 100]
//         The minimum (of the natural or not-a-knot spline) should be near x=1.5.
//         I've found the value through manual calculations to be approximately -18.125.
//         We'll check that the computed value is near this.
//------------------------------------------------------------------------------
void test_known_shape_negative()
{

    int M = 1;
    int N = 4;
    Matrix m_in_host(M, N, PROCESSOR::HOST, false); // We'll fill data manually
    // Fill row 0: 1, 3, 2, 0, 2
    m_in_host[0][0] = 100.f;
    m_in_host[0][1] = -5.f;
    m_in_host[0][2] = -5.f;
    m_in_host[0][3] = 100.f;

    // Create output
    Matrix m_out_host(M, 1, PROCESSOR::HOST);

    // Launch
    runNotAKnotKernel(m_in_host, m_out_host);

    float val = m_out_host[0][0];
    // printf("  Computed min = %f\n", val);
    assert(closeEnough(val, -18.125f, 0.2f) && "Min is expected to be near -18.125 for this known shape.");
}

// Simple wrapper that measures time, runs the test, and prints the result.
void runTest(const std::string &testName, void (*testFunc)()) {
    std::cout << "[Running] " << testName << "...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Run the test
    testFunc();

    auto end = std::chrono::high_resolution_clock::now();
    // Calculate duration as a floating-point number in milliseconds
    double durationMs = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "[Success] Test took " << durationMs << " ms.\n\n";
}

//------------------------------------------------------------------------------
// Main function: Runs all the tests
//------------------------------------------------------------------------------
int main()
{
    // Run tests on host-based matrices
    // runTest("Matrix Device init with specified fill", testHostMatrixWithFill);
    // runTest("Matrix Device init with default fill", testHostMatrixDefault);
    // runTest("Matrix Device init with data", testHostMatrixWithData);

    // // Run tests on device-based matrices
    // runTest("Matrix Host init with specified fill", testDeviceMatrixWithFill);
    // runTest("Matrix Host init with default fill", testDeviceMatrixDefault);
    // runTest("Matrix Host init with data", testDeviceMatrixWithData);

    // // Run cubic spline tests
    runTest("CubicSpline Constant function", test_constant_function);
    runTest("CubicSpline Known shape [1,3,2,0,2]", test_known_shape);
    runTest("CubicSpline Known shape [100,-5, -5, 100]", test_known_shape_negative);
    runTest("CubicSpline Multiple different splines in one matrix", test_multiple_splines);
    runTest("CubicSpline Stress test: 10 million rows x 100 cols, strictly increasing", test_stress_increasing); 
        // note: about 1.7s for 10M rows and 0.17s for 1M rows, segfault for 100M rows (too large since 100M * 100 = 40GB)

    std::cout << "All tests completed.\n";
    return 0;
}
