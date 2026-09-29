#include "cubic_spline.cuh"

const int MAX_N = 100;

__device__ void solveNotAKnotSystem(const float *__restrict__ y,
                                    int n,
                                    float *__restrict__ c)
{
    // The idea is to transform the not-a-knot system into a pure tridiagonal system
    // by eliminating the first and last variables using the not-a-knot conditions

    // Handle edge cases
    if (n <= 3)
    {
        for (int i = 0; i < n; i++)
        {
            c[i] = 0.0f;
        }
        return;
    }

    // not-a-knot conditions:
    // 1. c₀ - 2c₁ + c₂ = 0  =>  c₀ = 2c₁ - c₂
    // 2. c_{n-3} - 2c_{n-2} + c_{n-1} = 0  =>  c_{n-1} = 2c_{n-2} - c_{n-3}

    // solve a reduced (n-2)×(n-2) tridiagonal system for c₁, c₂, ..., c_{n-2}
    // the first and last variables (c₀ and c_{n-1}) will be computed afterward

    float a[MAX_N]; // lower diagonal
    float b[MAX_N]; // main diagonal
    float d[MAX_N]; // upper diagonal
    float r[MAX_N]; // right-hand side

    // First, set up the reduced system for variables c₁ through c_{n-2}

    // First equation (for c₁):
    // Substitute c₀ = 2c₁ - c₂ into the original equation for c₁:
    // c₀ + 4c₁ + c₂ = 6(y₂-2y₁+y₀)
    // (2c₁-c₂) + 4c₁ + c₂ = 6(y₂-2y₁+y₀)
    // 6c₁ = 6(y₂-2y₁+y₀)
    // c₁ = y₂-2y₁+y₀
    b[0] = 6.0f; // Coefficient for c₁ (was 4, but +2 from substitution)
    d[0] = 0.0f; // Coefficient for c₂ (was 1, but -1 from substitution)
    r[0] = 6.0f * (y[2] - 2.0f * y[1] + y[0]);

    // Interior equations (for c₂ through c_{n-3}):
    // These remain unchanged since they don't involve c₀ or c_{n-1}
    for (int i = 1; i < n - 3; i++)
    {
        a[i] = 1.0f; // Coefficient for c_{i}
        b[i] = 4.0f; // Coefficient for c_{i+1}
        d[i] = 1.0f; // Coefficient for c_{i+2}
        r[i] = 6.0f * (y[i + 2] - 2.0f * y[i + 1] + y[i]);
    }

    // Last equation (for c_{n-2}):
    // Substitute c_{n-1} = 2c_{n-2} - c_{n-3} into the original equation for c_{n-2}:
    // c_{n-3} + 4c_{n-2} + c_{n-1} = 6(y_{n-1}-2y_{n-2}+y_{n-3})
    // c_{n-3} + 4c_{n-2} + (2c_{n-2}-c_{n-3}) = 6(y_{n-1}-2y_{n-2}+y_{n-3})
    // 6c_{n-2} = 6(y_{n-1}-2y_{n-2}+y_{n-3})
    // c_{n-2} = y_{n-1}-2y_{n-2}+y_{n-3}
    a[n - 3] = 0.0f; // Coefficient for c_{n-3} (was 1, but -1 from substitution)
    b[n - 3] = 6.0f; // Coefficient for c_{n-2} (was 4, but +2 from substitution)
    r[n - 3] = 6.0f * (y[n - 1] - 2.0f * y[n - 2] + y[n - 3]);

    // Now solve the tridiagonal system using the Thomas algorithm

    // Forward elimination
    for (int i = 1; i < n - 2; i++)
    {
        float m = a[i] / b[i - 1];
        b[i] -= m * d[i - 1];
        r[i] -= m * r[i - 1];
    }

    // Back substitution
    float temp_c[MAX_N]; // Temporary array to store c₁ through c_{n-2}
    temp_c[n - 3] = r[n - 3] / b[n - 3];

    for (int i = n - 4; i >= 0; i--)
    {
        temp_c[i] = (r[i] - d[i] * temp_c[i + 1]) / b[i];
    }

    // Now recover c₀ and c_{n-1} using the not-a-knot conditions
    // c₀ = 2c₁ - c₂
    c[0] = 2.0f * temp_c[0] - temp_c[1];

    // Copy c₁ through c_{n-2}
    for (int i = 1; i <= n - 2; i++)
    {
        c[i] = temp_c[i - 1];
    }

    // c_{n-1} = 2c_{n-2} - c_{n-3}
    c[n - 1] = 2.0f * temp_c[n - 3] - temp_c[n - 4];
}

// Updated splineValue function to match SciPy's coefficients format
__device__ float splineValue(int i, float x, const float *y, const float *c)
{
    // Compute polynomial coefficients for the interval [i, i+1]
    // Assuming uniform spacing h = 1
    float h = 1.0f;

    // Using the standard cubic spline formula:
    // S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3

    // Compute coefficients
    float a = y[i]; // constant term

    // Calculate b_i (coefficient of linear term)
    float b = (y[i + 1] - y[i]) / h - h * (2.0f * c[i] + c[i + 1]) / 6.0f;

    // c_i is already the coefficient of quadratic term divided by 2
    float c_coef = c[i] / 2.0f;

    // d_i is the coefficient of cubic term divided by 6
    float d_coef = (c[i + 1] - c[i]) / (6.0f * h);

    // Calculate the value using Horner's method
    float dx = x - (float)i; // x - x_i
    return a + dx * (b + dx * (c_coef + dx * d_coef));
}

// Updated splineDeriv function for consistency
__device__ float splineDeriv(int i, float x, const float *y, const float *c)
{
    float h = 1.0f;

    // First derivative of the spline at x
    float b = (y[i + 1] - y[i]) / h - h * (2.0f * c[i] + c[i + 1]) / 6.0f;
    float c_coef = c[i] / 2.0f;
    float d_coef = (c[i + 1] - c[i]) / (6.0f * h);

    float dx = x - (float)i;
    return b + dx * (2.0f * c_coef + 3.0f * dx * d_coef);
}

// Main function to find the minimum
__device__ float notAKnotCubicSplineMin(const float *y, int n)
{
    const int MAX_N = 100; // As specified in your notes
    float c[MAX_N];        // Second derivatives at knots

    // Solve for second derivatives
    solveNotAKnotSystem(y, n, c);

// Debug: Print spline coefficients
#ifdef DEBUG_PRINT
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Second derivatives: ");
        for (int i = 0; i < n; i++)
        {
            printf("%f ", c[i]);
        }
        printf("\n");

        printf("Intervals: ");
        for (int i = 0; i < n - 1; i++)
        {
            printf("[%d,%d] ", i, i + 1);
        }
        printf("\n");

        for (int i = 0; i < n - 1; i++)
        {
            float h = 1.0f;
            float a = y[i];
            float b = (y[i + 1] - y[i]) / h - h * (2.0f * c[i] + c[i + 1]) / 6.0f;
            float c_coef = c[i] / 2.0f;
            float d_coef = (c[i + 1] - c[i]) / (6.0f * h);

            printf("d (x^3): %f ", d_coef * 6); // Multiply by 6 to match with actual coef
            printf("c (x^2): %f ", c_coef * 2); // Multiply by 2 to match with actual coef
            printf("b (x): %f ", b);
            printf("a (const): %f ", a);
            printf("\n");
        }
    }
#endif

    // Find the global minimum
    float globalMinVal = y[0];

    // Evaluate at all data points
    for (int i = 0; i < n; i++)
    {
        if (y[i] < globalMinVal)
        {
            globalMinVal = y[i];
        }
    }

    // Find potential minima in each interval
    for (int i = 0; i < n - 1; i++)
    {
        // Compute coefficients for the derivative polynomial
        float h = 1.0f;
        float b = (y[i + 1] - y[i]) / h - h * (2.0f * c[i] + c[i + 1]) / 6.0f;
        float c_coef = c[i] / 2.0f;
        float d_coef = (c[i + 1] - c[i]) / (6.0f * h);

        // Derivative: b + 2*c_coef*dx + 3*d_coef*dx^2
        // Solve quadratic: 3*d_coef*dx^2 + 2*c_coef*dx + b = 0
        float alpha = 3.0f * d_coef;
        float beta = 2.0f * c_coef;
        float gamma = b;

        if (fabsf(alpha) < 1e-10f)
        {
            // Linear equation (if cubic term is negligible)
            if (fabsf(beta) > 1e-10f)
            {
                float dx = -gamma / beta;
                if (dx > 0.0f && dx < 1.0f)
                {
                    float x = i + dx;
                    float val = splineValue(i, x, y, c);
                    if (val < globalMinVal)
                    {
                        globalMinVal = val;
                    }
                }
            }
        }
        else
        {
            // Quadratic equation
            float discriminant = beta * beta - 4.0f * alpha * gamma;
            if (discriminant >= 0.0f)
            {
                float sqrtD = sqrtf(discriminant);
                float dx1 = (-beta + sqrtD) / (2.0f * alpha);
                float dx2 = (-beta - sqrtD) / (2.0f * alpha);

                // Check if roots are in [0,1]
                if (dx1 >= 0.0f && dx1 <= 1.0f)
                {
                    float x = i + dx1;
                    float val = splineValue(i, x, y, c);
                    if (val < globalMinVal)
                    {
                        globalMinVal = val;
                    }
                }

                if (dx2 >= 0.0f && dx2 <= 1.0f)
                {
                    float x = i + dx2;
                    float val = splineValue(i, x, y, c);
                    if (val < globalMinVal)
                    {
                        globalMinVal = val;
                    }
                }
            }
        }
    }

    return globalMinVal;
}

// A sample kernel that processes many splines in parallel. Each thread
// calls notAKnotCubicSplineMin() for its own subarray of length n.
// @param m_in: Input matrix of splines, each row is a spline of length n.
// m x n, where n is the length of each spline and m is the number of splines.
// @param m_out: Output matrix of minimum values, one per spline.
// m x 1, where m is the number of splines.
// @note: This kernel is not optimized for large n, as it uses a single thread
// @note: Both matrices should be stored in row-major order.
__global__ void kernelNotAKnotMin(Matrix m_in, Matrix m_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m_in.rows())
        return;

    float *d_in = m_in.data();
    float *d_out = m_out.data();
    int n = m_in.cols(); // n is the length of each spline
    const float *arr = d_in + idx * n;
    d_out[idx] = notAKnotCubicSplineMin(arr, n);
}
