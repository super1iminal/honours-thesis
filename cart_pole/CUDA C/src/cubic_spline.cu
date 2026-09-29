#include "cubic_spline.cuh"

//---------------------------------------------------------------------
// Solves the not-a-knot cubic spline boundary-value system for second
// derivatives c_i, i=0..n-1, given n data values y[i].
//
// The system is built as follows (uniform spacing = 1):
//
//   Row 0:      c0  - 2*c1  + c2           = 0
//   Row i=1..n-2:  c_{i-1} + 4*c_i + c_{i+1} = 6( y_{i+1} - 2y_i + y_{i-1} )
//   Row n-1:          c_{n-3} - 2*c_{n-2} + c_{n-1} = 0
//
// We'll build an n x n system [A][c] = b, then do a basic
// Gaussian elimination.  For small n, this is acceptable.
//
// c[i] on exit are the second derivatives at each node.
//
__device__ void solveNotAKnotSystem(const float *__restrict__ y,
                                    int n,
                                    float *__restrict__ c)
{
    // We'll store A as a 2D array in column-major or row-major – doesn't matter
    // so long as we index consistently. For n up to a few hundred, alloca usage is
    // typically okay. If n is large, consider other memory strategies.
    float *A = (float *)alloca(n * n * sizeof(float));
    float *b = (float *)alloca(n * sizeof(float));

    // Initialize A to 0
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = 0.0f;
    }

    // Helper lambda to index A(row,col):
    auto idxA = [n](int row, int col) -> int
    {
        return row * n + col; // row-major
    };

    //--------------------------------------------------------------------------
    // 1) Fill row 0:   c0 - 2 c1 + c2 = 0
    //--------------------------------------------------------------------------
    A[idxA(0, 0)] = 1.0f;  // c0
    A[idxA(0, 1)] = -2.0f; // c1
    if (n > 2)
    {
        A[idxA(0, 2)] = 1.0f; // c2
    }
    b[0] = 0.0f;

    //--------------------------------------------------------------------------
    // 2) Fill interior rows i=1..n-2
    //    standard cubic spline interior: c_{i-1} + 4 c_i + c_{i+1} = 6 * [y_{i+1} - 2y_i + y_{i-1}]
    //--------------------------------------------------------------------------
    for (int i = 1; i <= n - 2; ++i)
    {
        A[idxA(i, i - 1)] = 1.0f; // c_{i-1}
        A[idxA(i, i)] = 4.0f;     // c_i
        A[idxA(i, i + 1)] = 1.0f; // c_{i+1}

        b[i] = 6.0f * (y[i + 1] - 2.0f * y[i] + y[i - 1]);
    }

    //--------------------------------------------------------------------------
    // 3) Fill row n-1:   c_{n-3} - 2 c_{n-2} + c_{n-1} = 0
    //--------------------------------------------------------------------------
    if (n >= 3)
    {
        A[idxA(n - 1, n - 3)] = 1.0f;
        A[idxA(n - 1, n - 2)] = -2.0f;
        A[idxA(n - 1, n - 1)] = 1.0f;
    }
    b[n - 1] = 0.0f;

    //--------------------------------------------------------------------------
    // 4) Solve A*c = b via naive Gaussian elimination (no pivoting).
    //    For small n, this is fine. For large n, a pivoting or band solver is better.
    //--------------------------------------------------------------------------
    // Forward elimination
    for (int k = 0; k < n - 1; ++k)
    {
        float pivot = A[idxA(k, k)];
        // In a robust code, check pivot ~ 0 => pivot or partial pivoting
        for (int i = k + 1; i < n; ++i)
        {
            float factor = A[idxA(i, k)] / pivot;
            // zero out column k in row i
            A[idxA(i, k)] = 0.0f;
            for (int j = k + 1; j < n; ++j)
            {
                A[idxA(i, j)] -= factor * A[idxA(k, j)];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i)
    {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j)
        {
            sum -= A[idxA(i, j)] * c[j];
        }
        c[i] = sum / A[idxA(i, i)];
    }
}

// Evaluate the cubic spline in interval i at x, given y[] and c[] (second derivs).
// x is in [i, i+1], uniform spacing assumed. Returns S_i(x).
__device__ float splineValue(int i, float x, const float *y, const float *c)
{
    // Precompute helpful terms for the interval [i, i+1]
    float a = y[i];
    float b = (y[i + 1] - y[i]) - (c[i + 1] + 2.0f * c[i]) / 6.0f;
    float cCoef = c[i] / 2.0f;
    float dCoef = (c[i + 1] - c[i]) / 6.0f;

    float dx = x - (float)i;
    // S_i(x) = a + b*dx + cCoef*dx^2 + dCoef*dx^3
    return a + b * dx + cCoef * dx * dx + dCoef * dx * dx * dx;
}

// Evaluate the derivative of the spline in interval i at x
__device__ float splineDeriv(int i, float x, const float *y, const float *c)
{
    float b = (y[i + 1] - y[i]) - (c[i + 1] + 2.0f * c[i]) / 6.0f;
    float cCoef = c[i] / 2.0f;
    float dCoef = (c[i + 1] - c[i]) / 6.0f;

    float dx = x - (float)i;
    // d/dx of [a + b*dx + cCoef*dx^2 + dCoef*dx^3]
    return b + 2.0f * cCoef * dx + 3.0f * dCoef * dx * dx;
}

// Find the minimum of the not-a-knot cubic spline on [0, n-1].
__device__ float notAKnotCubicSplineMin(const float *y, int n)
{
    // 1) Solve for second derivatives c_i
    float *c = (float *)alloca(n * sizeof(float));
    solveNotAKnotSystem(y, n, c);

    // 2) Search each interval [i, i+1] for local minima
    float globalMinVal = y[0]; // start with something
    // Evaluate the first endpoint explicitly
    globalMinVal = fminf(globalMinVal, splineValue(0, 0.0f, y, c));

    for (int i = 0; i < n - 1; ++i)
    {
        // Check right endpoint x = i+1
        float valRight = splineValue(i, (float)(i + 1), y, c);
        if (valRight < globalMinVal)
        {
            globalMinVal = valRight;
        }

        // Solve S'_i(x) = 0 for x in [i, i+1].
        // S'_i(x) = b + 2*cCoef*(x-i) + 3*dCoef*(x-i)^2 = 0  -> Quadratic in (x-i)
        float d0 = splineDeriv(i, (float)i, y, c);       // derivative at left endpoint
        float d1 = splineDeriv(i, (float)(i + 1), y, c); // derivative at right endpoint
        // or explicitly expand for a standard quadratic formula in terms of dx

        // Let's fetch the polynomial coefficients for derivative:
        float b = (y[i + 1] - y[i]) - (c[i + 1] + 2.0f * c[i]) / 6.0f; // constant term of derivative
        float cCoef = c[i] / 2.0f;                                     // so derivative gets 2*cCoef*(dx)
        float dCoef = (c[i + 1] - c[i]) / 6.0f;                        // so derivative gets 3*dCoef*(dx^2)

        // Derivative = b + 2*cCoef*dx + 3*dCoef*dx^2.
        // Let alpha = 3*dCoef, beta = 2*cCoef, gamma = b
        float alpha = 3.0f * dCoef;
        float beta = 2.0f * cCoef;
        float gamma = b;

        // Solve alpha*dx^2 + beta*dx + gamma = 0
        // We only care about real roots dx in [0, 1].
        float discriminant = beta * beta - 4.0f * alpha * gamma;
        if (fabsf(alpha) < 1e-12f)
        {
            // If alpha ~ 0, then it's linear: beta*dx + gamma = 0
            if (fabsf(beta) > 1e-12f)
            {
                float dxRoot = -gamma / beta;
                if (dxRoot >= 0.0f && dxRoot <= 1.0f)
                {
                    float xRoot = i + dxRoot;
                    float valRoot = splineValue(i, xRoot, y, c);
                    if (valRoot < globalMinVal)
                    {
                        globalMinVal = valRoot;
                    }
                }
            }
        }
        else if (discriminant >= 0.0f)
        {
            // two real solutions
            float sqrtD = sqrtf(discriminant);
            float dxRoot1 = (-beta + sqrtD) / (2.0f * alpha);
            float dxRoot2 = (-beta - sqrtD) / (2.0f * alpha);

            // Check each root
            if (dxRoot1 >= 0.0f && dxRoot1 <= 1.0f)
            {
                float xRoot = i + dxRoot1;
                float valRoot = splineValue(i, xRoot, y, c);
                if (valRoot < globalMinVal)
                {
                    globalMinVal = valRoot;
                }
            }
            if (dxRoot2 >= 0.0f && dxRoot2 <= 1.0f)
            {
                float xRoot = i + dxRoot2;
                float valRoot = splineValue(i, xRoot, y, c);
                if (valRoot < globalMinVal)
                {
                    globalMinVal = valRoot;
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
