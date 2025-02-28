#include "simulation.h"
#include "logger.h"
#include "utility.h" // For cudaTry and generate_random_numbers


// ==================== INIT FUNCTIONS ====================
Matrix* matrix_init(int rows, int cols, float fill, Processor proc)
{
    if (proc == HOST) {
        return matrix_init_host(rows, cols, fill);
    } else if (proc == DEVICE) {
        return matrix_init_device(rows, cols, fill);
    } else {
        exit(-1);
    }
}


/* Returns an in-function allocated Matrix on host memory */
Matrix* matrix_init_host(int rows, int cols, float fill)
{
    // Check if fill is NaN; if so, set it to 0.0f
    if (isnan(fill)) {
        fill = 0.0f;
    }

    // Allocate memory for the Matrix struct
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Failed to allocate memory for Matrix struct.\n");
        return NULL;
    }

    // Set rows and cols and processor
    mat->rows = rows;
    mat->cols = cols;
    mat->proc = HOST;

    // Allocate a contiguous block for matrix values
    mat->data_values = (float *)malloc(rows * cols * sizeof(float));
    if (!mat->data_values) {
        fprintf(stderr, "Failed to allocate memory for matrix data_values.\n");
        free(mat);  // Remember to free the struct if other allocations fail
        return NULL;
    }

    // Initialize all elements with the fill value
    for (int i = 0; i < rows * cols; i++) {
        mat->data_values[i] = fill;
    }

    // Allocate array of row pointers
    mat->data = (float **)malloc(rows * sizeof(float *));
    if (!mat->data) {
        fprintf(stderr, "Failed to allocate memory for matrix data.\n");
        free(mat->data_values);
        free(mat);
        return NULL;
    }

    // Set each row pointer to the corresponding location in data_values
    for (int i = 0; i < rows; i++) {
        mat->data[i] = &mat->data_values[i * cols];
    }

    return mat;
}


/* -------------------- Device-only version -------------------- */
/*
  A small kernel to fill the device array with a single float value.
*/
__global__ void fill_kernel(float *data, int length, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        data[idx] = value;
    }
}

Matrix* matrix_init_device(int rows, int cols, float fill)
{
    // Check if fill is NaN; if so, set it to 0.0f
    if (isnan(fill)) {
        fill = 0.0f;
    }

    // Allocate the Matrix structure on the host
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Failed to allocate Matrix struct on host.\n");
        return NULL;
    }

    // Rows and cols and processor identifier remain on the host
    mat->rows = rows;
    mat->cols = cols;
    mat->proc = DEVICE;

    // 1) Allocate contiguous data array on the device
    size_t totalSize = (size_t)rows * (size_t)cols;
    cudaTry(cudaMalloc((void**)&(mat->data_values), totalSize * sizeof(float)));

    // 2) Fill that device array with 'fill' using a kernel
    int blockSize = 256;
    int gridSize  = (int)((totalSize + blockSize - 1) / blockSize);
    fillKernel<<<gridSize, blockSize>>>(mat->data_values, fill, (int)totalSize);
    cudaTry(cudaDeviceSynchronize());

    // 3) Allocate array of row pointers on the device
    cudaTry(cudaMalloc((void**)&(mat->data), rows * sizeof(float*)));

    // 4) Create a temporary host array of row pointers to copy to device
    float **host_data_ptrs = (float **)malloc(rows * sizeof(float *));
    if (!host_data_ptrs) {
        fprintf(stderr, "Failed to allocate host_data_ptrs.\n");
        // Clean up so we don't leak device memory
        cudaTry(cudaFree(mat->data_values));
        cudaTry(cudaFree(mat->data));
        free(mat);
        return NULL;
    }

    // Each row pointer points into the device data_values array
    for (int i = 0; i < rows; i++) {
        host_data_ptrs[i] = mat->data_values + (size_t)i * (size_t)cols;
    }

    // 5) Copy the row-pointer array to the device
    cudaTry(cudaMemcpy(mat->data, host_data_ptrs, rows * sizeof(float*),
                       cudaMemcpyHostToDevice));

    // Free the host array of pointers (no longer needed)
    free(host_data_ptrs);

    return mat;  // The Matrix struct is host-allocated; device pointers are inside it
}


// ==================== FREE FUNCTIONS ====================
void matrix_free(Matrix* matrix) {
    if (matrix->proc == HOST) {
        matrix_host_free(matrix);
        return;
    } else if (matrix->proc == DEVICE) {
        matrix_device_free(matrix);
    }
}

void matrix_host_free(Matrix *matrix)
{
    if (!matrix)
        return;

    // Free host-allocated contiguous data
    if (matrix->data_values) {
        free(matrix->data_values);
        matrix->data_values = NULL;
    }

    // Free host-allocated array of row pointers
    if (matrix->data) {
        free(matrix->data);
        matrix->data = NULL;
    }

    // Finally, free the host struct
    free(matrix);
}

void matrix_device_free(Matrix *matrix)
{
    if (!matrix)
        return;

    // Free device-allocated contiguous data
    if (matrix->data_values) {
        cudaFree(matrix->data_values);
        matrix->data_values = NULL;
    }

    // Free device-allocated array of row pointers
    if (matrix->data) {
        cudaFree(matrix->data);
        matrix->data = NULL;
    }

    // The Matrix struct itself is on the host, so free with normal free()
    free(matrix);
}


// ==================== NN FUNCTIONS ====================
// __device__ void ReLU(Matrix input, Matrix output, Matrix weights, Matrix bias) {
//     int idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride;
// }

// ==================== SIM FUNCTIONS ====================

__device__ float controller(float* sample, int n) {
    float out = 0.f;
    for (int i = 0; i < n; i++) {
        out += sample[i] * policy_weights[i];
    }
    return out;
}

__device__ float* model(float* sample, float u, int n) {
    float force = u;
    // clamp 
    force = force > 0.f ? force > max_force ? max_force : force : force < -max_force ? -max_force : max_force;
    float x = sample[0];
    float x_dot = sample[1];
    float theta = sample[2];
    float theta_dot = sample[3];

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float temp = cart_mass + pole_mass * sin_theta * sin_theta;
    float theta_acc = ((-force * cos_theta) - (pole_mass * length * theta_dot * theta_dot * cos_theta * sin_theta) + (total_mass * gravity * sin_theta)) / (length * temp);
    float x_acc = (force + (pole_mass * sin_theta * ((length * theta_dot * theta_dot) - (gravity * cos_theta))))/temp;

    float x_next = x + (tau * x_dot);
    float x_dot_next = x_dot + (tau * x_acc);
    float theta_next = theta + (tau * theta_dot);
    float theta_dot_next = theta_dot + (tau * theta_acc)

    return {x_next, x_dot_next, theta_next, theta_dot_next};
}
 

/*
Parameters:
    tensor_samples_in: a data tensor of the input samples
    tensor_samples_out: a data tensor of the output samples
Explanation:
    - Given a data tensor of all samples (i.e. a matrix where each row corresponds to the values of a samples), 
      return a data tensor of all samples at the next time step
    - To avoid data bloat by passing the same weights for every thread, I've put them in constant memory (see __constant__ declarations in .h)
        this will save space and time at the cost of less dynamic capability. sorry
*/
__global__ void step(Matrix* tensor_samples_in_d, Matrix* tensor_samples_out_d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float* sample = tensor_samples_in_d->data[idx];
    int n = tensor_samples_in_d->cols; // dimension of controller
    // for each sample (this thread will have the sample idx):
        // apply vector-vector multiplication between the 1x4 sample vector and some static weight vector ([ 0.6234,  1.8060, 34.6404, 11.8123])
        // apply model to sample****:
            // constraints (input: time)
                // actually time invariant, pretty easy
            // planner (input: samples, constraints output)
                // currently just returns [0,0,0,0]
            // controller (input: sample, planner output)
                // currently ignores r, just outputs NN eval of samples
    float u = controller(sample, n);
            // model (input: sample, controller output)
    float* next_sample = model(sample, u, n);
                // very long, but pretty straightforward math
        // send it to samples_out
    for (int i = 0; i < n; i++) {
        tensor_samples_out_d->data[idx][i] = next_sample[i];
    }
    
    
}

/*
Parameters:
    tensor_samples_in: a data tensor of the input samples
    tensor_samples_out: a data tensor of the output samples
Explanation:
    - Given a data tensor of all samples (i.e. a matrix where each row corresponds to the values of a samples), 
        return a data tensor of all samples at the next time step
    - To avoid data bloat by passing the same weights for every thread, I've put them in constant memory (see __constant__ declarations in .h)
        this will save space and time at the cost of less dynamic capability. sorry
*/
__global__ void lyapunov(Matrix* samples_in, Matrix* lyapunov) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // for each sample:
        // multiply 1x4 sample vector with 4x32 weight matrix
        // relu
        // multiply 1x32 vector with 32x32 weight matrix
        // relu
        // multiply 32x32 matrix with 32x1 weight matrix
    // need to do all of the above in each individual thread so that millions/billions of samples can be run concurrently
    // should use __constant__ memory for weights for broadcast accesses (very fast accesses)
}
