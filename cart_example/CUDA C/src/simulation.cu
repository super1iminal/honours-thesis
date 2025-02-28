#include "simulation.cuh"
#include "utility.cuh" // For cudaTry and generate_random_numbers

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

__device__ void model(float* sample, float u, int n, float* out) {
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
    float theta_dot_next = theta_dot + (tau * theta_acc);

    out[0] = x_next;
    out[1] = x_dot_next;
    out[2] = theta_next;
    out[3] = theta_dot_next;

    return;
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
    float* sample = (*tensor_samples_in_d)[idx];
    int n = tensor_samples_in_d->cols(); // dimension of controller
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
    float next_sample[4];
    model(sample, u, n, next_sample);
                // very long, but pretty straightforward math
        // send it to samples_out
    for (int i = 0; i < n; i++) {
        (*tensor_samples_out_d)[idx][i] = next_sample[i];
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
