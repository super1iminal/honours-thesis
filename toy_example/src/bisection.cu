#include "bisection.h"
#include "simulation.h"       // For definitions of Samples and Sim_Metadata
#include "logger.h"           // For full definition of Logger
#include "utility.h"          // For cudaTry and utility functions

// Kernel to calculate distances in the ellipse's coordinate system
__global__ void calculate_elliptical_distances(float2* d_x, float* d_distances, int n, Ellipse ellipse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = d_x[idx].x;
    float y = d_x[idx].y;
    float x_rot, y_rot;
    float cos_theta = cosf(ellipse.theta);
    float sin_theta = sinf(ellipse.theta);
    x_rot = x * cos_theta + y * sin_theta;
    y_rot = -x * sin_theta + y * cos_theta;

    // Compute the normalized distance using the ellipse equation
    d_distances[idx] = (x_rot * x_rot) / (ellipse.a * ellipse.a) + (y_rot * y_rot) / (ellipse.b * ellipse.b);
}

// Updated bisection_init function
Bisection* bisection_init(Samples* samples, Logger *logger, int num_ellipses, Ellipse initial_ellipse) {
    // MEMORY ALLOCATION
    Bisection *bisection = (Bisection*)malloc(sizeof(Bisection));

    if (num_ellipses < 2) {
        log(logger, LOG_ERROR, "ERROR: Number of ellipses must be at least 2\n");
        free(bisection);
        exit(1);
    }
    bisection->num_ellipses = num_ellipses;
    bisection->h_ellipses = (Ellipse*)malloc(bisection->num_ellipses * sizeof(Ellipse));
    cudaTry(cudaMalloc((void**)&bisection->d_ellipses, bisection->num_ellipses * sizeof(Ellipse)));

    bisection->h_num_within_ellipse = (int*)malloc(bisection->num_ellipses * sizeof(int));
    cudaTry(cudaMalloc((void**)&bisection->d_num_within_ellipse, bisection->num_ellipses * sizeof(int)));

    bisection->h_distances = (float*)malloc(samples->n * sizeof(float));
    cudaTry(cudaMalloc((void**)&bisection->d_distances, samples->n * sizeof(float)));

    // Extract initial ellipse parameters
    float initial_a = initial_ellipse.a;
    float initial_b = initial_ellipse.b;
    float initial_theta = initial_ellipse.theta;

    // Compute distances in the ellipse's coordinate system
    calculate_elliptical_distances<<<((samples->n + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        samples->d_x, bisection->d_distances, samples->n, initial_ellipse
    );
    cudaTry(cudaMemcpy(bisection->h_distances, bisection->d_distances, samples->n * sizeof(float), cudaMemcpyDeviceToHost));

    // Find max and min normalized distances
    float max_norm = find_max(bisection->h_distances, samples->n);
    log(logger, LOG_DEBUG, "max_norm: %f\n", max_norm);
    float min_norm = find_min(bisection->h_distances, samples->n);
    log(logger, LOG_DEBUG, "min_norm: %f\n", min_norm);
    if (min_norm <= EPSILON_BISECT) {
        log(logger, LOG_ERROR, "WARNING: Smallest normalized distance is too small, falling back to sqrt of near-smallest float\n");
        min_norm = EPSILON_BISECT;
    }

    // Compute scaling factors based on normalized distances
    // Use the square root to maintain proportional scaling
    float scale_high = sqrtf(max_norm);
    float scale_low = sqrtf(min_norm);

    // Apply uniform scaling to maintain the ellipse shape
    bisection->high.a = initial_a * scale_high;
    bisection->high.b = initial_b * scale_high;
    bisection->high.theta = initial_theta;

    bisection->low.a = initial_a * scale_low;
    bisection->low.b = initial_b * scale_low;
    bisection->low.theta = initial_theta;

    // Initialize counts
    set_to_zero<<<((bisection->num_ellipses + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        bisection->d_num_within_ellipse, bisection->num_ellipses
    );
    cudaDeviceSynchronize();

    // Generate ellipses and compute points within them
    generate_ellipses(bisection, logger);
    within_ellipses(bisection, samples);

    // Copy back the counts
    cudaTry(cudaMemcpy(bisection->h_num_within_ellipse, bisection->d_num_within_ellipse, bisection->num_ellipses * sizeof(int), cudaMemcpyDeviceToHost));

    // Check if the initial ellipses contain at least 90% of the samples
    float low_p = ((float)bisection->h_num_within_ellipse[0]) / ((float)samples->n);
    float high_p = ((float)bisection->h_num_within_ellipse[bisection->num_ellipses - 1]) / ((float)samples->n);
    if ((low_p >= TARGET_PERCENTAGE) || (high_p <= TARGET_PERCENTAGE)) {
        log(logger, LOG_BASIC, "ERROR: Initial ellipse does not contain 90 percent of samples\n");
        bisection_free(bisection);
        return NULL;
    }

    return bisection;
}

void bisection_free(Bisection* bisection) {
    if (bisection != NULL) {
        free(bisection->h_ellipses);
        free(bisection->h_num_within_ellipse);
        free(bisection->h_distances);
        // No need to free bisection->high and bisection->low as they are not dynamically allocated
        cudaTry(cudaFree(bisection->d_ellipses));
        cudaTry(cudaFree(bisection->d_num_within_ellipse));
        cudaTry(cudaFree(bisection->d_distances));
        free(bisection);
    }
    return;
}

void generate_ellipses(Bisection* bisection, Logger *logger) {
    float t;
    float a_low = bisection->low.a;
    float b_low = bisection->low.b;
    float theta_low = bisection->low.theta;

    float a_high = bisection->high.a;
    float b_high = bisection->high.b;
    float theta_high = bisection->high.theta;

    for (int i = 0; i < bisection->num_ellipses; i++) {
        t = ((float)i) / ((float)(bisection->num_ellipses - 1.0f)); // t ranges from 0 to 1

        // Interpolate parameters
        float a_t = (1.0f - t) * a_low + t * a_high;
        float b_t = (1.0f - t) * b_low + t * b_high;
        float theta_t = (1.0f - t) * theta_low + t * theta_high;

        // Store the parameters
        bisection->h_ellipses[i].a = a_t;
        bisection->h_ellipses[i].b = b_t;
        bisection->h_ellipses[i].theta = theta_t;

        log(logger, LOG_DEBUG, "Ellipse %d: a = %f, b = %f, theta = %f\n", i, a_t, b_t, theta_t);
    }

    // Copy the interpolated ellipses to device memory
    cudaTry(cudaMemcpy(bisection->d_ellipses, bisection->h_ellipses,
                       bisection->num_ellipses * sizeof(Ellipse), cudaMemcpyHostToDevice));
}

int eq_ellipses(Ellipse e1, Ellipse e2, float threshold) {
    return (fabsf(e1.a - e2.a) <= threshold) &&
           (fabsf(e1.b - e2.b) <= threshold) &&
           (fabsf(e1.theta - e2.theta) <= threshold);
}

int bisection_search(Samples* samples, Sim_Metadata* meta, Logger* logger, Ellipse* ellipse) {
    Bisection *bisection = bisection_init(samples, logger, NUM_ELLIPSES, *ellipse);
    if (bisection == NULL) {
        log(logger, LOG_ERROR, "ERROR: Unable to initialize bisection search\n");
        return 0;
    }
    bool converged = false;
    int counter = 0;
    float high_p = 0.0f;
    float low_p = 0.0f;
    float p = 0.0f;
    int low_index = 0;
    int high_index = 0;
    int best_index = 0;
    float relative_threshold;

    if (samples->n > 1 / (EPSILON_BISECT * EPSILON_BISECT)) {
        log(logger, LOG_ERROR, "WARNING: Number of samples is too large for relative threshold, falling back to near-smallest float absolute threshold\n");
        relative_threshold = EPSILON_BISECT * EPSILON_BISECT;
    } else {
        relative_threshold = 0.01f / samples->n;
    }

    while ((!converged) && (counter < MAX_ITERATIONS_BISECT)) {
        counter++;
        p = 0.0f;
        high_p = 0.0f;
        low_p = 0.0f;
        low_index = 0;
        high_index = 0;
        set_to_zero<<<((bisection->num_ellipses + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(bisection->d_num_within_ellipse, bisection->num_ellipses);
        log(logger, LOG_DEBUG, "\nIteration %d\n", counter);
        generate_ellipses(bisection, logger);
        // Run within_ellipses to find number of samples within each ellipse
        within_ellipses(bisection, samples);
        // Copy counts back to host
        cudaTry(cudaMemcpy(bisection->h_num_within_ellipse, bisection->d_num_within_ellipse, bisection->num_ellipses * sizeof(int), cudaMemcpyDeviceToHost));

        // Find two ellipses with values over and under 90% of samples
        low_p = ((float)bisection->h_num_within_ellipse[0]) / ((float)samples->n);
        high_p = ((float)bisection->h_num_within_ellipse[bisection->num_ellipses - 1]) / ((float)samples->n);
        low_index = 0;
        high_index = bisection->num_ellipses - 1;
        for (int i = 0; i < bisection->num_ellipses; i++) {
            p = ((float)bisection->h_num_within_ellipse[i]) / ((float)samples->n);
            log(logger, LOG_DEBUG, "Ellipse %d has %0.3e percent of samples\n", i, p);
            if (p < TARGET_PERCENTAGE) {
                if ((fabsf(p - TARGET_PERCENTAGE) <= fabsf(low_p - TARGET_PERCENTAGE))) {
                    low_p = p;
                    low_index = i;
                }
            } else if (p > TARGET_PERCENTAGE) {
                if ((fabsf(TARGET_PERCENTAGE - p) <= fabsf(TARGET_PERCENTAGE - high_p))) {
                    high_p = p;
                    high_index = i;
                }
            } else {
                low_p = p;
                high_p = p;
                high_index = i;
                low_index = i;
                break;
            }
        }

        // Update low and high ellipses
        bisection->low = bisection->h_ellipses[low_index];
        bisection->high = bisection->h_ellipses[high_index];

        if ((fabsf(low_p - TARGET_PERCENTAGE) < relative_threshold) || (fabsf(high_p - TARGET_PERCENTAGE) < relative_threshold)) {
            log(logger, LOG_DEBUG, "\nBisection search converged within %d iterations\n", counter);
            if (fabsf(low_p - TARGET_PERCENTAGE) <= fabsf(high_p - TARGET_PERCENTAGE)) {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples (percentage within %0.3e of %0.3e)\n", low_p, relative_threshold, TARGET_PERCENTAGE);
                best_index = low_index;
            } else {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples (percentage within %0.3e of %0.3e)\n", high_p, relative_threshold, TARGET_PERCENTAGE);
                best_index = high_index;
            }
            converged = true;
        } else if (fabsf(low_p - high_p) < relative_threshold) {
            log(logger, LOG_DEBUG, "Converged within %d iterations\n", counter);
            if (fabsf(low_p - TARGET_PERCENTAGE) <= fabsf(high_p - TARGET_PERCENTAGE)) {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples\n", low_p);
                best_index = low_index;
            } else {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples\n", high_p);
                best_index = high_index;
            }
            converged = true;
        } else if (eq_ellipses(bisection->low, bisection->high, relative_threshold)) {
            log(logger, LOG_DEBUG, "Converged within %d iterations\n", counter);
            if (fabsf(low_p - TARGET_PERCENTAGE) <= fabsf(high_p - TARGET_PERCENTAGE)) {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples\n", low_p);
                best_index = low_index;
            } else {
                log(logger, LOG_DEBUG, "Best ellipse has %0.3e percent of samples\n", high_p);
                best_index = high_index;
            }
            converged = true;
        }
    }

    // Copy best ellipse to output
    *ellipse = bisection->h_ellipses[best_index];
    bisection_free(bisection);
    return 1;
}
