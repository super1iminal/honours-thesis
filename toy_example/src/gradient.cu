#include "gradient.h"
#include "bisection.h"
#include "simulation.h"       // For definitions of Samples and Sim_Metadata
#include "logger.h"           // For Logger definition
#include "utility.h"          // For utility functions like linspace

#define NUM_PARAMETERS 3      // a, b, theta

// Initialize the Gradient structure
Gradient* gradient_init(float perturbation, float initial_learning_rate) {
    // Memory allocation
    Gradient* gradient = (Gradient*)malloc(sizeof(Gradient));
    if (gradient == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Gradient.\n");
        exit(EXIT_FAILURE);
    }

    gradient->ellipse = (Ellipse*)malloc(sizeof(Ellipse));
    gradient->perturbed_ellipse = (Ellipse*)malloc(sizeof(Ellipse));
    gradient->grad = (float*)malloc(NUM_PARAMETERS * sizeof(float)); // [grad_a, grad_b, grad_theta]

    if (gradient->ellipse == NULL || gradient->perturbed_ellipse == NULL || gradient->grad == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Gradient parameters.\n");
        gradient_free(gradient);
        exit(EXIT_FAILURE);
    }

    // Setting initial values
    gradient->ellipse->a = 1.0f;
    gradient->ellipse->b = 1.0f;
    gradient->ellipse->theta = 0.0f;

    *gradient->perturbed_ellipse = *gradient->ellipse; // Copy initial ellipse

    // Initialize grad to zeros
    gradient->grad[0] = 0.0f;
    gradient->grad[1] = 0.0f;
    gradient->grad[2] = 0.0f;

    // Setting hyperparameters
    gradient->perturbation = perturbation;
    gradient->initial_learning_rate = initial_learning_rate;
    gradient->threshold = EPSILON_GRAD;
    gradient->max_iterations = MAX_ITERATIONS_GRAD;

    return gradient;
}

// Free the Gradient structure
void gradient_free(Gradient* gradient) {
    if (gradient != NULL) {
        free(gradient->ellipse);
        free(gradient->perturbed_ellipse);
        free(gradient->grad);
        free(gradient);
    }
    return;
}

// Hyperparameter tuning function
void hyperparameter_tuning(Samples *samples, Sim_Metadata *meta, Logger* logger) {
    int num_steps = 20;
    float perturbations[500];
    linspace(1e-4f, 1e-2f, perturbations, num_steps);

    float learning_rates[500];
    linspace(1e-3f, 1e-1f, learning_rates, num_steps);

    float smallest_area = 1e12f;
    float best_perturbation = perturbations[0];
    float best_learning_rate = learning_rates[0];
    float area;
    Ellipse best_ellipse;
    Ellipse ellipse;

    for (int i = 0; i < num_steps; i++) {
        for (int j = 0; j < num_steps; j++) {
            area = gradient_descent(samples, meta, logger, perturbations[i], learning_rates[j], &ellipse);
            if (area < smallest_area) {
                smallest_area = area;
                best_perturbation = perturbations[i];
                best_learning_rate = learning_rates[j];
                best_ellipse = ellipse;
            }
        }
    }
    log(logger, -1, "Smallest area found: %f, at perturbation %e and learning rate %e\n", smallest_area, best_perturbation, best_learning_rate);
    log(logger, -1, "Ellipse params are: a = %f, b = %f, theta = %f\n", best_ellipse.a, best_ellipse.b, best_ellipse.theta);
    log_ellipse(logger, &best_ellipse, 1, meta->t_curr);
}

// Gradient Descent Function with Decreasing Learning Rate
float gradient_descent(Samples *samples, Sim_Metadata *meta, Logger *logger, float perturbation, float initial_learning_rate, Ellipse* best_ellipse) {
    best_ellipse->a = 1.0f;
    best_ellipse->b = 1.0f;
    best_ellipse->theta = 0.0f;
    Gradient *gradient = gradient_init(perturbation, initial_learning_rate);
    int counter = 1;
    int end = 0;
    float smallest_area = 1e12f;
    float next_area = 1e12f;
    float area = 1e12f;
    float learning_rate;
    int since_best_ellipse = 0;

    log(logger, LOG_BASIC, "Starting gradient descent with hyperparameters: perturbation = %f, initial learning rate = %f, threshold = %f, max iterations = %d\n",
        gradient->perturbation, gradient->initial_learning_rate, gradient->threshold, gradient->max_iterations);

    log(logger, LOG_ADVANCED, "\nStarting iteration 0\n");

    if (!bisection_search(samples, meta, logger, gradient->ellipse)) { end = 1; }
    area = calculate_area(gradient->ellipse);
    if (area < smallest_area) {
        smallest_area = area;
        *best_ellipse = *gradient->ellipse;
    }

    while (!end) {
        // Update learning rate using Inverse Scaling
        learning_rate = gradient->initial_learning_rate / sqrtf((float)counter);
        log(logger, LOG_ADVANCED, "Iteration %d: Learning rate = %f\n", counter, learning_rate);
        log(logger, LOG_ADVANCED, "Area of current ellipse: %f\n", area);
        log(logger, LOG_ADVANCED, "Current ellipse: a = %f, b = %f, theta = %f\n",
            gradient->ellipse->a, gradient->ellipse->b, gradient->ellipse->theta);

        // Compute gradients for 'a', 'b', and 'theta'
        for (int param = 0; param < NUM_PARAMETERS; param++) {
            // Apply perturbation
            *gradient->perturbed_ellipse = *gradient->ellipse; // Copy the struct

            // Perturb the appropriate parameter
            if (param == 0) {
                gradient->perturbed_ellipse->a += gradient->perturbation;
            } else if (param == 1) {
                gradient->perturbed_ellipse->b += gradient->perturbation;
            } else if (param == 2) {
                gradient->perturbed_ellipse->theta += gradient->perturbation;
            }

            // Validate perturbed ellipse
            if (!check_valid_ellipse(gradient->perturbed_ellipse, logger)) { end = 1; break; }
            if (!bisection_search(samples, meta, logger, gradient->perturbed_ellipse)) { end = 1; break; }

            // Calculate area for perturbed ellipse
            float perturbed_area = calculate_area(gradient->perturbed_ellipse);
            // Compute gradient (partial derivative)
            gradient->grad[param] = (perturbed_area - area) / gradient->perturbation;

            log(logger, LOG_ADVANCED, "Gradient for parameter %d: %f\n", param, gradient->grad[param]);
        }

        if (end) {
            break;
        }

        // Check for convergence based on gradient magnitude
        int converged = 1;
        for (int param = 0; param < NUM_PARAMETERS; param++) {
            if (fabsf(gradient->grad[param]) >= gradient->threshold) {
                converged = 0;
                break;
            }
        }
        if (converged) {
            log(logger, LOG_ADVANCED, "Gradient descent converged (gradient small)\n");
            break;
        }

        // Update ellipse parameters
        gradient->ellipse->a -= learning_rate * gradient->grad[0];
        gradient->ellipse->b -= learning_rate * gradient->grad[1];
        gradient->ellipse->theta -= (learning_rate / 50.0f) * gradient->grad[2];

        log(logger, LOG_ADVANCED, "Updated ellipse: a = %f, b = %f, theta = %f\n",
            gradient->ellipse->a, gradient->ellipse->b, gradient->ellipse->theta);

        // Ensure the updated ellipse is valid
        if (!check_valid_ellipse(gradient->ellipse, logger)) { end = 1; break; }
        log(logger, LOG_ADVANCED, "Updated ellipse (after check valid ellipse): a = %f, b = %f, theta = %f\n",
            gradient->ellipse->a, gradient->ellipse->b, gradient->ellipse->theta);

        // Temporary ellipse for bisection
        Ellipse temp = *gradient->ellipse; // Copy the struct
        if (!bisection_search(samples, meta, logger, &temp)) {
            end = 1; break;
        } else {
            *gradient->ellipse = temp;
        }
        log(logger, LOG_ADVANCED, "Updated ellipse (after bisection): a = %f, b = %f, theta = %f\n",
            gradient->ellipse->a, gradient->ellipse->b, gradient->ellipse->theta);

        // Calculate new area
        next_area = calculate_area(gradient->ellipse);
        if (next_area < smallest_area) {
            smallest_area = next_area;
            *best_ellipse = *gradient->ellipse;
            since_best_ellipse = 0;
        } else {
            since_best_ellipse++;
        }

        // Check for convergence based on area difference
        if (fabsf(next_area - area) < gradient->threshold) {
            log(logger, LOG_ADVANCED, "Gradient descent converged in %d iterations (area difference small)\n", counter);
            break;
        }

        // Check for convergence based on number of iterations
        if (counter > gradient->max_iterations) {
            log(logger, LOG_ADVANCED, "Gradient descent reached max iterations\n");
            break;
        }

        // Check for convergence based on iterations since best area
        if (since_best_ellipse > MAX_ITERATIONS_SINCE_BEST) {
            log(logger, LOG_ADVANCED, "Gradient descent converged in %d iterations (no improvement in %d iterations)\n", counter, MAX_ITERATIONS_SINCE_BEST);
            break;
        }

        area = next_area;
        counter++;
    }

    // Log the final area
    log(logger, LOG_BASIC, "Final area of ellipse after gradient descent: %f\n", calculate_area(gradient->ellipse));
    gradient_free(gradient);
    return calculate_area(best_ellipse);
}

// Calculate the area of an ellipse given its parameters
float calculate_area(const Ellipse *ellipse) {
    float a = ellipse->a;      // Semi-major axis
    float b = ellipse->b;      // Semi-minor axis

    // Area of ellipse: π * a * b
    return M_PI * a * b;
}

// Check if the ellipse is valid (a > 0, b > 0)
int check_valid_ellipse(Ellipse *ellipse, Logger *logger) {
    float a = ellipse->a;
    float b = ellipse->b;
    float theta = ellipse->theta;

    if (a <= 0.0f || b <= 0.0f) {
        log(logger, LOG_ERROR, "ERROR: Invalid ellipse dimensions: a = %f, b = %f must be positive.\n", a, b);
        return 0;
    }

    // Normalize theta to be within [0, 2π)
    if (theta < 0.0f || theta >= 2.0f * M_PI) {
        ellipse->theta = fmodf(theta, 2.0f * M_PI);
        if (ellipse->theta < 0.0f) {
            ellipse->theta += 2.0f * M_PI;
        }
    }

    return 1;
}
