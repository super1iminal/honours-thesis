#include "logger.h"
#include "simulation.h"  // For Samples and Sim_Metadata
#include "utility.h"     // For cudaTry

// Function to log the states (positions of points) to a file
void log_states(Logger *logger, Samples *samples, Sim_Metadata *meta) {
    // Ensure that host samples are up-to-date
    cudaTry(cudaMemcpy(samples->h_x_values, samples->d_x_values, samples->n * samples->stride * sizeof(float), cudaMemcpyDeviceToHost));

    char filename[100];
    snprintf(filename, sizeof(filename), "plot_info/states_%d.txt", meta->t_curr);
    printf("Writing state to file %s\n", filename);

    FILE *state_file = fopen(filename, "w");
    if (state_file == NULL) {
        fprintf(stderr, "ERROR: Unable to open file %s for writing in log_states.\n", filename);
        return;
    }

    int number_of_points = samples->n;
    if (number_of_points > 100000) {
        number_of_points = 100000;
    }

    for (int j = 0; j < number_of_points; j++) {
        fprintf(state_file, "%d", j);
        for (int k = 0; k < samples->stride; k++) {
            fprintf(state_file, ",%f", samples->h_x[j][k]);
        }
        fprintf(state_file, "\n");
    }

    fclose(state_file);
}

// Function to log ellipse parameters to a file or console
void log_ellipse(Logger *logger, const Ellipse *ellipse, const int r_id, const int t_curr) {
    float a = ellipse->a;        // Semi-major axis
    float b = ellipse->b;        // Semi-minor axis
    float theta = ellipse->theta; // Rotation angle in radians

    // Normalize theta to be within [0, 2π)
    if (theta < 0.0f || theta >= 2.0f * M_PI) {
        theta = fmodf(theta, 2.0f * M_PI);
        if (theta < 0.0f) {
            theta += 2.0f * M_PI;
        }
    }

    char filename[100];
    // Construct the filename based on current time step
    snprintf(filename, sizeof(filename), "plot_info/ellipses_%d.txt", t_curr);
    printf("Writing ellipse to file %s\n", filename);

    // Open the file for writing
    FILE *ellipse_file;
    if (r_id == 1) {
        ellipse_file = fopen(filename, "w");
    } else {
        ellipse_file = fopen(filename, "a");
    }

    if (ellipse_file == NULL) {
        log(logger, LOG_ERROR, "ERROR: Unable to open file %s for writing in log_ellipse.\n", filename);
        return;
    }

    // Write ellipse parameters to the file
    fprintf(ellipse_file, "%d,%f,%f,%f\n", r_id, a, b, theta);

    // Close the file
    fclose(ellipse_file);
    // Log ellipse parameters to the designated output (file and/or console)
    log(logger, LOG_ERROR, "Ellipse: a = %f, b = %f, theta = %f radians\n", a, b, theta);
}

// Variadic function to log messages based on log level
void log(Logger *logger, int msg_log_level, const char *format, ...) {
    // Check if the message's log level is greater than or equal to the logger's log level
    if (msg_log_level <= logger->log_level) {
        va_list args;
        va_start(args, format);

        if (logger->output == 0) {
            // Output to both file and console
            va_list args_copy;  // Create a copy of args for the second vfprintf
            va_copy(args_copy, args);  // Copy original va_list

            if (logger->log_file != NULL) {
                vfprintf(logger->log_file, format, args);
                fflush(logger->log_file); // Ensure the message is written to the file
            }
            vfprintf(stdout, format, args_copy);  // Use the copied va_list

            va_end(args_copy);  // Free the copied va_list
        } else if (logger->output == 1) {
            // Output only to file
            if (logger->log_file != NULL) {
                vfprintf(logger->log_file, format, args);
                fflush(logger->log_file); // Ensure the message is written to the file
            }
        } else {
            // Output only to console
            vfprintf(stdout, format, args);
        }

        va_end(args);  // Free the original va_list
    }
}

// Function to initialize the Logger structure
Logger* logger_init(int log_level, int output) {
    // Allocate memory for the Logger structure
    Logger* logger = (Logger*)malloc(sizeof(Logger));
    if (logger == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Logger.\n");
        return NULL;
    }

    logger->log_level = log_level;
    logger->output = output;

    // Get the current time
    time_t now = time(NULL);
    struct tm* time_info = localtime(&now);
    if (time_info == NULL) {
        fprintf(stderr, "ERROR: Unable to get local time for logger.\n");
        free(logger);
        return NULL;
    }

    // Format the time string for the filename (e.g., "log_2024-09-29_10h35m45s.txt")
    char time_str[25];
    if (strftime(time_str, sizeof(time_str), "%Y-%m-%d_%Hh%Mm%Ss", time_info) == 0) {
        fprintf(stderr, "ERROR: Unable to format time string for logger.\n");
        free(logger);
        return NULL;
    }

    // Construct the full log file path (e.g., "logs/log_YYYY-MM-DD_HHhMMmSSs.txt")
    char log_file_name[100];  // Ensure this buffer is large enough for the full path
    snprintf(log_file_name, sizeof(log_file_name), "logs/log_%s.txt", time_str);

    // Open the log file in write mode
    logger->log_file = fopen(log_file_name, "w");
    if (logger->log_file == NULL) {
        fprintf(stderr, "ERROR: Unable to open log file: %s\n", log_file_name);
        free(logger);
        return NULL;
    }

    // Write the start time to the log file
    fprintf(logger->log_file, "Log started at: %s\n", time_str);
    fflush(logger->log_file); // Ensure the message is written to the file

    return logger;
}

// Function to free the Logger structure
void logger_free(Logger* logger) {
    if (logger != NULL) {
        if (logger->log_file != NULL) {
            fclose(logger->log_file);
        }
        free(logger);
    }
    return;
}
