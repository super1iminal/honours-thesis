#include "regions.h"
#include "simulation.h"       // For definitions of Sim_Metadata, Samples
#include "logger.h"

Iteration_Info* iteration_init(int n) {
    Iteration_Info* info = (Iteration_Info*)malloc(sizeof(Iteration_Info));
    if (info == NULL) {
        fprintf(stderr, "ERROR: Unable to allocate memory for Iteration_Info.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for Ellipse struct
    info->h_ellipse = (Ellipse*)malloc(sizeof(Ellipse));
    info->samples = samples_init_random(n, 4);

    return info;
}

void iteration_free(Iteration_Info* info) {
    if (info != NULL) {
        free(info->h_ellipse);
        samples_free(info->samples);
        free(info);
    }
    return;
}

void find_regions(Sim_Metadata *meta, Logger *logger, int num_regions) {
    if (num_regions <= 0) {
        return;
    }
    // do for every time step
    // while (meta->t_curr < meta->t ) {
    //     // needs to be in the loop to reset samples
    //     Iteration_Info* info = iteration_init(100000);
    //     if (info == NULL) {
    //         fprintf(stderr, "ERROR: Unable to create Iteration_Info.\n");
    //         exit(EXIT_FAILURE);
    //     }
    //     log(logger, LOG_BASIC, "Running region finding algorithm for time step %d\n", meta->t_curr);
    //     int region_it = 0;
    //     // don't have an ellipse yet, we use iteration info starting samples

    //     for (int i = 0; i < meta->t_curr; i++) {
    //         step(info->samples, meta, logger);
    //     }
    //     log_states(logger, info->samples, meta); // log states to file
    //     printf("outputted states\n");

    //     // Run gradient descent to find the first ellipse
    //     log(logger, LOG_BASIC, "Running region finding algorithm for region %d\n", region_it, meta->t_curr);
    //     gradient_descent(info->samples, meta, logger, PERTURBATION, LEARNING_RATE, info->h_ellipse);
    //     region_it++;

    //     // Log the ellipse
    //     log_ellipse(logger, info->h_ellipse, region_it, meta->t_curr);

    //     while (region_it < num_regions) {
    //         log(logger, LOG_BASIC, "Running region finding algorithm for region %d\n", region_it, meta->t_curr);
    //         generate_points(info->samples, info->h_ellipse, meta->t_curr, meta->kp, meta->kd, logger);
    //         // dont log states cuz that'll overwrite basic logged states

    //         // Run gradient descent on new samples
    //         gradient_descent(info->samples, meta, logger, PERTURBATION, LEARNING_RATE, info->h_ellipse);
    //         region_it++;
            
    //         log_ellipse(logger, info->h_ellipse, region_it, meta->t_curr);
    //     }        
    //     meta->t_curr++;
    //     iteration_free(info);
    // }





    // //// for a single time:
    // meta->t_curr = 5;
    // Iteration_Info* info = iteration_init(1000000);
    // if (info == NULL) {
    //     fprintf(stderr, "ERROR: Unable to create Iteration_Info.\n");
    //     exit(EXIT_FAILURE);
    // }
    // log(logger, LOG_BASIC, "Running region finding algorithm for time step %d\n", meta->t_curr);
    // int region_it = 0;
    // // don't have an ellipse yet, we use iteration info starting samples

    // for (int i = 0; i < meta->t_curr; i++) {
    //     step(info->samples, meta, logger);
    // }
    // log_states(logger, info->samples, meta); // log states to file
    // printf("outputted states\n");

    // // Run gradient descent to find the first ellipse
    // log(logger, LOG_BASIC, "Running region finding algorithm for region %d\n", region_it, meta->t_curr);
    // gradient_descent(info->samples, meta, logger, PERTURBATION, LEARNING_RATE, info->h_ellipse);
    // region_it++;

    // // Log the ellipse
    // log_ellipse(logger, info->h_ellipse, region_it, meta->t_curr);

    // while (region_it < num_regions) {
    //     log(logger, LOG_BASIC, "Running region finding algorithm for region %d\n", region_it, meta->t_curr);
    //     generate_points(info->samples, info->h_ellipse, meta->t_curr, meta->kp, meta->kd, logger);
    //     // dont log states cuz that'll overwrite basic logged states

    //     // Run gradient descent on new samples
    //     gradient_descent(info->samples, meta, logger, PERTURBATION, LEARNING_RATE, info->h_ellipse);
    //     region_it++;
        
    //     log_ellipse(logger, info->h_ellipse, region_it, meta->t_curr);
    // }        
    // iteration_free(info);
}
