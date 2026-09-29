#include "main.h"
#include <time.h>

/*
USING TOY CAR:
./toycar <number of samples> <time step> <kp> <kd>
./toycar <number of samples> <time step>
./toycar (default values: 10, 10, 1, 0.1)

./run [-n 5000] [-t 15] [-kp -0.25] [-kd -1.5] [-nom 0.0 0.0] [-log 1]
-n: number of samples
-t: time horizon
-kp: proportional gain
-kd: derivative gain
-nom: nominal state
-log: logging level (0, 1, 2 or 3)
default values: 
n = 5000, t = 15, kp = -0.25, kd = -1.5, nom = (0.0, 0.0), log = 1

The distance to a point determines the radius of the circle

*/

int main(int argc, char* argv[]) {
    //////////////// INITIAL SETUP ////////////////
    int n = 1000000; // number of samples
    int t = 15;      // time horizon
    float kp = -0.25f; // proportional gain
    float kd = -1.5f;  // derivative gain
    float2 nom = {0.0f, 0.0f}; // nominal state
    int log_level = LOG_BASIC;
    int output = 0;

    // Parsing command line arguments
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: ./run [-n 5000] [-t 10] [-kp -0.25] [-kd -1.5] [-nom 0.0 0.0] [-log 2]\n");
            printf("n: number of samples\n");
            printf("t: time horizon\n");
            printf("kp: proportional gain\n");
            printf("kd: derivative gain\n");
            printf("nom: nominal state\n");
            printf("log: logging level (0, 1, 2 or 3)\n");
            printf("default values: n = 5000, t = 15, kp = -0.25, kd = -1.5, nom = (0.0, 0.0), log = 1\n");
            return 0;
        } else if (strcmp(argv[i], "-n") == 0) {
            n = atoi(argv[i+1]);
            i += 2;
        } else if (strcmp(argv[i], "-t") == 0) {
            t = atoi(argv[i+1]);
            i += 2;
        } else if (strcmp(argv[i], "-kp") == 0) {
            kp = atof(argv[i+1]);
            i += 2;
        } else if (strcmp(argv[i], "-kd") == 0) {
            kd = atof(argv[i+1]);
            i += 2;
        } else if (strcmp(argv[i], "-nom") == 0) {
            nom.x = atof(argv[i+1]);
            nom.y = atof(argv[i+2]);
            i += 3;
        } else if (strcmp(argv[i], "-log") == 0) {
            log_level = atoi(argv[i+1]);
            i += 2;
        } else {
            i++;
        }
    }

    //////////////// STATIC DECLARATIONS ////////////////
    Logger *logger = logger_init(log_level, output);
    if (logger == NULL) {
        fprintf(stderr, "ERROR: Unable to initialize logger.\n");
        return EXIT_FAILURE;
    }

    Sim_Metadata meta = {
        .n = n,
        .t = t,
        .t_curr = 0,
        .kp = kp,
        .kd = kd,
        .nom = nom,
    };

    // Run the simulation steps
    // step(samples, disturbances, &meta, logger); // Step 1
    // step(samples, disturbances, &meta, logger); // Step 2
    // Add more steps if needed
    // step(samples, disturbances, meta, logger); // Step 3

    // Log the states
    // log_states(logger, samples, &meta);

    log(logger, LOG_ERROR, "Using the following values for n, t, kp, kd and nom: %d, %d, %f, %f, (%f, %f)\nLogging is at level %d\n",
        n, t, kp, kd, nom.x, nom.y, log_level);

    // Run the region finding algorithm
    clock_t start_time = clock();

    find_regions(&meta, logger, 5);

    clock_t end_time = clock();

    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    log(logger, LOG_ERROR, "Time spent in find_regions: %f seconds\n", time_spent);

    // Optionally, run hyperparameter tuning
    // hyperparameter_tuning(samples, meta, logger);

    // Free allocated memory
    logger_free(logger);

    return 0;
}

/*
Questions/Comments
- What plotting library should I use?
*/
