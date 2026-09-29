#include "main.h"
#include <time.h>

/*
USING TOY CAR:
./toycar <number of samples> <time step> <kp> <kd>
./toycar <number of samples> <time step>
./toycar (default values: 10, 10, 1, 0.1)

./run [-n 1000000] [-t 5] [-kp -0.25] [-kd -1.5] [-log 1]
-n: number of samples
-t: time horizon
-kp: proportional gain
-kd: derivative gain
-log: logging level (0, 1, 2 or 3)
default values:
n = 1000000, t = 5, kp = -0.25, kd = -1.5, log = 1
The nominal state is always (0, 0).

The distance to a point determines the radius of the circle

*/

int main(int argc, char* argv[]) {
    //////////////// INITIAL SETUP ////////////////
    int n = 1000000; // number of samples
    int t = 5;      // time horizon
    float kp = -0.25f; // proportional gain
    float kd = -1.5f;  // derivative gain
    int log_level = LOG_BASIC;
    int output = 0;

    // Parsing command line arguments
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: ./run [-n 1000000] [-t 5] [-kp -0.25] [-kd -1.5] [-log 2]\n");
            printf("n: number of samples\n");
            printf("t: time horizon\n");
            printf("kp: proportional gain\n");
            printf("kd: derivative gain\n");
            printf("log: logging level (0, 1, 2 or 3)\n");
            printf("default values: n = 1000000, t = 5, kp = -0.25, kd = -1.5, log = 1\n");
            return 0;
        }

        // Every other option takes exactly one value
        if (i + 1 >= argc) {
            printf("Missing value for %s.\n", argv[i]);
            printf("Use -h for help.\n");
            return EXIT_FAILURE;
        }

        if (strcmp(argv[i], "-n") == 0) {
            n = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "-t") == 0) {
            t = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "-kp") == 0) {
            kp = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-kd") == 0) {
            kd = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-log") == 0) {
            log_level = atoi(argv[i+1]);
        } else {
            printf("Incorrect use of command line arguments.\n");
            printf("Use -h for help.\n");
            return EXIT_FAILURE;
        }
        i += 2;
    }

    if (n <= 0 || t <= 0) {
        printf("n and t must be positive.\n");
        return EXIT_FAILURE;
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
    };

    // Run the simulation steps
    // step(samples, disturbances, &meta, logger); // Step 1
    // step(samples, disturbances, &meta, logger); // Step 2
    // Add more steps if needed
    // step(samples, disturbances, meta, logger); // Step 3

    // Log the states
    // log_states(logger, samples, &meta);

    log(logger, LOG_ERROR, "Using the following values for n, t, kp and kd: %d, %d, %f, %f\nLogging is at level %d\n",
        n, t, kp, kd, log_level);

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
