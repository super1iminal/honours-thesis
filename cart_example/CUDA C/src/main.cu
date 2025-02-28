#include "main.cuh"
#include <time.h>
#include <iostream>

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

int main() {
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << std::endl;
    return 0;
}