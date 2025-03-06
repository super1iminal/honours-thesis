#include "utility.cuh"
#include "simulation.cuh"

int main()
{
    // using Clock = std::chrono::high_resolution_clock;
    // double time_samples_in = 0.0, time_print_samples_in = 0.0;
    // double time_samples_out = 0.0, kernel_time = 0.0, time_print_samples_out = 0.0;

    const int rows = 100000000, cols = 4;

    // auto t2 = Clock::now();
    Matrix samples_in(rows, cols, PROCESSOR::DEVICE, true);
    // auto t3 = Clock::now();
    // time_samples_in = std::chrono::duration<double>(t3 - t2).count();

    // auto t4 = Clock::now();
    std::cout << "samples_in: " << std::endl;
    samples_in.print_data();
    // auto t5 = Clock::now();
    // time_print_samples_in = std::chrono::duration<double>(t5 - t4).count();

    // auto t6 = Clock::now();
    Matrix samples_out(rows, cols, PROCESSOR::DEVICE);
    // auto t7 = Clock::now();
    // time_samples_out = std::chrono::duration<double>(t7 - t6).count();

    dim3 block_size(256);
    dim3 grid_size((static_cast<size_t>(samples_in.size()) + block_size.x - 1) / block_size.x);

    // cudaEvent_t cudaStart, cudaStop;
    // cudaEventCreate(&cudaStart);
    // cudaEventCreate(&cudaStop);
    // cudaEventRecord(cudaStart, 0);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    cudaTry(cudaDeviceSynchronize());

    Matrix disturbances(rows, cols, PROCESSOR::DEVICE, true);
    step<<<grid_size, block_size>>>(samples_in, disturbances, samples_out);
    // cudaTry(cudaDeviceSynchronize());
    cudaTry(cudaMemcpy(samples_in.data(), samples_out.data(), samples_in.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaTry(cudaDeviceSynchronize());

    refresh_rand(disturbances);
    step<<<grid_size, block_size>>>(samples_in, disturbances, samples_out);
    // cudaTry(cudaDeviceSynchronize());
    cudaTry(cudaMemcpy(samples_in.data(), samples_out.data(), samples_in.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaTry(cudaDeviceSynchronize());

    refresh_rand(disturbances);
    step<<<grid_size, block_size>>>(samples_in, disturbances, samples_out);
    cudaTry(cudaDeviceSynchronize());
    cudaTry(cudaMemcpy(samples_in.data(), samples_out.data(), samples_in.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaTry(cudaDeviceSynchronize());

    // cudaEventRecord(cudaStop, 0);
    // cudaEventSynchronize(cudaStop);
    // float kernelTimeMs = 0.0f;
    // cudaEventElapsedTime(&kernelTimeMs, cudaStart, cudaStop);
    // kernel_time = kernelTimeMs / 1000.0; // Convert milliseconds to seconds
    // cudaEventDestroy(cudaStart);
    // cudaEventDestroy(cudaStop);

    // auto t8 = Clock::now();
    std::cout << "samples_out: " << std::endl;
    samples_out.print_data();
    // auto t9 = Clock::now();
    // time_print_samples_out = std::chrono::duration<double>(t9 - t8).count();

    // std::cout << "\ntiming summary:\n";
    // std::cout << "1. creating samples_in matrix: "    << time_samples_in       << " seconds\n";
    // std::cout << "2. printing samples_in matrix: "    << time_print_samples_in << " seconds\n";
    // std::cout << "3. creating samples_out matrix: "   << time_samples_out      << " seconds\n";
    // std::cout << "4. kernel execution (steps): "       << kernel_time           << " seconds\n";
    // std::cout << "5. printing samples_out matrix: "   << time_print_samples_out<< " seconds\n";
    return 0;
}