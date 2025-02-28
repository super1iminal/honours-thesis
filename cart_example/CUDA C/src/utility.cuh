#pragma once
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <curand.h> // needs export LD_LIBRARY_PATH=/cs/local/lib/pkg/cudatoolkit-12.0.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH in bashrc
#include <cuda_runtime.h>
#include <vector_types.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <iostream>
#include <iomanip>

#include <stdexcept>
#include <cassert>

#include <time.h>
#include <chrono>
#include <math.h>
#include <vector>

#define BLOCK_SIZE 256

// ==================== UTIL FUNCTIONS ====================
// Macro for CUDA error checking
#define cudaTry(cudaStatus) _cudaTry(cudaStatus, __FILE__, __LINE__)

inline __host__ __device__
bool is_running_on_host() {
#ifndef __CUDA_ARCH__
    return true;
#else
    return false;
#endif
};

// ==================== META ====================
enum class PROCESSOR { HOST, DEVICE };

// Typedefs and function prototypes
typedef unsigned long long uint64;

// Utility function prototypes
extern void rand_vector(void *v, uint n, uint elem_size);
extern void usage(int i, char **opt_names);
extern uint get_uint(int argc, char **argv, const char **optnames, int i, uint v0);
extern double get_double(int argc, char **argv, const char **optnames, int i, double v0);
extern double mean(uint n, double sum);
extern double stdev(uint n, double sum, double sum_sq);
extern void _cudaTry(cudaError_t cuda_return, const char *FileName, int line);

float find_min(float arr[], int n);
float find_max(float arr[], int n);
void linspace(float a, float b, float *r, int x);
int eq_arrays(float* arr1, float* arr2, int n, float threshold);
__host__ bool within_threshold(float *arr1, float *arr2, int n, float threshold);

// ==================== SIM-RELATED UTIL FUNCTIONS ====================
// CUDA kernels
__global__ void set_to_zero(int *arr, int n);
__global__ void fill_kernel(float *data, int length, float value);

// ==================== STRUCTS AND CLASSES ====================
// Matrix class
class Matrix {
public:
    static constexpr float default_fill = 0.f;

    // Constructor: initializes the matrix based on processor type
    __host__ __device__
    Matrix(int rows, int cols, float fill, PROCESSOR proc) 
        : rows_(rows), cols_(cols), proc_(proc), size_(rows*cols), owning_(true)
    {
        if (proc_ == PROCESSOR::HOST) {
            init_host(fill);
        } else if (proc_ == PROCESSOR::DEVICE) {
            init_device(fill);
        } else {
            assert(false && "Invalid processor type");
        }
    }
    // Delegating constructor (sets default fill to default_fill or to random if rand is true)
    __host__ __device__
    Matrix(int rows, int cols, PROCESSOR proc, bool rand = false)
        : rows_(rows), cols_(cols), proc_(proc), size_(rows*cols), owning_(true)
    {
        if (!rand) {
            if (proc_ == PROCESSOR::HOST) {
                init_host(default_fill);
            } else if (proc_ == PROCESSOR::DEVICE) {
                init_device(default_fill);
            } else {
                assert(false && "Invalid processor type");
            }
        } else {
            if (proc_ == PROCESSOR::HOST) {
                init_host_rand();
            } else if (proc_ == PROCESSOR::DEVICE) {
                init_device_rand();
            } else {
                assert(false && "Invalid processor type");
            }
        }
    }

    // takes HOST data and copies it into matrix
    __host__ __device__
    Matrix(int rows, int cols, float** data, PROCESSOR proc)
        : rows_(rows), cols_(cols), proc_(proc), size_(rows*cols), owning_(true)
    {
        if (proc_ == PROCESSOR::HOST) {
            init_host_deterministic(data);
        } else if (proc_ == PROCESSOR::DEVICE) {
            init_device_deterministic(data);
        } else {
            assert(false && "Invalid processor type");
        }
    }


    // Destructor
    __host__ __device__
    ~Matrix() {
        if (!owning_) {
            return;
        }
        if (!is_running_on_host()) {
            return;
        }
        if (proc_ == PROCESSOR::HOST) {
            free_host();
        } else if (proc_ == PROCESSOR::DEVICE) {
            free_device();
        } else {
            assert(false && "Invalid processor type for destructor");
        }
    }

    // Copy constructor, is run on HOST when passed as a parameter to a device function
    __host__ __device__
    Matrix(const Matrix &other):
        rows_(other.rows_), cols_(other.cols_), size_(other.size_), data_(other.data_), proc_(other.proc_), owning_(true)
    {
        if(!is_running_on_host()) {
            owning_ = false;
        } else {
            if (proc_ == PROCESSOR::DEVICE) {
                owning_ = false;
            } else { // only deep copy occurs for host->host with proc_ as host
                data_ = new float[size_];
                memcpy(data_, other.data_, size_ * sizeof(float));
                owning_ = true;
            }
        }
    }
    Matrix& operator=(const Matrix&) = delete;

    // overload [] operator to return a pointer to specified row (can be used like matrix[row][col] now)
    __host__ __device__
    float* operator[](int row) {
        if (is_data_on_host() != is_running_on_host()) {
            assert(false && "Using [] for data on the wrong processor");
        }
        if (row >= rows_) {
            assert(false && "Row access out of range");
        }
        return data_ + (row * cols_);
    }
    __host__ __device__
    const float* operator[](int row) const {
        if (is_data_on_host() != is_running_on_host()) {
            assert(false && "Using [] for data on the wrong processor");
        }
        if (row >= rows_) {
            assert(false && "Row access out of range");
        }
        return data_ + (row * cols_);
    }
    __host__ __device__
    float& operator()(int row, int col) {
        if (is_data_on_host() != is_running_on_host()) {
            assert(false && "Using () for data on the wrong processor");
        }
        if ((row >= rows_) || (col >= cols_)) {
            if ((row >= rows_) && !(col >= cols_)) {
                assert(false && "Row access out of range");
            } else if (!(row >= rows_) && (col >= cols_)) {
                assert(false && "Col access out of range");
            } else {
                assert(false && "Row AND col access out of range");
            }
        }
        return data_[row * cols_ + col];
    }
    __host__ __device__
    const float& operator()(int row, int col) const {
        if (is_data_on_host() != is_running_on_host()) {
            assert(false && "Using () for data on the wrong processor");
        }
        if ((row >= rows_) || (col >= cols_)) {
            if ((row >= rows_) && !(col >= cols_)) {
                assert(false && "Row access out of range");
            } else if (!(row >= rows_) && (col >= cols_)) {
                assert(false && "Col access out of range");
            } else {
                assert(false && "Row AND col access out of range");
            }
        }
        return data_[row * cols_ + col];
    }

    __host__ __device__
    bool is_data_on_host() const {
        return proc_ == PROCESSOR::HOST;
    }

    // pretty-prints data :)
    __host__ void print_data() const;


    // some getters
    // gets num rows
    __host__ __device__ int rows() const { return rows_; }
    // gets num cols
    __host__ __device__ int cols() const { return cols_; }
    // returns rows * cols
    __host__ __device__ int size() const { return size_; }
    __host__ __device__ PROCESSOR proc() const {return proc_; }
    __host__ __device__ float* data() const {return data_; }
    
private:
    int rows_;
    int cols_;
    int size_;
    float *data_;   // Contiguous block of data
    PROCESSOR proc_;
    bool owning_; // to prevent copies from freeing memory

    // Host-specific initialization
    __host__
    void init_host(float fill) {
        // allocate mem on host
        data_ = new float[static_cast<size_t>(size_)];

        // fill
        for (int i = 0; i < rows_ * cols_; i++) {
            data_[i] = fill;
        }
    }
    __host__
    void init_host_deterministic(float** in_data) {
        data_ = new float[static_cast<size_t>(size_)];
    
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                data_[i * cols() + j] = in_data[i][j];
            }
        }
    }

    __host__ 
    void init_host_rand() {
        data_ = new float[static_cast<size_t>(size_)];
        
        // generate random numbers
        float* temp_data;
        cudaTry(cudaMalloc((void**)&temp_data, static_cast<size_t>(size_) * sizeof(float)));      
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateNormal(gen, temp_data, size_, 0.f, 1.f);
        cudaTry(cudaDeviceSynchronize());

        // set random numbers to host data_
        cudaTry(cudaMemcpy(data_, temp_data, static_cast<size_t>(size_) * sizeof(float), cudaMemcpyDeviceToHost));

        curandDestroyGenerator(gen);
        cudaTry(cudaFree(temp_data));
    }

    // Device-specific initialization
    __host__
    void init_device(float fill) {
        // allocate mem on device
        cudaTry(cudaMalloc((void**)&data_, static_cast<size_t>(size_) * sizeof(float)));

        // take advantage of memset if fill is 0
        if (fill == 0.0f) {
            cudaMemset(data_, 0, static_cast<size_t>(size_) * sizeof(float));
        } else {
            dim3 block_size(256);
            dim3 grid_size((static_cast<size_t>(size_) + block_size.x - 1) / block_size.x);
            fill_kernel<<<grid_size, block_size>>>(data_, size_, fill);
            // check for errors below
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                // Clean up device memory before throwing
                cudaFree(data_);
                data_ = nullptr;
                assert(false && "Kernel or sync failed in init_device");
            }
        }

    }

    __host__
    void init_device_deterministic(float** in_data_h) {
        // assuming that we're passing data that's on the host
        cudaTry(cudaMalloc((void**)&data_, static_cast<size_t>(size_) * sizeof(float)));
        float* flattened = new float[static_cast<size_t>(size_)];

        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                flattened[i * cols() + j] = in_data_h[i][j];
            }
        }

        cudaTry(cudaMemcpy(data_, flattened, size_ * sizeof(float), cudaMemcpyHostToDevice));

        delete[] flattened;
    }

    __host__ 
    void init_device_rand() {
        cudaTry(cudaMalloc((void**)&data_, static_cast<size_t>(size_) * sizeof(float)));
        
        // generate random numbers 
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateNormal(gen, data_, size_, 0.f, 1.f);
        cudaTry(cudaDeviceSynchronize());

        curandDestroyGenerator(gen);
    }

    // Host-specific cleanup
    __host__ __device__
    void free_host() {
        if (data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    // Device-specific cleanup
    __host__ __device__
    void free_device() {
        if (data_ != nullptr) {
            cudaTry(cudaFree(data_));
            data_ = nullptr;
        }
    }
};