#include "utility.cuh"
void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber)
{
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s in %s line %d\n",
                cudaGetErrorString(cudaStatus), fileName, lineNumber);
        exit(1);
    }
};

float find_min(float arr[], int n)
{
    float min = arr[0]; // Start with the first element
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    return min;
};

float find_max(float arr[], int n)
{
    float max = arr[0]; // Start with the first element
    for (int i = 1; i < n; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }
    return max;
};

void linspace(float a, float b, float *r, int x)
{
    // Calculate the step size
    float step = (b - a) / (x - 1);

    // Generate the numbers
    for (int i = 0; i < x; i++)
    {
        r[i] = a + i * step;
    }
    // Ensures the last element is exactly 'b'
    r[x - 1] = b;
};

int eq_arrays(float *arr1, float *arr2, int n, float threshold)
{
    if (arr1 == NULL || arr2 == NULL || n <= 0 || threshold < 0.0f)
    {
        // Invalid input parameters
        return 0;
    }
    for (int i = 0; i < n; i++)
    {
        if (fabsf(arr1[i] - arr2[i]) > threshold)
        {
            return 0; // Arrays are not equal within the threshold
        }
    }
    return 1; // Arrays are equal within the threshold
};

__global__ void set_to_zero(int *arr, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        arr[idx] = 0;
    }
};

__global__ void set_to_zero_f2(float2 *arr, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        arr[idx].x = 0.0f;
        arr[idx].y = 0.0f;
    }
};

__global__ void fill_kernel(float *data, int length, float value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length)
    {
        data[idx] = value;
    }
};

__host__ bool within_threshold(float *arr1, float *arr2, int n, float threshold)
{
    for (int i = 0; i < n; i++)
    {
        if (fabsf(arr1[i] - arr2[i]) > threshold)
        {
            std::cout << "Mismatch at index " << i << ": " << arr1[i] << " != " << arr2[i] << std::endl;
            return false;
        }
    }
    return true;
};



// ==================== MATRIX ====================
__global__ void update_rand_to_covariance(Matrix matrix)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= matrix.rows())
    {
        return;
    }

    float a_sqrt = sqrt(noise_alpha); // need to sqrt because of covariance multiplication

    float temp[4];
    for (int i = 0; i < matrix.cols(); i++)
    {
        temp[i] = matrix[idx][i];
    }

    for (int i = 0; i < matrix.cols(); i++)
    {
        matrix[idx][i] = 0.f;
        for (int j = 0; j < matrix.cols(); j++)
        {
            matrix[idx][i] += a_sqrt * noise_profile_L[i][j] * temp[j];
        }
    }
}

__host__ void refresh_rand(Matrix matrix)
{
    if (matrix.is_data_on_host())
    {
        assert(false && "This function does not work for host data");
    }

    if (matrix.cols() != 4)
    {
        assert(false && "Matrix must have 4 columns (sorry)");
    }
    // generate random numbers
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, matrix.data(), matrix.size(), 0.f, 1.f);
    cudaTry(cudaDeviceSynchronize());

    curandDestroyGenerator(gen);

    // update the data to be in the covariance space
    dim3 block_size(256);
    dim3 grid_size((static_cast<size_t>(matrix.size()) + block_size.x - 1) / block_size.x);
    update_rand_to_covariance<<<grid_size, block_size>>>(matrix);
    cudaTry(cudaDeviceSynchronize());
    return;
}

__host__ void Matrix::print_data() const
{
    if (!is_running_on_host())
    {
        assert(false && "print() matrix should not be called from device.");
    }

    // pretty numbers
    auto format_number = [](int n) -> std::string
    {
        std::string s = std::to_string(n);
        int insertPosition = s.length() - 3;
        while (insertPosition > 0)
        {
            s.insert(insertPosition, ",");
            insertPosition -= 3;
        }
        return s;
    };

    float *printable_data;
    if (is_data_on_host())
    {
        printable_data = data_;
    }
    else
    {
        printable_data = new float[rows() * cols()];
        cudaTry(cudaMemcpy(printable_data, data_, rows() * cols() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    const int total_rows = rows();
    const int total_cols = cols();

    // header line
    std::cout << "matrix (" << format_number(total_rows) << " x "
              << format_number(total_cols) << ") | device: "
              << (is_data_on_host() ? "host" : "device") << "\n";

    // if > 10 rows, use first 5, marker, last 5; else, all rows
    std::vector<int> row_indices;
    const bool truncate_rows = total_rows > 10;
    if (!truncate_rows)
    {
        for (int i = 0; i < total_rows; ++i)
            row_indices.push_back(i);
    }
    else
    {
        for (int i = 0; i < 5; ++i)
            row_indices.push_back(i);
        row_indices.push_back(-1); // marker
        for (int i = total_rows - 5; i < total_rows; ++i)
            row_indices.push_back(i);
    }

    // if > 10 cols, use first 5, marker, last 5; else, all cols
    std::vector<int> col_indices;
    const bool truncate_cols = total_cols > 10;
    if (!truncate_cols)
    {
        for (int j = 0; j < total_cols; ++j)
            col_indices.push_back(j);
    }
    else
    {
        for (int j = 0; j < 5; ++j)
            col_indices.push_back(j);
        col_indices.push_back(-1); // marker
        for (int j = total_cols - 5; j < total_cols; ++j)
            col_indices.push_back(j);
    }

    const int cellWidth = 10;

    for (int i : row_indices)
    {
        if (i == -1)
        {
            // marker row
            for (size_t k = 0; k < col_indices.size(); ++k)
                std::cout << std::setw(cellWidth) << "...";
            std::cout << "\n";
        }
        else
        {
            for (int j : col_indices)
            {
                if (j == -1)
                {
                    std::cout << std::setw(cellWidth) << "...";
                }
                else
                {
                    std::cout << std::setw(cellWidth)
                              << std::fixed << std::setprecision(4)
                              << printable_data[i * total_cols + j];
                }
            }
            std::cout << "\n";
        }
    }

    if (!is_data_on_host())
    {
        delete[] printable_data;
    }
};

__host__ void Matrix::init_device_deterministic(float **in_data_h)
{
    cudaTry(cudaMalloc((void **)&data_, static_cast<size_t>(size_) * sizeof(float)));
    float *flattened = new float[static_cast<size_t>(size_)];

    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            flattened[i * cols() + j] = in_data_h[i][j];
        }
    }

    cudaTry(cudaMemcpy(data_, flattened, size_ * sizeof(float), cudaMemcpyHostToDevice));

    delete[] flattened;
}

__host__ void Matrix::init_device_rand()
{
    cudaTry(cudaMalloc((void **)&data_, static_cast<size_t>(size_) * sizeof(float)));
    cudaTry(cudaDeviceSynchronize());

    // generate random numbers
    refresh_rand(*this);
}

__host__ void Matrix::init_host_rand()
{
    assert(false && "This function does not work for host data");
}

__host__ void Matrix::init_device(float fill)
{
    // allocate mem on device
    cudaTry(cudaMalloc((void **)&data_, static_cast<size_t>(size_) * sizeof(float)));

    // take advantage of memset if fill is 0
    if (fill == 0.0f)
    {
        cudaMemset(data_, 0, static_cast<size_t>(size_) * sizeof(float));
    }
    else
    {
        dim3 block_size(256);
        dim3 grid_size((static_cast<size_t>(size_) + block_size.x - 1) / block_size.x);
        fill_kernel<<<grid_size, block_size>>>(data_, size_, fill);
        // check for errors below
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            // Clean up device memory before throwing
            cudaFree(data_);
            data_ = nullptr;
            assert(false && "Kernel or sync failed in init_device");
        }
    }
}

__host__ void Matrix::init_host_deterministic(float **in_data)
{
    data_ = new float[static_cast<size_t>(size_)];
    for (int i = 0; i < rows(); ++i)
    {
        for (int j = 0; j < cols(); ++j)
        {
            data_[i * cols() + j] = in_data[i][j];
        }
    }
}

__host__ void Matrix::init_host(float fill)
{
    // allocate mem on host
    data_ = new float[static_cast<size_t>(size_)];

    // fill
    for (int i = 0; i < rows_ * cols_; i++)
    {
        data_[i] = fill;
    }
}
