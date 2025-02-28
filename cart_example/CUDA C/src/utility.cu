#include "utility.cuh"

__host__ void Matrix::print_data() const {
    if (!is_running_on_host()) {
        assert(false && "print() matrix should not be called from device.");
    }

    // pretty numbers
    auto format_number = [](int n) -> std::string {
        std::string s = std::to_string(n);
        int insertPosition = s.length() - 3;
        while (insertPosition > 0) {
            s.insert(insertPosition, ",");
            insertPosition -= 3;
        }
        return s;
    };

    float* printable_data;
    if (is_data_on_host()) {
        printable_data = data_;
    } else {
        printable_data = new float[rows() * cols()];
        cudaTry(cudaMemcpy(printable_data, data_, rows() * cols() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    const int total_rows = rows();
    const int total_cols = cols();

    // header
    std::cout << "Matrix (" << format_number(total_rows) << " x " << format_number(total_cols) << ") | Device: "
              << (is_data_on_host() ? "HOST" : "DEVICE") << "\n";

    // if more than 6 rows, we use: first 3, then a marker row (-1), then last 3.
    std::vector<int> row_indices;
    const bool truncate_rows = total_rows > 6;
    if (!truncate_rows) {
        for (int i = 0; i < total_rows; ++i)
            row_indices.push_back(i);
    } else {
        row_indices.push_back(0);
        row_indices.push_back(1);
        row_indices.push_back(2);
        row_indices.push_back(-1); // marker for ellipsis
        row_indices.push_back(total_rows - 3);
        row_indices.push_back(total_rows - 2);
        row_indices.push_back(total_rows - 1);
    }

    // if more than 6 columns, we use: first 3, then a marker (-1), then last 3.
    std::vector<int> col_indices;
    const bool truncate_cols = total_cols > 6;
    if (!truncate_cols) {
        for (int j = 0; j < total_cols; ++j)
            col_indices.push_back(j);
    } else {
        col_indices.push_back(0);
        col_indices.push_back(1);
        col_indices.push_back(2);
        col_indices.push_back(-1); // marker for ellipsis
        col_indices.push_back(total_cols - 3);
        col_indices.push_back(total_cols - 2);
        col_indices.push_back(total_cols - 1);
    }

    const int cellWidth = 10;

    for (int i : row_indices) {
        if (i == -1) {
            // marker
            for (size_t k = 0; k < col_indices.size(); ++k) {
                std::cout << std::setw(cellWidth) << "..." ;
            }
            std::cout << "\n";
        } else {
            for (int j : col_indices) {
                if (j == -1) {
                    std::cout << std::setw(cellWidth) << "..." ;
                } else {
                    std::cout << std::setw(cellWidth)
                              << std::fixed << std::setprecision(4)
                              << printable_data[i * total_cols + j];
                }
            }
            std::cout << "\n";
        }
    }

    if (!is_data_on_host()) {
        delete[] printable_data;
    }
};



void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s in %s line %d\n",
                cudaGetErrorString(cudaStatus), fileName, lineNumber);
        exit(1);
    }
};

float find_min(float arr[], int n) {
    float min = arr[0];  // Start with the first element
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
};

float find_max(float arr[], int n) {
    float max = arr[0];  // Start with the first element
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
};

void linspace(float a, float b, float *r, int x) {
    // Calculate the step size
    float step = (b - a) / (x - 1);

    // Generate the numbers
    for (int i = 0; i < x; i++) {
        r[i] = a + i * step;
    }
    // Ensures the last element is exactly 'b'
    r[x - 1] = b;
};

int eq_arrays(float* arr1, float* arr2, int n, float threshold) {
    if (arr1 == NULL || arr2 == NULL || n <= 0 || threshold < 0.0f) {
        // Invalid input parameters
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (fabsf(arr1[i] - arr2[i]) > threshold) {
            return 0; // Arrays are not equal within the threshold
        }
    }
    return 1; // Arrays are equal within the threshold
};

__global__ void set_to_zero(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = 0;
    }
};

__global__ void set_to_zero_f2(float2 *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx].x = 0.0f;
        arr[idx].y = 0.0f;
    }
};

__global__ void fill_kernel(float *data, int length, float value) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        data[idx] = value;
    }
};

__host__ bool within_threshold(float *arr1, float *arr2, int n, float threshold) {
    for (int i = 0; i < n; i++) {
        if (fabsf(arr1[i] - arr2[i]) > threshold) {
            std::cout << "Mismatch at index " << i << ": " << arr1[i] << " != " << arr2[i] << std::endl;
            return false;
        }
    }
    return true;
};