#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "matrix.h"

#define idx(i, j, N) ((i) * (N)) + (j)

void print_array(float* array, int size) {
  printf("array [%d]: ", size);
  for (int i = 0; i < size; i++) {
    printf("%.05f ", array[i]);
  }
  cout << endl;
}

////////////////////////////////////////////////////////////////////////////////

Matrix::Matrix(){};
Matrix::Matrix(vector2d& input) : data(input){};
Matrix::Matrix(int x, int y) {
  for (int i = 0; i < x; i++) {
    vector1d row;
    for (int j = 0; j < y; j++) {
      row.push_back(1.0);
    }
    this->data.push_back(row);
  }
}

////////////////////////////////////////////////////////////////////////////////

void Matrix::to_array() {
  int x = this->size()[0];
  int y = this->size()[1];
  int N = x * y;

  float* array = new float[N];
  for (int i = 0; (i < x); i++) {
    for (int j = 0; (j < y); j++) {
      array[idx(i, j, y)] = this->data[i][j];
    }
  }
  this->array = array;
}

void Matrix::from_array() {
  int x = this->size()[0];
  int y = this->size()[1];

  // TODO: check if size of array matches
  for (int i = 0; (i < x); i++) {
    for (int j = 0; (j < y); j++) {
      this->data[i][j] = this->array[idx(i, j, y)];
    }
  }
}

void Matrix::init_grad() {
  this->requires_grad = true;
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix* grad = new Matrix(x, y);
  this->grad = grad;
}

////////////////////////////////////////////////////////////////////////////////

void Matrix::uniform(float a, float b) {
  int x = this->size()[0];
  int y = this->size()[1];
  // random_device rand_dev;
  // default_random_engine generator(rand_dev());
  default_random_engine generator;
  uniform_real_distribution<float> uniform(a, b);
  for (int i = 0; i < x; i++) {
    vector<float> row;
    for (int j = 0; j < y; j++) {
      this->data[i][j] = uniform(generator);
    }
  }
}

void Matrix::ones() {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result(x, y);
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      this->data[i][j] = 1.0;
    }
  }
};

Matrix Matrix::tanh() {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result(x, y);
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      result.data[i][j] = std::tanh(this->data[i][j]);
    }
  }
  return result;
};

Matrix Matrix::square() {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result(x, y);

  if (this->requires_grad) {
    result.init_grad();
  }

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      result.data[i][j] = this->data[i][j] * this->data[i][j];
    }
  }
  return result;
};

////////////////////////////////////////////////////////////////////////////////

__global__ void mul_kernel(float* result, float* self, float other, int N) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    result[i] = self[i] * other;
  }
}

Matrix Matrix::mul(float other) {
  int x = this->size()[0];
  int y = this->size()[1];
  int N = x * y;

  Matrix result(x, y);

  if (this->requires_grad) {
    result.init_grad();
  }

  int size = N * sizeof(float);

  float* self_cuda;
  cudaMalloc((void**)&self_cuda, size);
  this->to_array();
  cudaMemcpy(self_cuda, this->array, size, cudaMemcpyHostToDevice);

  float* result_cuda;
  cudaMalloc((void**)&result_cuda, size);
  result.to_array();
  cudaMemcpy(result_cuda, result.array, size, cudaMemcpyHostToDevice);

  int num_blocks = ceil((float)N / 512);
  mul_kernel<<<num_blocks, 512>>>(result_cuda, self_cuda, other, N);

  cudaMemcpy(result.array, result_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(self_cuda);
  cudaFree(result_cuda);

  result.from_array();

  return result;
};

////////////////////////////////////////////////////////////////////////////////

__global__ void mulip_kernel(float* self, float* other, int N) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    self[i] = self[i] * other[i];
  }
}

void Matrix::mulip(Matrix* other) {
  int x = this->size()[0];
  int y = this->size()[1];
  int N = x * y;

  if (this->requires_grad || other->requires_grad) {
    if (!this->requires_grad) {
      this->init_grad();
    }
    if (!other->requires_grad) {
      other->init_grad();
    }
  }

  int size = N * sizeof(float);

  float* self_cuda;
  cudaMalloc((void**)&self_cuda, size);
  this->to_array();
  cudaMemcpy(self_cuda, this->array, size, cudaMemcpyHostToDevice);

  float* other_cuda;
  cudaMalloc((void**)&other_cuda, size);
  other->to_array();
  cudaMemcpy(other_cuda, other->array, size, cudaMemcpyHostToDevice);

  mulip_kernel<<<1, 512>>>(self_cuda, other_cuda, N);

  cudaMemcpy(this->array, self_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(self_cuda);
  cudaFree(other_cuda);

  this->from_array();
};

////////////////////////////////////////////////////////////////////////////////

__global__ void transpose_kernel(float* result, float* self, int x, int y, int N) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx(row, col, x) < N && idx(col, row, y) < N) {
    result[idx(row, col, x)] = self[idx(col, row, y)];
  }
}

Matrix Matrix::transpose() {
  int x = this->size()[0];
  int y = this->size()[1];
  int N = x * y;

  Matrix result = Matrix(y, x);

  int size = N * sizeof(float);

  float* self_cuda;
  cudaMalloc((void**)&self_cuda, size);
  this->to_array();
  cudaMemcpy(self_cuda, this->array, size, cudaMemcpyHostToDevice);

  float* result_cuda;
  cudaMalloc((void**)&result_cuda, size);
  result.to_array();
  cudaMemcpy(result_cuda, result.array, size, cudaMemcpyHostToDevice);

  dim3 num_threads(32, 32);
  dim3 num_blocks(1, 1);
  num_blocks.x = ceil((float)x / 512);
  num_blocks.y = ceil((float)y / 512);
  transpose_kernel<<<num_blocks, num_threads>>>(result_cuda, self_cuda, x, y, N);

  cudaMemcpy(result.array, result_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(self_cuda);
  cudaFree(result_cuda);

  result.from_array();

  return result;
};

////////////////////////////////////////////////////////////////////////////////

float Matrix::sum() {
  int x = this->size()[0];
  int y = this->size()[1];
  float sum = 0;
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      sum += data[i][j];
    }
  }
  return sum;
};

Matrix Matrix::cols(int a, int b) {
  int x = this->size()[0];
  Matrix result = Matrix(x, b - a);
  for (int i = 0; i < x; i++) {
    for (int j = a; j < b; j++) {
      result.data[i][j - a] = this->data[i][j];
    }
  }
  return result;
};

////////////////////////////////////////////////////////////////////////////////
// Core operations
////////////////////////////////////////////////////////////////////////////////

__global__ void matmul_kernel(float* result, float* self, float* other, int x,
                              int y, int z) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = threadIdx.y + blockDim.y * blockIdx.y;
  float sum = 0.0;
  if (row < x && col < y) {
    for (int i = 0; i < z; i++) {
      sum += self[idx(row, i, z)] * other[idx(i, col, y)];
    }
    result[idx(row, col, y)] = sum;
  }
}

Matrix Matrix::matmul(Matrix& other) {
  if (size()[1] != other.size()[0]) {
    cout << "Sizes of matrices should be compatible. Got " << size_str()
         << " and " << other.size_str() << endl;
    exit(EXIT_FAILURE);
  }
  int x = this->size()[0];
  int y = other.size()[1];
  int z = this->size()[1];
  int N = x * y;

  Matrix result = Matrix(x, y);

  if (this->requires_grad || other.requires_grad) {
    if (!this->requires_grad) {
      this->init_grad();
    }
    if (!other.requires_grad) {
      other.init_grad();
    }
    result.init_grad();
  }

  int size_self = x * z * sizeof(float);
  int size_other = z * y * sizeof(float);
  int size_result = N * sizeof(float);

  float* self_cuda;
  cudaMalloc((void**)&self_cuda, size_self);
  this->to_array();
  cudaMemcpy(self_cuda, this->array, size_self, cudaMemcpyHostToDevice);

  float* other_cuda;
  cudaMalloc((void**)&other_cuda, size_other);
  other.to_array();
  cudaMemcpy(other_cuda, other.array, size_other, cudaMemcpyHostToDevice);

  float* result_cuda;
  cudaMalloc((void**)&result_cuda, size_result);
  result.to_array();
  cudaMemcpy(result_cuda, result.array, size_result, cudaMemcpyHostToDevice);

  dim3 num_threads(32, 32);
  dim3 num_blocks(1, 1);
  num_blocks.x = ceil((float)x / 512);
  num_blocks.y = ceil((float)y / 512);
  matmul_kernel<<<num_blocks, num_threads>>>(result_cuda, self_cuda, other_cuda,
                                             x, y, z);

  cudaMemcpy(result.array, result_cuda, size_result, cudaMemcpyDeviceToHost);

  cudaFree(self_cuda);
  cudaFree(other_cuda);
  cudaFree(result_cuda);

  result.from_array();
  return result;
};

////////////////////////////////////////////////////////////////////////////////

__global__ void add_kernel(float* result, float* self, float* other, int N) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    result[i] = self[i] + other[i];
  }
}

Matrix Matrix::add(Matrix other) {
  if (this->size()[1] != other.size()[1]) {
    cout << "Sizes of matrices should be compatible. Got " << size_str()
         << " and " << other.size_str() << endl;
    exit(EXIT_FAILURE);
  }

  int x = this->size()[0];
  int y = this->size()[1];
  int N = x * y;

  if (other.size()[0] == 1) {
    Matrix interm = Matrix(x, y);
    for (int i = 0; i < x; i++) {
      for (int j = 0; j < y; j++) {
        interm.data[i][j] = other.data[0][j];
      }
    }
    other = interm;
  }

  if (this->size()[0] != other.size()[0]) {
    cout << "Sizes of matrices should be compatible. Got " << size_str()
         << " and " << other.size_str() << endl;
    exit(EXIT_FAILURE);
  }

  Matrix result = Matrix(x, y);

  if (this->requires_grad || other.requires_grad) {
    if (!this->requires_grad) {
      this->init_grad();
    }
    if (!other.requires_grad) {
      other.init_grad();
    }
    result.init_grad();
  }

  int size = N * sizeof(float);

  float* self_cuda;
  cudaMalloc((void**)&self_cuda, size);
  this->to_array();
  cudaMemcpy(self_cuda, this->array, size, cudaMemcpyHostToDevice);

  float* other_cuda;
  cudaMalloc((void**)&other_cuda, size);
  other.to_array();
  cudaMemcpy(other_cuda, other.array, size, cudaMemcpyHostToDevice);

  float* result_cuda;
  cudaMalloc((void**)&result_cuda, size);
  result.to_array();
  cudaMemcpy(result_cuda, result.array, size, cudaMemcpyHostToDevice);

  int num_blocks = ceil((float)N / 512);
  add_kernel<<<num_blocks, 512>>>(result_cuda, self_cuda, other_cuda, N);

  cudaMemcpy(result.array, result_cuda, size, cudaMemcpyDeviceToHost);

  cudaFree(self_cuda);
  cudaFree(other_cuda);
  cudaFree(result_cuda);

  result.from_array();

  return result;
};

// Helper functions

void Matrix::print_data() {
  for (vector1d row : data) {
    for (float col : row) {
      printf("%10.4f ", col);
    }
    cout << endl;
  }
};

vector<int> Matrix::size() {
  vector<int> size;
  size.push_back(data.size());
  size.push_back(data[0].size());
  return size;
};

string Matrix::size_str() {
  vector<int> size_vec = size();
  string size_str;
  size_str = "(";
  for (size_t i = 0; i < size_vec.size(); i++) {
    size_str += to_string(size_vec[i]);
    if (i < (size_vec.size() - 1)) {
      size_str += ", ";
    }
  }
  size_str += ")";
  return size_str;
};

void Matrix::print_size() {
  // useless comment
  cout << size_str() << endl;
};