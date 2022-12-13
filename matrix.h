#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

using namespace std;

typedef vector<float> vector1d;
typedef vector<vector1d> vector2d;

enum dev {cpu, cuda};

class Matrix {
 private:

 public:
  // initialize
  Matrix();
  Matrix(vector2d& input);
  Matrix(int x, int y);
  ~Matrix();
  // attributes
  vector2d data;
  Matrix* grad;
  float* array;
  float* cuda_array;
  bool requires_grad = false;
  dev device = cpu;
  // reshape
  Matrix cols(int a, int b);
  // initialize
  void init_grad();
  void ones();
  void uniform(float a, float b);
  // self operations
  Matrix tanh();
  Matrix square();
  Matrix transpose();
  // scalar operations
  Matrix mul(float other);
  // aggregate operations
  float sum();
  // matrix operations
  void mulip(Matrix* other);
  Matrix add(Matrix other);
  Matrix matmul(Matrix& other);
  // size and data
  void to_arrays();
  void from_cuda_array();
  vector<int> size();
  string size_str();
  void print_size();
  void print_data();
  void print_array(int N);
  void print_cuda(int N);
};

#endif