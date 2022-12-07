#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>

using namespace std;

typedef vector<float> vector1d;
typedef vector<vector1d> vector2d;

class Matrix {
 private:

 public:
  // initialize
  Matrix();
  Matrix(vector2d& input);
  Matrix(int x, int y);
  // attributes
  vector2d data;
  Matrix* grad;
  float* array;
  bool requires_grad = false;
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
  void to_array();
  void from_array();
  vector<int> size();
  void print_data();
  string size_str();
  void print_size();
};

#endif