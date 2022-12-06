#include <cassert>
#include <cmath>
#include <iostream>

#include "matrix.h"

void assert_equal(Matrix mat, vector2d test) {
  int x = mat.size()[0];
  int y = mat.size()[1];
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      float diff = (mat.data[i][j] - test[i][j]);
      assert(abs(diff) < 0.0001);
    }
  }
}

Matrix test_init(int x, int y) {
  cout << "test_init" << endl;
  Matrix out = Matrix(x, y);
  int num = 0;
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      out.data[i][j] = ++num;
    }
  }
  out.print_size();
  return out;
}

Matrix test_transpose(Matrix& mat) {
  cout << "test_transpose" << endl;
  Matrix out = mat.transpose();
  out.print_size();
  return out;
}

Matrix test_matmul(Matrix& mat1, Matrix& mat2) {
  cout << "test_matmul" << endl;
  Matrix out = mat1.matmul(mat2);
  out.print_size();
  return out;
}

Matrix test_add(Matrix& mat1, Matrix& mat2) {
  cout << "test_add" << endl;
  Matrix out = mat1.add(mat2);
  out.print_size();
  return out;
}

Matrix test_square(Matrix& mat) {
  cout << "test_square" << endl;
  Matrix out = mat.square();
  out.print_size();
  return out;
}

Matrix test_mul(Matrix& mat, float other) {
  cout << "test_mul" << endl;
  Matrix out = mat.mul(other);
  out.print_size();
  return out;
}

Matrix test_cols(Matrix& mat, int a, int b) {
  cout << "test_cols" << endl;
  Matrix out = mat.cols(a, b);
  out.print_size();
  return out;
}

Matrix test_tanh(Matrix& mat) {
  cout << "test_tanh" << endl;
  Matrix out = mat.tanh();
  out.print_size();
  return out;
}

int main() {
  Matrix mat1, mat2, mat3;
  vector2d test;
  mat1 = test_init(3, 3);
  assert_equal(mat1, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  mat2 = test_init(1, 3);
  assert_equal(mat2, {{1, 2, 3}});
  mat3 = test_add(mat1, mat2);
  assert_equal(mat3, {{2, 4, 6}, {5, 7, 9}, {8, 10, 12}});
  
  mat1 = test_init(2, 3);
  assert_equal(mat1, {{1, 2, 3}, {4, 5, 6}});
  mat2 = test_transpose(mat1);
  assert_equal(mat2, {{1, 4}, {2, 5}, {3, 6}});
  mat3 = test_matmul(mat1, mat2);
  assert_equal(mat3, {{14, 32}, {32, 77}});
  // release mat1, mat2
  mat1 = test_init(2, 2);
  mat2 = test_add(mat3, mat1);
  assert_equal(mat2, {{15, 34}, {35, 81}});
  // release mat1, mat2, mat3
  mat2 = test_mul(mat1, 0.25);
  assert_equal(mat2, {{0.25, 0.5}, {0.75, 1.0}});
  // release mat1
  mat1 = test_square(mat2);
  assert_equal(mat1, {{0.0625, 0.25}, {0.5625, 1.0}});
  // release mat2
  mat2 = test_tanh(mat1);
  assert_equal(mat2, {{0.0624, 0.2449}, {0.5098, 0.7616}});
  // release mat1
  mat1 = test_cols(mat2, 0, 1);
  assert_equal(mat1, {{0.0624}, {0.5098}});
}