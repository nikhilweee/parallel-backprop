#include "matrix.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>

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

void Matrix::init_grad() {
  this->requires_grad = true;
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix* grad = new Matrix(x, y);
  this->grad = grad;
}

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

void Matrix::zero() {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result(x, y);
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      this->data[i][j] = 0.0;
    }
  }
};

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

Matrix Matrix::mul(float other) {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result(x, y);

  if (this->requires_grad) {
    result.init_grad();
  }

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      result.data[i][j] = this->data[i][j] * other;
    }
  }
  return result;
};

void Matrix::mulip(Matrix* other) {
  int x = this->size()[0];
  int y = this->size()[1];

  if (this->requires_grad || other->requires_grad) {
    if (!this->requires_grad) {
      this->init_grad();
    }
    if (!other->requires_grad) {
      other->init_grad();
    }
  }

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      this->data[i][j] *= other->data[i][j];
    }
  }
};

Matrix Matrix::transpose() {
  int x = this->size()[0];
  int y = this->size()[1];
  Matrix result = Matrix(y, x);
  for (int i = 0; i < y; i++) {
    for (int j = 0; j < x; j++) {
      result.data[i][j] = this->data[j][i];
    }
  }
  return result;
};

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

// Core operations

Matrix Matrix::matmul(Matrix& other) {
  if (size()[1] != other.size()[0]) {
    cout << "Sizes of matrices should be compatible. Got " << size_str()
         << " and " << other.size_str() << endl;
    exit(EXIT_FAILURE);
  }
  int x = this->size()[0];
  int y = other.size()[1];
  int z = size()[1];
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

  result.zero();
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      for (int k = 0; k < z; k++) {
        result.data[i][j] += data[i][k] * other.data[k][j];
      }
    }
  }

  return result;
};

Matrix Matrix::add(Matrix other) {
  if (this->size()[1] != other.size()[1]) {
    cout << "Sizes of matrices should be compatible. Got " << size_str()
         << " and " << other.size_str() << endl;
    exit(EXIT_FAILURE);
  }

  int x = this->size()[0];
  int y = this->size()[1];

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

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      result.data[i][j] = this->data[i][j] + other.data[i][j];
    }
  }

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