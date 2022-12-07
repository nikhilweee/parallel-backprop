#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"

Matrix load_csv(string filename) {
  ifstream file(filename);
  vector2d dataset;
  if (file.is_open()) {
    string line;
    int num_lines = 0;
    while (getline(file, line)) {
      string col;
      stringstream ss(line);
      vector1d row;
      while (getline(ss, col, ',')) {
        float value = stof(col);
        row.push_back(value);
      }
      dataset.push_back(row);
      num_lines++;
      if (num_lines == 100000) {
        break;
      }
    }
    file.close();
  }
  return Matrix(dataset);
}

class MLP {
 public:
  Matrix weight = Matrix(8, 1);
  Matrix bias = Matrix(1, 1);
  MLP() {
    weight.uniform(-0.35, 0.35);
    weight.init_grad();
    bias.uniform(-0.35, 0.35);
    bias.init_grad();
  }
};

int main() {
  Matrix train = load_csv("train.csv");
  Matrix input = train.cols(0, 8);
  cout << input.size_str() << endl;
  Matrix target = train.cols(8, 9);
  float batch_size = input.size()[0];
  float lr = 1e-7;
  MLP net;

  Matrix prod, out, target_neg, diff, diff_sq, diff_sq_div;
  Matrix diff_sq_grad, diff_grad, bias_grad, weight_grad, update;

  for (int epoch = 0; epoch < 101; epoch++) {
    ///////////////
    // FORWARD
    ///////////////

    // prod: (N, 1), input: (N, 8), weight: (8, 1)
    prod = input.matmul(net.weight);
    // out: (N, 1), prod: (N, 1), bias: (1, 1)
    out = prod.add(net.bias);

    target_neg = target.mul(-1);
    // diff: (N, 1), out: (N, 1), target: (N, 1)
    diff = out.add(target_neg);

    diff_sq = diff.square();
    diff_sq_div = diff_sq.mul(1/batch_size);
    float loss = diff_sq_div.sum();

    if (epoch < 10 || epoch % 10 == 0) {
      printf("epoch: %04d loss: %.05f\n", epoch, loss);
    }

    ///////////////
    // BACKWARD
    ///////////////

    // loss.grad starts with one
    // distribute over addition
    // diff_sq_div.grad is also ones
    diff_sq_div.grad->ones();
    // diff_sq_div.grad: (N, 1)

    // route over multiply
    diff_sq_grad = diff_sq.grad->mul(1 / batch_size);
    diff_sq.grad = &diff_sq_grad;
    // diff_sq.grad: (N, 1)

    // d/dx(x^2) = 2x
    diff_grad = diff.mul(2);
    diff.grad = &diff_grad;
    diff.grad->mulip(diff_sq.grad);
    // diff.grad: (N, 1)

    // distribute over sum
    out.grad->mulip(diff.grad);
    // out.grad: (N, 1)

    // distribute over sum
    prod.grad->mulip(out.grad);
    // prod.grad: (N, 1)

    // net.bias was resized
    // so we need to sum
    bias_grad = net.bias.grad->mul(out.grad->sum());
    net.bias.grad = &bias_grad;
    // net.bias.grad: (1, 1)

    // rules of matrix backprop
    weight_grad = input.transpose().matmul(*out.grad);
    net.weight.grad = &weight_grad;
    // net.weight.grad: (8, 1)

    ///////////////
    // UPDATE
    ///////////////

    update = net.weight.grad->mul(-lr);
    net.weight = net.weight.add(update);

    update = net.bias.grad->mul(-lr);
    net.bias = net.bias.add(update);
  }
}