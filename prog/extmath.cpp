#include "extmath.h"

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <set>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


MatrixXd euclidean_distances(const MatrixXd& X, const MatrixXd& Y, const VectorXd& X_norm_squared, const VectorXd& Y_norm_squared, bool squared) {
    VectorXd XX;
    if (X_norm_squared.size() != 0) {
        XX = X_norm_squared;
    } else {
        XX = X.rowwise().squaredNorm();
    }

    VectorXd YY;
    if (&X == &Y) {
        YY = XX;
    } else if (Y_norm_squared.size() != 0) {
        YY = Y_norm_squared;
    } else {
        YY = Y.rowwise().squaredNorm();
    }

    MatrixXd distances_matrix = -2. * X * Y.transpose();
    distances_matrix.colwise() += XX;
    distances_matrix.rowwise() += YY.transpose();

    distances_matrix = distances_matrix.cwiseMax(0);


    if (&X == &Y) {
        distances_matrix.diagonal().setZero();
    }

    if (!squared) {
        distances_matrix = distances_matrix.array().sqrt();
    }

    return distances_matrix;
}

VectorXd calculate_mean(const MatrixXd& data) {
    VectorXd mean = data.colwise().mean();
    return mean;
}

VectorXd calculate_variance(const MatrixXd& data) {
    VectorXd mean = calculate_mean(data);
    VectorXd variance = ((data.rowwise() - mean.transpose()).array().square().colwise().sum()) / data.rows();
    return variance;
}

MatrixXd partition(const MatrixXd& v, int kth) {
    MatrixXd result = v;  // Copy v to result

    for (int i = 0; i < v.cols(); ++i) {
        // Get the i-th column
        VectorXd column = result.col(i);

        // Partition the column
        if(column.size() > kth) {
            std::nth_element(column.data(), column.data() + kth, column.data() + column.size());
        }

        // Store the partitioned column in the result
        result.col(i) = column;
    }

    return result;
}

VectorXi unique(const VectorXi& v) {
    std::set<int> unique_set; // Use a set to get unique elements
    for (int i = 0; i < v.size(); ++i) {
        unique_set.insert(v(i));
    }
    VectorXi unique_values(unique_set.size());
    std::copy(unique_set.begin(), unique_set.end(), unique_values.data());
    return unique_values;
}

std::pair<Eigen::VectorXi, Eigen::VectorXi> triu_indices(int n) {
    int num_indices = n * (n - 1) / 2;
    Eigen::VectorXi i_indices(num_indices), j_indices(num_indices);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            i_indices(k) = i;
            j_indices(k) = j;
            k++;
        }
    }
    return {i_indices, j_indices};
}