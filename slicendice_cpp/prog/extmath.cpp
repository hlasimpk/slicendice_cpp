#include "extmath.h"
#include <cmath>
#include <vector>

typedef std::vector<std::vector< double > > Vector3D;

Vector3D transpose(Vector3D data){
    size_t rows = data.size();
    size_t cols = data[0].size();

    Vector3D transposed(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
}

Vector3D dot_product(const Vector3D& X, const Vector3D& Y) {
    size_t X_rows = X.size();
    size_t X_cols = X[0].size();
    size_t Y_cols = Y[0].size();

    Vector3D result(X_rows, std::vector<double>(Y_cols, 0.0));
    for (size_t i = 0; i < X_rows; ++i) {
        for (size_t j = 0; j < Y_cols; ++j) {
            for (size_t k = 0; k < X_cols; ++k) {
                result[i][j] += X[i][k] * Y[k][j];
            }
        }
    }

    return result;
}

std::vector<double> row_norms(Vector3D data, bool squared) {
    std::vector<double> norms;
    for (int i = 0; i < data.size(); i++) {
        double norm = 0.0;
        for (int j = 0; j < data[i].size(); j++) {
            norm += data[i][j] * data[i][j];
        }
        norms.push_back(norm);
    }

    if (!squared) {
        for (int i = 0; i < norms.size(); i++) {
            norms[i] = std::sqrt(norms[i]);
        }
    }

    return norms;
}


Vector3D euclidean_distances(Vector3D X, Vector3D Y, std::vector<double> x_squared_norms, std::vector<double> y_squared_norms, bool squared) {

    // Generate the squared norms of the rows of X and Y
    std::vector<double> XX_row_norms = row_norms(X, true);

    // Convert the squared norms of the rows of X to a column vector
    Vector3D XX(XX_row_norms.size(), std::vector<double>(1));
    for (size_t i=0; i < XX_row_norms.size(); i++) {
        XX[i][0] = XX_row_norms[i];
    }

    // Convert the squared norms of the rows of Y to a row vector
    Vector3D YY(1, row_norms(Y, true));

    // Calculate the distances matrix with a dot product
    Vector3D Y_transposed = transpose(Y);

    // Calculate the distances matrix with a dot product
    Vector3D distances_matrix = dot_product(X, Y_transposed);


    // Multiply the distances matrix by -2
    for (auto& row : distances_matrix) {
        for (auto& element : row) {
            element *= -2;
        }
    }

    // Add the squared norms of the rows of X and Y to the distances matrix
    for (size_t i = 0; i < distances_matrix.size(); i++) {
        for (size_t j = 0; j < distances_matrix[i].size(); j++) {
            distances_matrix[i][j] += XX[i][0];
            distances_matrix[i][j] += YY[0][j];
        }
    }

    // Set all negative elements of the distances matrix to 0
    for (auto& row : distances_matrix) {
        for (auto& element : row) {
            element = std::max(element, 0.0);
        }
    }

    // If X is Y, set the diagonal elements of the distances matrix to 0
    if (X == Y) {
        for (size_t i = 0; i < distances_matrix.size(); i++) {
            distances_matrix[i][i] = 0;
        }
    }

    // If squared is false, return the square root of the distances matrix
    if (!squared) {
        for (auto& row : distances_matrix) {
            for (auto& element : row) {
                element = std::sqrt(element);
            }
        }
    }

    return distances_matrix;
}

std::vector<double> calculate_mean(const Vector3D& data) {
    std::vector<double> mean(data[0].size(), 0.0);

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            mean[j] += row[j];
        }
    }

    for (auto& m : mean) {
        m /= data.size();
    }

    return mean;
}

std::vector<double> calculate_variance(const Vector3D& data) {
    std::vector<double> mean = calculate_mean(data);
    std::vector<double> variance(data[0].size(), 0.0);

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            variance[j] += std::pow(row[j] - mean[j], 2);
        }
    }

    for (auto& v : variance) {
        v /= data.size();
    }

    return variance;
}