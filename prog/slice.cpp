#include "kmeans.h"

#include <json/json.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdio.h>
#include <string>

#include<chrono>

using namespace std;

// Convert a Json::Value to an Eigen::MatrixXd
Eigen::MatrixXd json_to_matrix(Json::Value root) {
    int rows = root.size();
    int cols = 3; // Assuming each vector has 3 elements (x, y, z)

    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        matrix(i, 0) = root[to_string(i)]["x"].asDouble();
        matrix(i, 1) = root[to_string(i)]["y"].asDouble();
        matrix(i, 2) = root[to_string(i)]["z"].asDouble();
    }
    return matrix;
}


int main(){

    ifstream file("data.json");
    Json::Value root;
    file >> root;

    Eigen::MatrixXd atomic_matrix = json_to_matrix(root);
    // std::cout << "Atomic matrix: " << std::endl;
    // std::cout << atomic_matrix << std::endl;

    // MatrixXd centers_init = KMeans_plus_plus(atomic_matrix, 8).run().first;
    // std::cout << "Initial centers: " << std::endl;
    // std::cout << centers_init << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    KMeans kmeans;
    kmeans.fit(atomic_matrix);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Time taken for KMeans (microseconds):  " << execution_time << std::endl;

    Eigen::VectorXi labels = kmeans.labels_;
    std::cout << "Labels: " << std::endl;
    std::cout << labels.transpose() << std::endl;

    double inertia = kmeans.inertia_;
    std::cout << "Inertia: " << inertia << std::endl;

    return 0;
}