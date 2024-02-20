#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <json/json.h>
#include "kmeans.h"
#include "extmath.h"
#include <numeric>

using namespace std;

// Define a vector type that contains vectors of 3 doubles
typedef vector<vector< double > > Vector3D;


// Convert a Json::Value to a Vector3D
Vector3D json_to_vector(Json::Value root) {
    Vector3D output_vector;
    for (int i = 0; i < root.size(); i++) {
        vector<double> vect;
        vect.push_back(root[to_string(i)]["x"].asDouble());
        vect.push_back(root[to_string(i)]["y"].asDouble());
        vect.push_back(root[to_string(i)]["z"].asDouble());
        output_vector.push_back(vect);
    }
    return output_vector;
}

int main(){

    ifstream file("data.json");
    Json::Value root;
    file >> root;

    Vector3D atomic_vector = json_to_vector(root);
    // for (int i = 0; i < atomic_vector.size(); i++) {
    //     for (int j = 0; j < atomic_vector[i].size(); j++) {
    //         cout << atomic_vector[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // KMeans_plus_plus kmeans_pp(atomic_vector);
    // std::pair<Vector3D, std::vector<int>> result = kmeans_pp.run();
    // Vector3D centers = KMeans_plus_plus(atomic_vector).run().first;
    // Vector3D centers = result.first;
    // std::cout << "Centers: " << std::endl;
    // for (int i = 0; i < centers.size(); i++) {
    //     for (int j = 0; j < centers[i].size(); j++) {
    //         std::cout << centers[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::vector<int> indices = result.second;
    // std::cout << "Indices: " << std::endl;
    // for (int i = 0; i < indices.size(); i++) {
    //     std::cout << indices[i] << " ";
    // }

    KMeans kmeans;
    Vector3D centers = kmeans.fit(atomic_vector);


}