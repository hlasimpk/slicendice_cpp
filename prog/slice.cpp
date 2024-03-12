#include "agglomerative.h"
#include "birch.h"
#include "kmeans.h"

#include <json/json.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>


// Convert a Json::Value to an Eigen::MatrixXd
Eigen::MatrixXd json_to_matrix(Json::Value root) {
    int rows = root.size();
    int cols = 3; // Assuming each vector has 3 elements (x, y, z)

    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        matrix(i, 0) = root[std::to_string(i)]["x"].asDouble();
        matrix(i, 1) = root[std::to_string(i)]["y"].asDouble();
        matrix(i, 2) = root[std::to_string(i)]["z"].asDouble();
    }
    return matrix;
}

// Convert an Eigen::VectorXd to a Json::Value
Json::Value vector_to_json(Eigen::VectorXi vector) {
    Json::Value root(Json::arrayValue);
    for (int i = 0; i < vector.size(); i++) {
        Json::Value item;
        item[std::to_string(i)] = vector(i);
        root.append(item);
    }
    return root;
}

int main(int argc, char** argv){
    std::string input_json;
    std::string clustering_method = "birch";
    int nclusters = 3;
    std::string output_json = "output.json"; 
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input_json") {
            if (i + 1 < argc) { 
                input_json = argv[++i]; 
            } else {
                std::cerr << "--input_json option requires one argument." << std::endl;
                return 1;
            }  
        } else if (arg == "--nclusters") {
            if (i + 1 < argc) { 
                nclusters = std::stoi(argv[++i]); 
            }
        } else if (arg == "--clustering_method") {
            if (i + 1 < argc) { 
                clustering_method = argv[++i]; 
            }
        } else if (arg == "--output_json") {
            if (i + 1 < argc) { 
                output_json = argv[++i]; 
            } 
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --input_json path/to/input.json --nclusters (Default: 3) --clustering_method <agglomerative, birch, kmeans> (Default: birch) --output_json path/to/output.json (Default: output.json)" << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help to get usage information." << std::endl;
            return 1;
        }
    }

    std::ifstream file(input_json);
    if (!file) {
        std::cerr << "Unable to open JSON file at path: " << input_json << std::endl;
    return 1;
}
    Json::Value root;
    file >> root;

    Eigen::MatrixXd atomic_matrix = json_to_matrix(root);

    Eigen::VectorXi labels;
    
    if (clustering_method == "agglomerative") {
        Agglomerative agglomerative(nclusters);
        agglomerative.fit(atomic_matrix);
        labels = agglomerative.labels_;
    } else if (clustering_method == "birch") {
        Birch birch(0.5, 50, nclusters);
        birch.fit(atomic_matrix);
        labels = birch.labels_;
    } else if (clustering_method == "kmeans") {
        KMeans kmeans(nclusters);
        kmeans.fit(atomic_matrix);
        labels = kmeans.labels_;
    } else {
        std::cout << "Clustering method: " << clustering_method << " not yet implemented." << std::endl;
    }
    

    Json::Value output;
    output = vector_to_json(labels);
    
    std::ofstream outputFile(output_json);
    if (outputFile.is_open()) {
        outputFile << output;
        outputFile.close();
    } else {
        std::cerr << "Unable to open output file." << std::endl;
        return 1;
    }

    return 0;
}