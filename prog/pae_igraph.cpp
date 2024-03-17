#include "agglomerative.h"
#include "pae_igraph.h"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <igraph/igraph.h>
#include <json/json.h>
#include <set>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

MatrixXd pae_json_to_matrix(Json::Value data) {
    if (data.isObject() && data.isMember("pae")) {
        // ColabFold 1.3 produces a JSON file different from AlphaFold database.
        Json::Value pae_data = data["pae"];
        Eigen::MatrixXd matrix(pae_data.size(), pae_data[0].size());
        for (int i = 0; i < pae_data.size(); ++i) {
            for (int j = 0; j < pae_data[i].size(); ++j) {
                matrix(i, j) = pae_data[i][j].asDouble();
            }
        }
        return matrix;
    } else if (data.isArray()) {
        data = data[0];
        if (data.isMember("residue1") && data.isMember("distance")) {
            // Support original JSON file format.
            Json::Value residue1_data = data["residue1"];
            Json::Value distance_data = data["distance"];
            int size = 0;
            for (int i = 0; i < residue1_data.size(); ++i) {
                size = std::max(size, residue1_data[i].asInt());
            }
            Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);
            for (int i = 0; i < distance_data.size(); ++i) {
                matrix(i / size, i % size) = distance_data[i].asDouble();
            }
            return matrix;
        } else if (data.isMember("predicted_aligned_error")) {
            // Support new AlphaFold database JSON file format.
            Json::Value pae_data = data["predicted_aligned_error"];
            Eigen::MatrixXd matrix(pae_data.size(), pae_data[0].size());
            for (int i = 0; i < pae_data.size(); ++i) {
                for (int j = 0; j < pae_data[i].size(); ++j) {
                    double value = pae_data[i][j].asDouble();
                    matrix(i, j) = (value == 0) ? 0.2 : value;
                }
            }
            return matrix;
        } else {
            throw std::runtime_error("PAE data not detected in json file");
        }
    } else {
        throw std::runtime_error("Invalid data format in json file");
    }
}

PAE::PAE(int input_nclusters,
         std::string input_pae_file,
         bool input_merge_clusters,
         double input_pae_power, 
         double input_pae_cutoff, 
         double input_graph_resolution, 
         double input_distance_power) :
    nclusters(input_nclusters),
    pae_file(input_pae_file),
    merge_clusters(input_merge_clusters),
    pae_power(input_pae_power),
    pae_cutoff(input_pae_cutoff),
    graph_resolution(input_graph_resolution),
    distance_power(input_distance_power) {
    parse_pae_file(pae_file);
    }

void PAE::parse_pae_file(std::string pae_file) {
    std::ifstream file(pae_file);
    if (!file.good()) {
        throw std::runtime_error("Error: file " + pae_file + " not found.");
    }

    std::size_t pos = pae_file.find_last_of(".");
    std::string pae_file_extension = pae_file.substr(pos);

    if (pae_file_extension == ".json" || pae_file_extension == ".jsn") {
        Json::Value root;
        file >> root;
        pae_matrix = pae_json_to_matrix(root);

    } else {
        throw std::runtime_error("Error: file " + pae_file + " not supported.");
    }

}

void PAE::fit(MatrixXd& X) {
    MatrixXd weights = MatrixXd(pae_matrix.rows(), pae_matrix.cols());
    weights = MatrixXd::Constant(pae_matrix.rows(), pae_matrix.cols(), 1.0).array() / pae_matrix.array().pow(pae_power);

    igraph_t g;
    igraph_empty(&g, 0, IGRAPH_UNDIRECTED);
    igraph_integer_t size = weights.rows();
    igraph_add_vertices(&g, size, 0);

    MatrixXd temp_matrix = pae_matrix.unaryExpr([this](double val) { return val >= this->pae_cutoff ? 0 : val; });
    Eigen::SparseMatrix<double> edges = temp_matrix.sparseView();

    VectorXd sel_weights(edges.nonZeros());
    int counter = 0;
    for (int k=0; k < edges.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(edges,k); it; ++it) {
            sel_weights(counter++) = weights(it.row(), it.col());
        }
    }

    std::vector<igraph_integer_t> edge_list;
    for (int k=0; k < edges.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(edges,k); it; ++it) {
            edge_list.push_back(it.row());
            edge_list.push_back(it.col());
        }
    }

    igraph_vector_int_t igraph_edge_list;
    igraph_vector_int_init(&igraph_edge_list, edge_list.size());

    for (int i = 0; i < edge_list.size(); ++i) {
        VECTOR(igraph_edge_list)[i] = edge_list[i];
    }
    igraph_add_edges(&g, &igraph_edge_list, 0);

    igraph_real_t resolution = graph_resolution / 100;
    igraph_real_t beta = 0.01;
    igraph_bool_t start = 1;
    igraph_vector_int_t community_sizes;

    igraph_vector_int_init(&community_sizes, 0);

    igraph_vector_t igraph_weights;
    igraph_vector_init(&igraph_weights, sel_weights.size());

    for (int i = 0; i < sel_weights.size(); ++i) {
        VECTOR(igraph_weights)[i] = sel_weights(i);
    }

    igraph_vector_int_t membership;
    igraph_vector_int_init(&membership, igraph_vcount(&g));

    for (int i = 0; i < igraph_vcount(&g); ++i) {
        VECTOR(membership)[i] = 0;  // or i for unique values
    }

    igraph_community_leiden(
        &g, 
        &igraph_weights, 
        NULL, 
        resolution, 
        beta,
        start,
        -1, 
        &membership, 
        NULL, 
        NULL);

    labels_.resize(igraph_vector_int_size(&membership));
    for (int i = 0; i < igraph_vector_int_size(&membership); ++i) {
        labels_(i) = VECTOR(membership)[i];
    }

    igraph_destroy(&g);
    igraph_vector_int_destroy(&membership);
    igraph_vector_destroy(&igraph_weights);

    if ((nclusters < 0 || !merge_clusters) && (nclusters > labels_.maxCoeff() + 1)) {
        return;
    } else {
        // Calculate the centroids of the clusters
        std::set<int> cluster_ids(labels_.data(), labels_.data() + labels_.size());
        MatrixXd centroids = MatrixXd::Zero(labels_.size(), 3);

        for (int cluster_id : cluster_ids) {
            std::vector<Eigen::RowVector3d> centroid_cluster;
            for (int idx = 0; idx < labels_.size(); ++idx) {
                if (labels_(idx) == cluster_id) {
                    centroid_cluster.push_back(X.row(idx));
                }
            }

            MatrixXd centroid_cluster_matrix(centroid_cluster.size(), 3);
            for (size_t i = 0; i < centroid_cluster.size(); ++i) {
                centroid_cluster_matrix.row(i) = centroid_cluster[i];
            }

            Eigen::RowVector3d centroid = centroid_cluster_matrix.colwise().mean();

            for (int idx = 0; idx < labels_.size(); ++idx) {
                if (labels_(idx) == cluster_id) {
                    centroids.row(idx) = centroid;
                }
            }
        }

        // Merge clusters with agglomerative
        Agglomerative agglomerative(nclusters);
        agglomerative.fit(centroids);
        labels_ = agglomerative.labels_;
    }

    return;
}