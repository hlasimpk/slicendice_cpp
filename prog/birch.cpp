#include "agglomerative.h"
#include "birch.h"

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

Birch::Birch(double input_threshold, int input_branching_factor, int input_nclusters, bool input_compute_labels, bool input_copy) :
    threshold(input_threshold),
    branching_factor(input_branching_factor),
    nclusters(input_nclusters),
    compute_labels(input_compute_labels),
    copy(input_copy) {}

std::vector<CFNode*> Birch::get_leaves() {
    CFNode* current = dummy_leaf_;
    std::vector<CFNode*> leaves;
    while (current->next_leaf != nullptr) {
        leaves.push_back(current->next_leaf);
        current = current->next_leaf;
    }
    return leaves;
}


VectorXi Birch::_predict(const MatrixXd& X) {
    VectorXi labels(X.rows());

    for (int i = 0; i < X.rows(); ++i) {
        // Calculate the squared Euclidean distance from the current point to each centroid
        VectorXd dists = (subcluster_centers.rowwise() - X.row(i)).rowwise().squaredNorm();

        // Find the index of the closest centroid
        VectorXd::Index min_index;
        dists.minCoeff(&min_index);

        // Assign the label of the closest centroid to the current point
        labels(i) = subcluster_labels(min_index);
    }
    return labels;
}


void Birch::global_clustering(Eigen::MatrixXd& X) {
    if (!X.isZero(0) && compute_labels) {
        compute_labels = true;
    }
    bool not_enough_centroids = false;

    Agglomerative clusterer(nclusters);

    if (n_features_out < nclusters) {
        not_enough_centroids = true;
    }

    if (not_enough_centroids) {
        subcluster_labels = VectorXi::LinSpaced(n_features_out, 0, n_features_out - 1);

        if (not_enough_centroids) {
            std::cerr << "Number of subclusters found (" << n_features_out << ") by BIRCH is less than (" << nclusters << "). Decrease the threshold.\n";
        }
    } else {
        clusterer.fit(subcluster_centers);
        subcluster_labels = clusterer.labels_;
    }

    if (compute_labels) {
        labels_ = _predict(X);
    }

}

void Birch::fit(MatrixXd& X, bool partial) {
    bool has_root = (root_ != nullptr);
    bool first_call = !(has_root && partial);

    int n_samples = X.rows();
    int n_features = X.cols();


    if (first_call) {
        root_ = new CFNode(threshold, branching_factor, true, n_features);

        // To enable getting back subclusters
        dummy_leaf_ = new CFNode(threshold, branching_factor, true, n_features);

        dummy_leaf_->next_leaf = root_;
        root_->prev_leaf = dummy_leaf_;
    }

    for (int i = 0; i < X.rows(); ++i) {
        VectorXd sample = X.row(i);
        CFSubcluster subcluster(sample);
        bool split = root_->insert_cf_subcluster(subcluster);

        if (split) {
            std::pair<CFSubcluster, CFSubcluster> new_subclusters = CFNode::split_node(*root_, threshold, branching_factor);
            delete root_;
            root_ = new CFNode(threshold, branching_factor, false, n_features);
            root_->append_subcluster(new_subclusters.first);
            root_->append_subcluster(new_subclusters.second);
        }

    }

    std::vector<MatrixXd> centroids;
    std::vector<CFNode*> leaves = get_leaves();
    for (CFNode* leaf : leaves) {
        centroids.push_back(leaf->centroids);
    }

    int total_rows = 0;
    int cols = centroids[0].cols();

    // Calculate total number of rows
    for (const auto& matrix : centroids) {
        total_rows += matrix.rows();
    }

    subcluster_centers.resize(total_rows, cols);

    int current_row = 0;
    for (const auto& matrix : centroids) {
        // Copy the data from the current matrix to the combined matrix
        subcluster_centers.block(current_row, 0, matrix.rows(), cols) = matrix;
        current_row += matrix.rows();
    }

    n_features_out = subcluster_centers.rows();
    global_clustering(X);
}
