#include "kmeans_common.h"

#include <Eigen/Dense>
#include <numeric>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

const int CHUNK_SIZE = 256;

void relocate_empty_clusters(
    const MatrixXd& X,
    const VectorXd& sample_weight,
    const MatrixXd& centers_old,
    MatrixXd& centers_new,
    VectorXd& weight_in_clusters,
    const VectorXi& labels,
    int n_clusters) {

    // Find the indices of the empty clusters
    std::vector<int> empty_clusters;
    for (int i = 0; i < weight_in_clusters.size(); i++) {
        if (weight_in_clusters(i) == 0) {
            empty_clusters.push_back(i);
        }
    }
    int n_empty = empty_clusters.size();

    if (n_empty == 0) {
        return;
    }

    int n_features = X.cols();

    // Compute the distances from each sample to its nearest center
    MatrixXd centers_old_full = MatrixXd::Zero(n_clusters, n_features);
    centers_old_full.block(0, 0, centers_old.rows(), centers_old.cols()) = centers_old;

    MatrixXd centers_for_X = MatrixXd::Zero(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++) {
        centers_for_X.row(i) = centers_old_full.row(labels(i));
    }

    VectorXd distances = (X - centers_for_X).rowwise().squaredNorm(); 

    // Initialize indices
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, ..., X.rows()-1

    // Find the indices of the samples that are farthest from their centers
    std::vector<int> far_from_centers(n_empty);
    std::partial_sort_copy(
        indices.begin(), indices.end(),
        far_from_centers.begin(), far_from_centers.end(),
        [&distances](int i, int j) { return distances(i) > distances(j); });

    for (int idx = 0; idx < n_empty; idx++) {
        int new_cluster_id = empty_clusters[idx];
        int far_idx = far_from_centers[idx];
        double weight = sample_weight(far_idx);
        int old_cluster_id = labels(far_idx);

        for (int k = 0; k < n_features; k++) {
            centers_new(old_cluster_id, k) -= X(far_idx, k) * weight;
            centers_new(new_cluster_id, k) = X(far_idx, k) * weight;
        }

        weight_in_clusters(new_cluster_id) = weight;
        weight_in_clusters(old_cluster_id) -= weight;
    }
}

void average_centers(MatrixXd& centers, const VectorXd& weight_in_clusters) {
    int n_clusters = centers.rows();
    int n_features = centers.cols();

    for (int j = 0; j < n_clusters; ++j) {
        if (weight_in_clusters(j) > 0) {
            double alpha = 1.0 / weight_in_clusters(j);
            centers.row(j) *= alpha;
        }
    }
}

void update_center_shift(
    const MatrixXd& centers_old, 
    const MatrixXd& centers_new, 
    VectorXd& center_shift) {
    
    int n_clusters = centers_old.rows();

    for (int j = 0; j < n_clusters; ++j) {
        center_shift(j) = (centers_new.row(j) - centers_old.row(j)).norm();
    }
}