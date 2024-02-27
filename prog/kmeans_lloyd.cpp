#include "kmeans_common.h"
#include "kmeans_lloyd.h"

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


void lloyd_iter(
    MatrixXd& X, 
    VectorXd& sample_weight, 
    MatrixXd& centers, 
    MatrixXd& centers_new, 
    VectorXd& weight_in_clusters, 
    VectorXi& labels, 
    VectorXd& center_shift,
    bool update_centers){

    int n_samples = X.rows();
    int n_features = centers.cols();
    int n_clusters = centers_new.rows();

    // An empty array was passed, do nothing and return early
    if (n_samples == 0) {
        return;
    }

    int n_samples_chunk  = std::min(CHUNK_SIZE, n_samples);
    int n_chunks = n_samples / n_samples_chunk;
    int n_samples_rem = n_samples % n_samples_chunk;

    if (n_samples != n_chunks * n_samples_chunk) {
        n_chunks += 1;
    }
    
    VectorXd centers_squared_norms = centers.rowwise().squaredNorm();
    MatrixXd centers_new_chunk = MatrixXd::Zero(n_clusters, n_features);
    VectorXd weight_in_clusters_chunk = VectorXd::Zero(n_clusters);
    MatrixXd pairwise_distances_chunk = MatrixXd::Zero(n_samples, n_clusters);

    if (update_centers) {
        centers_new.setZero();
        weight_in_clusters.setZero();
    }

    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        int start = chunk_idx * n_samples_chunk;
        int end = (chunk_idx == n_chunks - 1 && n_samples_rem) ? start + n_samples_rem : start + n_samples_chunk;

        update_chunk_lloyd(
            X,
            sample_weight,
            centers,
            centers_squared_norms,
            labels,
            centers_new_chunk,
            weight_in_clusters_chunk,
            pairwise_distances_chunk,
            update_centers,
            start,
            end
        );
    }

    if (update_centers) {
        relocate_empty_clusters(
            X, sample_weight, centers, centers_new, weight_in_clusters, labels, n_clusters
        );
        average_centers(centers_new, weight_in_clusters);
        update_center_shift(centers, centers_new, center_shift);
    }


    return;
}

void update_chunk_lloyd(
    const MatrixXd& X,
    const VectorXd& sample_weight,
    const MatrixXd& centers,
    const VectorXd& centers_squared_norms,
    VectorXi& labels,
    MatrixXd& centers_new,
    VectorXd& weight_in_clusters,
    MatrixXd& pairwise_distances, 
    bool update_centers,
    int start,
    int end) {

    // Create the segments inside the function
    auto X_chunk = X.middleRows(start, end - start);
    VectorXi::SegmentReturnType labels_chunk = labels.segment(start, end - start);
        
    // K-means combined EM step for one dense data chunk.
    // Compute the partial contribution of a single data chunk to the labels and centers.
    int n_samples = X_chunk.rows();
    int n_clusters = centers.rows();
    int n_features = X_chunk.cols();

    // Instead of computing the full pairwise squared distances matrix,
    // ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    // the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    // depends on the centers.
    // pairwise_distances = ||C||²
    pairwise_distances = centers_squared_norms.transpose().replicate(n_samples, 1);

    // pairwise_distances += -2 * X.dot(C.T)
    pairwise_distances += -2 * X_chunk * centers.transpose();

    double min_sq_dist;
    int label;
    for (int i = 0; i < n_samples; i++) {
        min_sq_dist = pairwise_distances(i, 0);
        label = 0;
        for (int j = 1; j < n_clusters; j++) {
            double sq_dist = pairwise_distances(i, j);
            if (sq_dist < min_sq_dist) {
                min_sq_dist = sq_dist;
                label = j;
            }
        }
        labels_chunk(i) = label;

        if (update_centers) {
            weight_in_clusters(label) += sample_weight(start + i);
            centers_new.row(label) += X_chunk.row(i) * sample_weight(start + i);
        }
    }
}