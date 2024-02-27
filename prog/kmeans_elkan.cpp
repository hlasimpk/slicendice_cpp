#include "kmeans_common.h"
#include "kmeans_elkan.h"

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


void init_bounds(
    MatrixXd& data, 
    MatrixXd& centers, 
    MatrixXd& center_half_distances, 
    VectorXi& labels, 
    MatrixXd& lower_bounds, 
    VectorXd& upper_bounds) {

    int n_samples = data.rows();
    int n_clusters = centers.rows();
    int n_features = data.cols();

    double min_dist, dist;
    int best_cluster;

    for (int i = 0; i < n_samples; ++i) {
        best_cluster = 0;
        min_dist = (data.row(i) - centers.row(0)).norm();
        lower_bounds(i, 0) = min_dist;

        for (int j = 1; j < n_clusters; ++j) {
            if (min_dist > center_half_distances(best_cluster, j)) {
                dist = (data.row(i) - centers.row(j)).norm();
                lower_bounds(i, j) = dist;
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
        }
        labels(i) = best_cluster;
        upper_bounds(i) = min_dist;
    }
}

void elkan_iter(
    MatrixXd& X, 
    VectorXd& sample_weight, 
    MatrixXd& centers, 
    MatrixXd& centers_new, 
    VectorXd& weight_in_clusters, 
    MatrixXd& center_half_distances, 
    VectorXd& distance_next_center, 
    VectorXd& upper_bounds, 
    MatrixXd& lower_bounds, 
    VectorXi& labels, 
    VectorXd& center_shift,
    bool update_centers){

    int n_samples = X.rows();
    int n_features = X.cols();
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

    MatrixXd centers_new_chunk = MatrixXd::Zero(n_clusters, n_features);
    VectorXd weight_in_clusters_chunk = VectorXd::Zero(n_clusters);

    if (update_centers) {
        centers_new.setZero();
        weight_in_clusters.setZero();
    }

    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        int start = chunk_idx * n_samples_chunk;
        int end = (chunk_idx == n_chunks - 1 && n_samples_rem) ? start + n_samples_rem : start + n_samples_chunk;

        update_chunk_elkan(
                           X,
                           sample_weight,
                           centers,
                           center_half_distances,
                           distance_next_center,
                           labels,
                           upper_bounds,
                           lower_bounds,
                           centers_new_chunk,
                           weight_in_clusters_chunk,
                           update_centers,
                           start,
                           end);
    }

    if (update_centers) {
        weight_in_clusters += weight_in_clusters_chunk;
        centers_new += centers_new_chunk;
    }

    if (update_centers) {
        relocate_empty_clusters(
            X, sample_weight, centers, centers_new, weight_in_clusters, labels, n_clusters
        );
        average_centers(centers_new, weight_in_clusters);
        update_center_shift(centers, centers_new, center_shift);

        for (int i = 0; i < n_samples; ++i) {
            upper_bounds(i) += center_shift(labels(i));

            for (int j = 0; j < n_clusters; ++j) {
                lower_bounds(i, j) -= center_shift(j);
                if (lower_bounds(i, j) < 0) {
                    lower_bounds(i, j) = 0;
                }
            }
        }
    }
}

void update_chunk_elkan(
    const MatrixXd& X,
    const VectorXd& sample_weight,
    MatrixXd& centers,
    MatrixXd& center_half_distances,
    VectorXd& distance_next_center,
    VectorXi& labels,
    VectorXd& upper_bounds,
    MatrixXd& lower_bounds,
    MatrixXd& centers_new_chunk,
    VectorXd& weight_in_clusters,
    bool update_centers,
    int start,
    int end) {
        
    // Create the segments inside the function
    auto X_chunk = X.middleRows(start, end - start);
    VectorXd::ConstSegmentReturnType sample_weight_chunk = sample_weight.segment(start, end - start);
    VectorXi::SegmentReturnType labels_chunk = labels.segment(start, end - start);
    VectorXd::SegmentReturnType upper_bounds_chunk = upper_bounds.segment(start, end - start);
    auto lower_bounds_chunk = lower_bounds.middleRows(start, end - start);

    int n_samples = X_chunk.rows();
    int n_clusters = centers.rows();
    int n_features = X_chunk.cols();

    double upper_bound, distance;
    bool bounds_tight;

    for (int i=0; i < n_samples; ++i){
        upper_bound = upper_bounds_chunk(i);
        bounds_tight = false;
        int label = labels_chunk(i);

        if (!(distance_next_center(label) < upper_bound)) {
            for (int j = 0; j < n_clusters; ++j) {
                if (j != label && (upper_bound > lower_bounds_chunk(i, j)) && (upper_bound > center_half_distances(label, j))) {
                    if (!bounds_tight) {
                        upper_bound = (X_chunk.row(i) - centers.row(label)).norm();
                        lower_bounds_chunk(i, label) = upper_bound;
                        bounds_tight = true;
                    }

                    if (upper_bound > lower_bounds_chunk(i, j) || (upper_bound > center_half_distances(label, j))) {
                        distance = (X_chunk.row(i) - centers.row(j)).norm();
                        lower_bounds_chunk(i, j) = distance;
                        if (distance < upper_bound) {
                            upper_bound = distance;
                            label= j;
                        }
                    }
                }
            }

            labels_chunk(i) = label;
            upper_bounds_chunk(i) = upper_bound;
        }

        if (update_centers) {
            weight_in_clusters(label) += sample_weight_chunk(i);
            for (int k = 0; k < n_features; ++k) {
                centers_new_chunk(label, k) += X_chunk(i, k) * sample_weight_chunk(i);
            }
        }
    }
}