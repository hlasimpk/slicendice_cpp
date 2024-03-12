#include "extmath.h"

#include <iostream>
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


class CFNode;

class CFSubcluster {
    public:
        int n_samples_;
        VectorXd linear_sum_;
        double squared_sum_;
        VectorXd centroid_;
        CFNode* child_;
        double sq_norm_;

        CFSubcluster(VectorXd linear_sum = VectorXd()) {
            if (linear_sum.size() == 0) {
                n_samples_ = 0;
                squared_sum_ = 0.0;
                centroid_ = linear_sum_ = VectorXd();
            } else {
                n_samples_ = 1;
                centroid_ = linear_sum_ = linear_sum;
                squared_sum_ = sq_norm_ = linear_sum.dot(linear_sum);
            }
            child_ = nullptr;
        }

        void update(CFSubcluster& subcluster) {
            n_samples_ += subcluster.n_samples_;
            int max_size = std::max(linear_sum_.size(), subcluster.linear_sum_.size());
            Eigen::VectorXd combined_linear_sum = Eigen::VectorXd::Zero(max_size);
            if(linear_sum_.size() > 0)
                combined_linear_sum.head(linear_sum_.size()) = linear_sum_;
            if(subcluster.linear_sum_.size() > 0)
                combined_linear_sum.head(subcluster.linear_sum_.size()) += subcluster.linear_sum_;
            linear_sum_ = combined_linear_sum;
            squared_sum_ += subcluster.squared_sum_;
            centroid_ = linear_sum_ / n_samples_;
            sq_norm_ = centroid_.dot(centroid_);
        }

        bool merge_subcluster(CFSubcluster& nominee_cluster, double threshold) {
            double new_ss = squared_sum_ + nominee_cluster.squared_sum_;
            VectorXd new_ls = linear_sum_ + nominee_cluster.linear_sum_;
            int new_n = n_samples_ + nominee_cluster.n_samples_;
            VectorXd new_centroid = new_ls / new_n;
            double new_sq_norm = new_centroid.dot(new_centroid);

            double sq_radius = new_ss / new_n - new_sq_norm;

            if (sq_radius <= threshold * threshold) {
                n_samples_ = new_n;
                linear_sum_ = new_ls;
                squared_sum_ = new_ss;
                centroid_ = new_centroid;
                sq_norm_ = new_sq_norm;
                return true;
            }
            return false;
        }

        bool is_equal(const CFSubcluster& subcluster) {
            return (n_samples_ == subcluster.n_samples_ &&
                    linear_sum_ == subcluster.linear_sum_ &&
                    squared_sum_ == subcluster.squared_sum_ &&
                    centroid_ == subcluster.centroid_ &&
                    sq_norm_ == subcluster.sq_norm_);
        }
};


class CFNode {
    public:
        double threshold;
        int branching_factor;
        bool is_leaf;
        int n_features;
        std::vector<CFSubcluster> subclusters;
        MatrixXd init_centroids;
        VectorXd init_sq_norm;
        VectorXd squared_norm;
        MatrixXd centroids;
        CFNode* prev_leaf = nullptr;
        CFNode* next_leaf = nullptr;

        CFNode(double input_threshold, 
               int input_branching_factor, 
               bool input_is_leaf, 
               int input_n_features) :
            threshold(input_threshold),
            branching_factor(input_branching_factor),
            is_leaf(input_is_leaf),
            n_features(input_n_features),
            init_centroids(MatrixXd::Zero(input_branching_factor + 1, input_n_features)),
            init_sq_norm(VectorXd::Zero(input_branching_factor + 1)) {};

        static std::pair<CFSubcluster, CFSubcluster> split_node(CFNode& node, double threshold, int branching_factor) {
            CFSubcluster new_subcluster1, new_subcluster2;
            CFNode* new_node1 = new CFNode(threshold, branching_factor, node.is_leaf, node.n_features);
            CFNode* new_node2 = new CFNode(threshold, branching_factor, node.is_leaf, node.n_features);
            new_subcluster1.child_ = new_node1;
            new_subcluster2.child_ = new_node2;


            if (node.is_leaf) {
                if (node.prev_leaf != nullptr) {
                    node.prev_leaf->next_leaf = new_node1;
                }
                new_node1->prev_leaf = node.prev_leaf;
                new_node1->next_leaf = new_node2;
                new_node2->prev_leaf = new_node1;
                new_node2->next_leaf = node.next_leaf;

                if (node.next_leaf != nullptr) {
                    node.next_leaf->prev_leaf = new_node2;
                }
            }

            MatrixXd dist = euclidean_distances(node.centroids, node.centroids, VectorXd(), node.squared_norm, true);

            int n_clusters = dist.rows();

            Eigen::MatrixXi::Index maxRow, maxCol;
            double max = dist.maxCoeff(&maxRow, &maxCol);
            std::pair<int, int> farthest_idx = std::make_pair(maxRow, maxCol);
            
            VectorXd node1_dist = dist.row(farthest_idx.first);
            VectorXd node2_dist = dist.col(farthest_idx.second);

            VectorXi node1_closer = (node1_dist.array() < node2_dist.array()).cast<int>();
            node1_closer(farthest_idx.first) = 1;

            for (int idx = 0; idx < node.subclusters.size(); ++idx) {
                if (node1_closer(idx)) {
                    new_node1->append_subcluster(node.subclusters[idx]);
                    new_subcluster1.update(node.subclusters[idx]);
                } else {
                    new_node2->append_subcluster(node.subclusters[idx]);
                    new_subcluster2.update(node.subclusters[idx]);
                }
            }
            return std::make_pair(new_subcluster1, new_subcluster2);
        }

        void append_subcluster(CFSubcluster subcluster) {
            int n_samples = subclusters.size();
            subclusters.push_back(subcluster);
            init_centroids.row(n_samples) = subcluster.centroid_;
            init_sq_norm(n_samples) = subcluster.sq_norm_;

            // Keep centroids and squared norm as views. In this way
            // if we change init_centroids and init_sq_norm_, it is
            // sufficient,
            centroids = init_centroids.topRows(n_samples + 1);
            squared_norm = init_sq_norm.head(n_samples + 1);
        }

        void update_split_subclusters(CFSubcluster subcluster, CFSubcluster new_subcluster1, CFSubcluster new_subcluster2) {
            // Find the index of the subcluster to replace
            int ind = -1;
            for (int i = 0; i < subclusters.size(); ++i) {
                if (subclusters[i].is_equal(subcluster)) {
                    ind = i;
                    break;
                }
            }

            // Replace the subcluster, centroid, and squared norm at the found index
            if (ind != -1) {
                subclusters[ind] = new_subcluster1;
                init_centroids.row(ind) = new_subcluster1.centroid_;
                init_sq_norm(ind) = new_subcluster1.sq_norm_;

                // Append the new subcluster
                append_subcluster(new_subcluster2);
            }
        }

        bool insert_cf_subcluster(CFSubcluster subcluster) {
            // Insert a new subcluster into the node.
            if (subclusters.empty()) {
                append_subcluster(subcluster);
                return false;
            }

            // We need to find the closest subcluster among all the
            // subclusters so that we can insert our new subcluster.
            MatrixXd dist_matrix = centroids * subcluster.centroid_;
            dist_matrix *= -2.0;
            dist_matrix += squared_norm;
            int closest_index = std::distance(dist_matrix.data(), std::min_element(dist_matrix.data(), dist_matrix.data() + dist_matrix.size()));
            CFSubcluster closest_subcluster = subclusters[closest_index];
            
            // If the subcluster has a child, we need a recursive strategy.
            if (closest_subcluster.child_ != nullptr) {
                bool split_child = closest_subcluster.child_->insert_cf_subcluster(subcluster);

                if (!split_child) {
                    // If it is determined that the child need not be split, we
                    // can just update the closest_subcluster
                    closest_subcluster.update(subcluster);
                    init_centroids.row(closest_index) = closest_subcluster.centroid_;
                    init_sq_norm(closest_index) = closest_subcluster.sq_norm_;
                    return false;
                }

                // things not too good. we need to redistribute the subclusters in
                // our child node, and add a new subcluster in the parent
                // subcluster to accommodate the new child.
                else {
                    std::pair<CFSubcluster, CFSubcluster> new_subclusters = split_node(*closest_subcluster.child_, threshold, branching_factor);
                    update_split_subclusters(closest_subcluster, new_subclusters.first, new_subclusters.second);

                    if (subclusters.size() > branching_factor) {
                        return true;
                    }
                    return false;
                }
            }

            else {
                bool merged = closest_subcluster.merge_subcluster(subcluster, threshold);
                if (merged) {
                    init_centroids.row(closest_index) = closest_subcluster.centroid_;
                    init_sq_norm(closest_index) = closest_subcluster.sq_norm_;
                    return false;
                }

                // not close to any other subclusters, and we still
                // have space, so add.
                else if (subclusters.size() < branching_factor) {
                    append_subcluster(subcluster);
                    return false;
                }

                // We do not have enough space nor is it closer to an
                // other subcluster. We need to split.
                else {
                    append_subcluster(subcluster);
                    return true;
                }
            }
        }
};


class Birch {
    double threshold; 
    int branching_factor;
    int nclusters; 
    bool compute_labels;
    bool copy;
    MatrixXd subcluster_centers;
    int n_features_out;
    VectorXd subcluster_norms;
    VectorXi subcluster_labels;

    public:
        MatrixXd cluster_centers_;
        VectorXi n_features_out_;
        VectorXi labels_;
        double inertia_;
        CFNode* root_;
        CFNode* dummy_leaf_;

        Birch(double input_threshold = 0.5, int input_branching_factor = 50, int input_nclusters = 3, bool input_compute_labels = true, bool input_copy = true);
        void fit(MatrixXd& X, bool partial = false);

    private:
        std::vector<CFNode*> get_leaves();
        void global_clustering(MatrixXd& X);
        VectorXi _predict(const MatrixXd& X);
};