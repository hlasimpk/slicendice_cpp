#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

void init_bounds(MatrixXd& data, 
                 MatrixXd& centers, 
                 MatrixXd& center_half_distances, 
                 VectorXi& labels, 
                 MatrixXd& lower_bounds, 
                 VectorXd& upper_bounds);
void elkan_iter(MatrixXd& X, 
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
                bool update_centers);
void update_chunk_elkan(const MatrixXd& X, 
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
                        int end);
