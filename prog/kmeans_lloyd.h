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
    bool update_centers);

void update_chunk_lloyd(
    const MatrixXd& X,
    const VectorXd& sample_weight,
    const MatrixXd& centers_old,
    const VectorXd& centers_squared_norms,
    VectorXi& labels,
    MatrixXd& centers_new,
    VectorXd& weight_in_clusters,
    MatrixXd& pairwise_distances, 
    bool update_centers,
    int start,
    int end);

