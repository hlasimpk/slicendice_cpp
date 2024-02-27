#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

extern const int CHUNK_SIZE;

void relocate_empty_clusters(
    const MatrixXd& X,
    const VectorXd& sample_weight,
    const MatrixXd& centers_old,
    MatrixXd& centers_new,
    VectorXd& weight_in_clusters,
    const VectorXi& labels,
    int n_clusters);

void average_centers(MatrixXd& centers, const VectorXd& weight_in_clusters);

void update_center_shift(const MatrixXd& centers, 
                         const MatrixXd& centers_new, 
                         VectorXd& center_shift);