#include <Eigen/Dense>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


MatrixXd euclidean_distances (const MatrixXd& X, 
                              const MatrixXd& Y, 
                              const VectorXd& x_squared_norms, 
                              const VectorXd& y_squared_norms, 
                              bool squared = false);

VectorXd calculate_mean (const MatrixXd& X);

VectorXd calculate_variance (const MatrixXd& X);

MatrixXd partition (const MatrixXd& v, int kth);

VectorXi unique(const VectorXi& v);

std::pair<Eigen::VectorXi, Eigen::VectorXi> triu_indices(int n);