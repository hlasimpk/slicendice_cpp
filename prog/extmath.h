#include <Eigen/Dense>

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

