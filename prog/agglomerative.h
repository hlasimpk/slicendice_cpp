#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <unordered_map>

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::VectorXi;


struct MatrixHash {
    size_t operator()(const MatrixXd& matrix) const {
        std::hash<double> hasher;
        size_t seed = 0;
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                seed ^= hasher(matrix(i, j)) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            }
        }
        return seed;
    }
};


// Class to run Agglomerative on a 3D dataset
class Agglomerative {
    int nclusters;
    std::string metric;
    std::unordered_map<MatrixXd, std::tuple<MatrixXd, int, int, VectorXi, MatrixXd>, MatrixHash> memory;
    SparseMatrix<double> connectivity;
    std::string compute_full_tree;
    std::string linkage;
    double distance_threshold;
    bool compute_distance;

    public:
        MatrixXd cluster_centers_;
        VectorXi n_features_out_;
        VectorXi labels_;
        double inertia_;

        Agglomerative(int input_nclusters = 8, 
                      std::string input_metric = "euclidean", 
                      std::unordered_map<MatrixXd, std::tuple<MatrixXd, int, int, VectorXi, MatrixXd>, MatrixHash> input_memory = {}, 
                      SparseMatrix<double> input_connectivity = SparseMatrix<double>(), 
                      std::string input_compute_full_tree="auto", 
                      std::string input_linkage = "ward",
                      double input_distance_threshold=-1.0, 
                      bool input_compute_distance=false);
        void fit(MatrixXd& data);

    private:
        void check_params();
        std::tuple<MatrixXd, int, int, VectorXi, MatrixXd>  ward_tree(
            MatrixXd X, 
            SparseMatrix<double>& connectivity, 
            int n_clusters, 
            bool return_distance);
        std::tuple<MatrixXd, int, int, VectorXi, MatrixXd> linkage_tree(
            MatrixXd X, 
            SparseMatrix<double>& connectivity, 
            int n_clusters,
            std::string linkage="complete",
            std::string affinity="euclidean", 
            bool return_distance=true);
        int connected_components_directed(const int* indices,
                                          const int* indptr,
                                          VectorXi& labels);
        int connected_components_undirected(const int* indices1,
                                            const int* indptr1,
                                            const int* indices2,
                                            const int* indptr2,
                                            VectorXi& labels);
        std::pair<int, VectorXi> connected_components(const SparseMatrix<double>& connectivity, 
                                                      bool directed=true, 
                                                      std::string connection="weak", 
                                                      bool return_labels=true);
        SparseMatrix<double> fix_connected_components(
            MatrixXd& X, 
            SparseMatrix<double>& connectivity, 
            int n_connected_components, 
            VectorXi& labels, 
            std::string affinity, 
            std::string connectivity_name);
        std::pair<SparseMatrix<double>, int> fix_connectivity(MatrixXd& X, SparseMatrix<double>& connectivity, std::string affinity);

};