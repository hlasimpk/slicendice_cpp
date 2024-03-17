#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

class PAE {
    int nclusters;
    std::string pae_file;
    bool merge_clusters; // If true, merge clusters until the number of clusters is equal to nclusters 
    double pae_power;
    double pae_cutoff; 
    double graph_resolution; 
    double distance_power;


    public:
        MatrixXd cluster_centers_;
        VectorXi n_features_out_;
        VectorXi labels_;
        double inertia_;
        MatrixXd pae_matrix;

        PAE(int input_nclusters = 3,
            std::string input_pae_file = "",
            bool input_merge_clusters = true,
            double input_pae_power = 1.0, 
            double input_pae_cutoff = 5.0, 
            double input_graph_resolution = 1.0, 
            double input_distance_power = 1.0);
        void fit(MatrixXd& X);
        void parse_pae_file(std::string pae_file);
};