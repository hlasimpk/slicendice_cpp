#include <Eigen/Dense>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

// Class to run KMeans plus plus initialisation on a 3D dataset
class KMeans_plus_plus {
    const MatrixXd& data; // The data to pick seeds from :  array-like (assumes dense matrix), shape (n_samples, n_features)
    int nclusters; // The number of centroids to initialise : int
    VectorXd sample_weight; // The weights for each observation in `X`. If empty, all observations are assigned equal weight : array-like, shape (n_samples,), optional
    VectorXd x_squared_norms; // Squared Euclidean norm of each data point : array-like, shape (n_samples,), optional
    int seed; // Seed used to determine random number generation for centroid initialization.
    int n_local_trials; // The number of seeding trials for each center (except the first), of which the one reducing inertia the most is chosen. : int, default=None

    public:
        KMeans_plus_plus(const MatrixXd& input_data, int input_nclusters = 2, VectorXd input_sample_weight = {}, VectorXd input_x_squared_norms = {}, int seed = 0, int input_n_local_trials = 0);
        std::pair<MatrixXd, VectorXi> run();

    private:
        int get_random_center();
};

// Class to run KMeans on a 3D dataset
class KMeans {
    int nclusters; // The number of clusters to form as well as the number of centroids to generate: int (default=8)
    int n_init; // Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. : int (default=10)
    int max_iter; // Maximum number of iterations of the k-means algorithm for a single run. : int (default=300)
    double tol; // Relative tolerance with regards to inertia to declare convergence : double (default=1e-4)
    bool copy_x; // When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True, then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. : bool (default=True)
    std::string algorithm; //   K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`. The `"elkan"` variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it's more memory intensive due to the allocation of an extra array of shape `(n_samples, n_clusters)`.: string (default="lloyd")

    public:
        MatrixXd cluster_centers_;
        VectorXi n_features_out_;
        VectorXi labels_;
        double inertia_;

        KMeans(int input_nclusters = 8, int input_n_init = 10, int input_max_iter = 300, double input_tol = 1e-4, bool input_copy_x = true, std::string input_algorithm = "elkan");
        void fit(MatrixXd& data, VectorXd sample_weight = {});

    private:
        void check_params_vs_input(MatrixXd& data);
        double tolerance(MatrixXd& data);
        double calculate_inertia(const MatrixXd& X, 
                                 const VectorXd& sample_weight, 
                                 const MatrixXd& centers, 
                                 const VectorXi& labels, 
                                 int n_threads,
                                 int single_label = -1);
        bool is_same_clustering(const VectorXi& labels1, const VectorXi& labels2, int n_clusters);
        std::tuple<VectorXi, double, MatrixXd> _kmeans_single_elkan(MatrixXd& data, 
                                                                    VectorXd& sample_weight, 
                                                                    MatrixXd centers_init, 
                                                                    int& max_iter, 
                                                                    double& tol);
        std::tuple<VectorXi, double, MatrixXd> _kmeans_single_lloyd(MatrixXd& data, 
                                                                    VectorXd& sample_weight, 
                                                                    MatrixXd centers_init, 
                                                                    int& max_iter, 
                                                                    double& tol);
};