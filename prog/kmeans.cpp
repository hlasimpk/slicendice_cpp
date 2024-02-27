#include "kmeans.h"
#include "kmeans_elkan.h"
#include "kmeans_lloyd.h"
#include "extmath.h"

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <set>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


KMeans_plus_plus::KMeans_plus_plus(const MatrixXd& input_data, 
                                   int input_nclusters, 
                                   VectorXd input_sample_weight, 
                                   VectorXd input_x_squared_norms, 
                                   int input_seed, 
                                   int input_n_local_trials) : 
data(input_data), 
nclusters(input_nclusters), 
sample_weight(input_sample_weight), 
x_squared_norms(input_x_squared_norms), 
seed(input_seed), 
n_local_trials(input_n_local_trials) {}

int KMeans_plus_plus::get_random_center() {
    VectorXd weights = sample_weight / sample_weight.sum();

    std::random_device rd;
    std::mt19937 gen(rd());
    if (seed != 0) {
        gen.seed(seed);
    }
    std::discrete_distribution<int> dist(weights.data(), weights.data() + weights.size());
    return dist(gen);
}

std::pair<MatrixXd, VectorXi> KMeans_plus_plus::run() {
    int n_samples = data.rows();
    int n_features = data.cols();

    MatrixXd centers(nclusters, n_features);
    
    // Set the number of local seeding trials if 0 is given
    if (n_local_trials == 0) {
        n_local_trials = 2 + int(std::log(nclusters));
    }

    // Set the sample weights if none are given
    if (sample_weight.size() == 0) {
        sample_weight = VectorXd::Ones(n_samples);
    }


    // Set the random state
    int center_id = get_random_center();

    // Set indices to -1
    VectorXi indices = VectorXi::Constant(nclusters, -1);

    centers.row(0) = data.row(center_id);
    indices(0) = center_id;

    // Initialise list of closest distances and calculate the current potential
    MatrixXd centers_obj(1, n_features);
    centers_obj.row(0) = centers.row(0);
    MatrixXd closest_dist_sq = euclidean_distances(centers_obj, data, VectorXd(), x_squared_norms, true);

    double current_pot = 0.0;
    VectorXd cumulative_sum = VectorXd::Zero(closest_dist_sq.size());

    for (int i = 0; i < closest_dist_sq.size(); i++) {
        current_pot += closest_dist_sq(i) * sample_weight(i);
        cumulative_sum(i) = current_pot;
    }

    // Iterate over the remaining nclusters - 1 centers
    for (int c = 1; c < nclusters; c++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, current_pot);
        VectorXd rand_vals(n_local_trials);
        for (int i = 0; i < n_local_trials; ++i) {
            rand_vals(i) = dis(gen);
        }

        VectorXi candidate_ids(n_local_trials);
        for (int i = 0; i < n_local_trials; i++) {
            auto it = std::lower_bound(cumulative_sum.data(), cumulative_sum.data() + cumulative_sum.size(), rand_vals(i));
            candidate_ids(i) = std::distance(cumulative_sum.data(), it);
        }

        // Ensure candidate_ids are within the range of the closest_dist_sq size
        for (int i = 0; i < candidate_ids.size(); ++i) {
            candidate_ids(i) = std::min(static_cast<int>(closest_dist_sq.cols()) - 1, std::max(candidate_ids(i), 0));
        }

        MatrixXd candidates(candidate_ids.size(), data.cols());
        for (int i = 0; i < candidate_ids.size(); i++) {
            candidates.row(i) = data.row(candidate_ids(i));
        }

        MatrixXd distance_to_candidates = euclidean_distances(candidates, data, VectorXd(), x_squared_norms, true);

        for (int i = 0; i < closest_dist_sq.cols(); ++i) {
            for (int j = 0; j < distance_to_candidates.rows(); ++j) {
                if (closest_dist_sq(0, i) < distance_to_candidates(j, i)) {
                    distance_to_candidates(j, i) = closest_dist_sq(0, i);
                }
            }
        }

        VectorXd candidates_pot = VectorXd::Zero(distance_to_candidates.rows());
        for (int i = 0; i < distance_to_candidates.rows(); ++i) {
            candidates_pot(i) = distance_to_candidates.row(i).dot(sample_weight);
        }

        // Find the best candidate
        int best_candidate;
        double min_value = candidates_pot.minCoeff(&best_candidate);
        double current_pot = candidates_pot(best_candidate);
        closest_dist_sq.row(0) = distance_to_candidates.row(best_candidate);
        best_candidate = candidate_ids(best_candidate);

        centers.row(c) = data.row(best_candidate);
        indices(c) = best_candidate;
    }

    return std::make_pair(centers, indices);
}

KMeans::KMeans(int input_nclusters, 
               int input_n_init, 
               int input_max_iter, 
               double input_tol, 
               bool input_copy_x, 
               std::string input_algorithm) : 
nclusters(input_nclusters), 
n_init(input_n_init), 
max_iter(input_max_iter), 
tol(input_tol), 
copy_x(input_copy_x), 
algorithm(input_algorithm) {}

double KMeans::tolerance(MatrixXd& data) {
    if (tol == 0) {
        return 0.0;
    } else {
        VectorXd variances = calculate_variance(data);
        return tol * (variances.sum() / variances.size());
    }
}

void KMeans::check_params_vs_input(MatrixXd& data) {
    if (nclusters > data.rows()) {
        std::cout << "The number of samples should be greater than the number of clusters." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    double _tol = tolerance(data);

    if (n_init <= 0) {
        std::cout << "n_init should be greater than 0." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (max_iter <= 0) {
        std::cout << "max_iter should be greater than 0." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (algorithm != "elkan" && algorithm != "lloyd") {
        std::cout << "algorithm should be 'elkan' or 'lloyd'." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (algorithm == "elkan" && nclusters == 1) {
        std::cout << "algorithm='elkan' doesn't make sense for a single cluster. Using 'algorithm='lloyd' instead." << std::endl;
        algorithm = "lloyd";
    }
}

double KMeans::calculate_inertia(
    const MatrixXd& X, 
    const VectorXd& sample_weight, 
    const MatrixXd& centers, 
    const VectorXi& labels, 
    int n_threads,
    int single_label) {
    int n_samples = X.rows();

    double sq_dist = 0.0;
    double inertia = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        int j = labels(i);
        if (single_label < 0 || single_label == j) {
            sq_dist = (X.row(i) - centers.row(j)).squaredNorm();
            inertia += sq_dist * sample_weight(i);
        }
    }

    return inertia;
}

bool KMeans::is_same_clustering(
    const VectorXi& labels1, 
    const VectorXi& labels2, 
    int n_clusters) {

    VectorXi mapping = VectorXi::Constant(n_clusters, -1);

    for (int i = 0; i < labels1.size(); ++i) {
        if (mapping(labels1(i)) == -1) {
            mapping(labels1(i)) = labels2(i);
        } else if (mapping(labels1(i)) != labels2(i)) {
            return false;
        }
    }

    return true;
}


void KMeans::fit(
    MatrixXd& data, 
    VectorXd sample_weight) {
    
    check_params_vs_input(data);

    // Copy data to X if copy_x is true
    MatrixXd X;
    if (copy_x) {
        X = data;
    } else {
        X = data; // Eigen automatically shares data when assigning matrices
    }

    int n_samples = X.rows();
    int n_features = X.cols();

    // Check sample weights
    if (sample_weight.size() == 0) {
        sample_weight = VectorXd::Ones(n_samples);
    } else if (sample_weight.size() != n_samples) {
        std::cout << "The number of sample weights should be equal to the number of samples." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Subtract mean of x for more accurate distance comparisons
    VectorXd mean = X.colwise().mean();
    X.rowwise() -= mean.transpose();

    double best_inertia = std::numeric_limits<double>::max();
    MatrixXd best_centers;
    VectorXi best_labels;

    // Initialise centers
    for (int i = 0; i < n_init; i++){
        MatrixXd centers_init = KMeans_plus_plus(X, nclusters, sample_weight).run().first;
 
        VectorXi labels;
        double inertia;
        MatrixXd centers;

        // Run the KMeans algorithm
        if (algorithm == "elkan") {
            std::tie(labels, inertia, centers) = _kmeans_single_elkan(X, sample_weight, centers_init, max_iter, tol);
        } else {
            std::tie(labels, inertia, centers) =_kmeans_single_lloyd(X, sample_weight, centers_init, max_iter, tol);
        }
        
        if (best_labels.size() == 0) {
            best_labels = labels;
        }
        
        if ((inertia < best_inertia) && !(is_same_clustering(labels, best_labels, nclusters))) {
            best_inertia = inertia;
            best_labels = labels;
            best_centers = centers;
        }
    }

    // Add mean of x back to results
    X.rowwise() += mean.transpose();
    best_centers.rowwise() += mean.transpose();


    std::set<int> distinct_clusters(best_labels.data(), best_labels.data() + best_labels.size());
    if (distinct_clusters.size() < nclusters) {
        std::cerr << "Warning: Number of distinct clusters (" << distinct_clusters.size() 
                << ") found smaller than n_clusters (" << nclusters 
                << "). Possibly due to duplicate points in X.\n";
    }

    labels_ = best_labels;
    cluster_centers_ = best_centers;
    n_features_out_ = VectorXi(2);
    n_features_out_ << best_centers.rows(), best_centers.cols();
    inertia_ = best_inertia;
    return;
}

std::tuple<VectorXi, double, MatrixXd> KMeans::_kmeans_single_elkan(
    MatrixXd& data, 
    VectorXd& sample_weight, 
    MatrixXd centers_init, 
    int& max_iter, 
    double& tol){

    int n_samples = data.rows();
    int n_clusters = centers_init.rows();

    MatrixXd centers = centers_init;
    MatrixXd centers_new = MatrixXd::Zero(n_clusters, data.cols());
    VectorXd weight_in_clusters = VectorXd::Zero(n_clusters);
    VectorXi labels = VectorXi::Constant(n_samples, -1);
    VectorXi labels_old = labels;


    // Get euclidean distances and divide by 2
    MatrixXd center_half_distances = euclidean_distances(centers, centers, VectorXd(), VectorXd(), false) / 2;
    VectorXd distance_to_next_center = partition(center_half_distances, 1).row(1);

    VectorXd upper_bounds = VectorXd::Zero(n_samples);
    MatrixXd lower_bounds = MatrixXd::Zero(n_samples, n_clusters);
    VectorXd center_shift = VectorXd::Zero(n_clusters);


    // Initialise bounds
    init_bounds(data, centers, center_half_distances, labels, lower_bounds, upper_bounds);

    bool strict_convergence = false;

    // Perform elkan iteration
    for (int i = 0; i < max_iter; i++) {
        elkan_iter(data, 
                   sample_weight, 
                   centers, 
                   centers_new, 
                   weight_in_clusters, 
                   center_half_distances, 
                   distance_to_next_center, 
                   upper_bounds, 
                   lower_bounds, 
                   labels, 
                   center_shift, 
                   true);

        center_half_distances = euclidean_distances(centers_new, centers_new, VectorXd(), VectorXd(), true) / 2;
        distance_to_next_center = partition(center_half_distances, 1).row(1);

        centers = centers_new;

        if (labels.isApprox(labels_old)) {
            strict_convergence = true;
            break;
        } else {
            double center_shift_tot = center_shift.squaredNorm();

            if (center_shift_tot <= tol) {
                break;
            }
        }
        labels_old = labels;
    }

    if (!strict_convergence) {
        elkan_iter(data, 
                   sample_weight, 
                   centers, 
                   centers_new, 
                   weight_in_clusters, 
                   center_half_distances, 
                   distance_to_next_center, 
                   upper_bounds, 
                   lower_bounds, 
                   labels, 
                   center_shift, 
                   false);
    }


    double inertia = calculate_inertia(data, sample_weight, centers, labels, 1);

    return std::make_tuple(labels, inertia, centers);
}

std::tuple<VectorXi, double, MatrixXd> KMeans::_kmeans_single_lloyd(
    MatrixXd& data, 
    VectorXd& sample_weight, 
    MatrixXd centers_init, 
    int& max_iter, 
    double& tol){

    int n_samples = data.rows();
    int n_clusters = centers_init.rows();

    MatrixXd centers = centers_init;
    MatrixXd centers_new = MatrixXd::Zero(n_clusters, data.cols());
    VectorXi labels = VectorXi::Constant(n_samples, -1);
    VectorXi labels_old = labels;
    VectorXd weight_in_clusters = VectorXd::Zero(n_clusters);
    VectorXd center_shift = VectorXd::Zero(n_clusters);

    bool strict_convergence = false;

    for (int i = 0; i < max_iter; i++) {
        lloyd_iter(data, 
                   sample_weight, 
                   centers, 
                   centers_new, 
                   weight_in_clusters, 
                   labels, 
                   center_shift, 
                   true);

        centers = centers_new;

        if (labels.isApprox(labels_old)) {
            strict_convergence = true;
            break;
        } else {
            double center_shift_tot = center_shift.squaredNorm();

            if (center_shift_tot <= tol) {
                break;
            }
        }
        labels_old = labels;
    }

    if (!strict_convergence) {
        lloyd_iter(data, 
                   sample_weight, 
                   centers,
                   centers_new, 
                   weight_in_clusters, 
                   labels, 
                   center_shift, 
                   true);
    }

    double inertia = calculate_inertia(data, sample_weight, centers, labels, 1);

    return std::make_tuple(labels, inertia, centers);
}

