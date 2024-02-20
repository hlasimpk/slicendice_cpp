#include "kmeans.h"
#include "extmath.h"
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdlib>


typedef std::vector<std::vector< double > > Vector3D;

KMeans_plus_plus::KMeans_plus_plus(Vector3D& input_data, int input_nclusters, std::vector<double> input_sample_weight, std::vector<double> input_x_squared_norms, int input_seed, int input_n_local_trials) 
: data(input_data), nclusters(input_nclusters), sample_weight(input_sample_weight), x_squared_norms(input_x_squared_norms), seed(input_seed), n_local_trials(input_n_local_trials) {}

int KMeans_plus_plus::get_random_center() {
    std::vector<double> weights;
    double sum_weights = 0.0;
    for (double weight : sample_weight) { 
        sum_weights += weight;
    }
    for (double weight : sample_weight) {
        weights.push_back(weight / sum_weights);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    if (seed != 0) {
        gen.seed(seed);
    }
    gen.seed(seed);
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return dist(gen);
}

std::pair<Vector3D, std::vector<int>> KMeans_plus_plus::run() {
    int n_samples = data.size();
    int n_features = data[0].size();

    Vector3D centers(nclusters, std::vector<double>(n_features));

    // Set the number of local seeding trials if 0 is given
    if (n_local_trials == 0) {
        n_local_trials = 2 + int(std::log(nclusters));
    }

    // Set the sample weights if none are given
    if (sample_weight.size() == 0) {
        for (int i = 0; i < data.size(); i++) {
            sample_weight.push_back(1.0);
        }
    }

    // Set the random state
    int center_id = get_random_center();
    // int center_id = 46; // hard code the center_id for testing

    // Set indices to -1
    std::vector<int> indices(nclusters, -1);

    centers[0] = data[center_id];
    indices[0] = center_id;

    // Initialise list of closest distances and calculate the current potential
    Vector3D centers_obj(1, std::vector<double>(n_features));
    centers_obj[0] = centers[0];
    Vector3D closest_dist_sq = euclidean_distances(centers_obj, data, std::vector<double>(), x_squared_norms, true);


    double current_pot = 0.0;
    std::vector<double> cumulative_sum(closest_dist_sq[0].size());

    for (int i = 0; i < closest_dist_sq.size(); i++) {
        for (int j = 0; j < closest_dist_sq[i].size(); j++) {
            current_pot += closest_dist_sq[i][j] * sample_weight[i];
            cumulative_sum[j] = current_pot;
        }
    }

    // Iterate over the remaining nclusters - 1 centers
    for (int c = 1; c < nclusters; c++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, current_pot);
        std::vector<double> rand_vals(n_local_trials);
        for (auto& val : rand_vals) {
            val = dis(gen);
        }
        // hard code the rand_vals for testing
        // std::vector<double> rand_vals = {16265.23297954, 19865.80160536};

        std::vector<int> candidate_ids(n_local_trials);
        for (int i = 0; i < n_local_trials; i++) {
            auto it = std::lower_bound(cumulative_sum.begin(), cumulative_sum.end(), rand_vals[i]);
            candidate_ids[i] = std::distance(cumulative_sum.begin(), it);
        }

        // Ensure candidate_ids are within the range of the closest_dist_sq size
        for (auto& id : candidate_ids) {
            id = std::min(static_cast<int>(closest_dist_sq[0].size()) - 1, std::max(id, 0));
        }

        Vector3D candidates;
        for (int i = 0; i < candidate_ids.size(); i++) {
            candidates.push_back(data[candidate_ids[i]]);
        }

        Vector3D distance_to_candidates = euclidean_distances(candidates, data, std::vector<double>(), x_squared_norms, true);

        for (size_t i = 0; i < closest_dist_sq[0].size(); ++i) {
            for (size_t j = 0; j < distance_to_candidates.size(); ++j) {
                if (closest_dist_sq[0][i] < distance_to_candidates[j][i]) {
                    distance_to_candidates[j][i] = closest_dist_sq[0][i];
                }
            }
        }

        std::vector<double> candidates_pot = {0, 0};
        for (int i = 0; i < distance_to_candidates.size(); i++) {
            for (int j = 0; j < distance_to_candidates[i].size(); j++) {
                candidates_pot[i] += distance_to_candidates[i][j] * sample_weight[i];
            }
        }

        // Find the best candidate
        auto min_element_it = std::min_element(candidates_pot.begin(), candidates_pot.end());
        int best_candidate = std::distance(candidates_pot.begin(), min_element_it);
        double current_pot = candidates_pot[best_candidate];
        closest_dist_sq[0] = distance_to_candidates[best_candidate];
        best_candidate = candidate_ids[best_candidate];

        centers[c] = data[best_candidate];
        indices[c] = best_candidate;
    }

    return std::make_pair(centers, indices);
}

KMeans::KMeans(int input_nclusters, int input_n_init, int input_max_iter, double input_tol, bool input_copy_x, std::string input_algorithm) : 
nclusters(input_nclusters), n_init(input_n_init), max_iter(input_max_iter), tol(input_tol), copy_x(input_copy_x), algorithm(input_algorithm) {}

double KMeans::tolerance(Vector3D& data) {
    if (tol == 0) {
        return 0.0;
    } else {
        std::vector<double> variances = calculate_variance(data);
        return tol * (std::accumulate(variances.begin(), variances.end(), 0.0) / variances.size());
    }
}

void KMeans::check_params_vs_input(Vector3D& data) {
    if (nclusters > data.size()) {
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

Vector3D KMeans::fit(Vector3D& data, std::vector<double> sample_weight) {
    check_params_vs_input(data);

    // Copy data to X if copy_x is true
    Vector3D X;
    if (copy_x) {
        X = data;
    } else {
        X = std::ref(data);
    }

    int n_samples = X.size();
    int n_features = X[0].size();

    // Check sample weights
    if (sample_weight.size() == 0) {
        for (int i = 0; i < data.size(); i++) {
            sample_weight.push_back(1.0);
        }
    } else if (sample_weight.size() != n_samples) {
        std::cout << "The number of sample weights should be equal to the number of samples." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Subtract mean of x for more accurate distance comparisons
    std::vector<double> mean = calculate_mean(X);
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++) {
            X[i][j] -= mean[j];
        }
    }
    
    Vector3D best_inertia;
    Vector3D best_labels;

    // Initialise centers
    for (int i = 0; i < n_init; i++){
        Vector3D centers_init = KMeans_plus_plus(X, nclusters, sample_weight).run().first;
        std::cout << i << " here " << n_init << std::endl;

        // Run the KMeans algorithm
        // if (algorithm == "elkan") {
        //     Vector3D labels = _kmeans_single_elkan(X, sample_weight, centers_init, max_iter, tol);
        // } else if (algorithm == "lloyd") {
        //     Vector3D labels = _kmeans_single_lloyd(X, sample_weight, centers_init, max_iter, tol);
        // } else {
        //     std::cout << "algorithm should be 'elkan' or 'lloyd'." << std::endl;
        //     std::exit(EXIT_FAILURE);
        // }

    }

    std::cout << "loop over " << std::endl;

    return best_labels;

}

Vector3D KMeans::_kmeans_single_elkan(Vector3D& data, std::vector<double> sample_weight, Vector3D centers_init, int max_iter, double tol){
    int n_samples = data.size();
    int n_clusters = centers_init.size();

    Vector3D centers = centers_init;
    Vector3D centers_new(n_clusters, std::vector<double>(centers_init[0].size()));
    std::vector<double> weight_in_clusters(n_clusters);
    std::vector<double> labels(n_samples, -1);
    std::vector<double> labels_old = labels;
    Vector3D center_distances = euclidean_distances(centers, centers, std::vector<double>(), std::vector<double>(), false);

    std::cout << "center_distances: " << std::endl;
    for (int i = 0; i < center_distances.size(); i++) {
        for (int j = 0; j < center_distances[i].size(); j++) {
            std::cout << center_distances[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return data;
}

Vector3D KMeans::_kmeans_single_lloyd(Vector3D& data, std::vector<double> sample_weight, Vector3D centers_init, int max_iter, double tol){

    return data;
}

