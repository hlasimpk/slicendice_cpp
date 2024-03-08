#include "agglomerative.h"
#include "dist_metrics.h"
#include "extmath.h"
#include "hierarchy.h"

#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>

using Eigen::MatrixXi;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::VectorXi;

Agglomerative::Agglomerative(
    int input_nclusters, 
    std::string input_metric, 
    std::unordered_map<MatrixXd, std::tuple<MatrixXd, int, int, VectorXi, MatrixXd>, MatrixHash> input_memory, 
    SparseMatrix<double> input_connectivity, 
    std::string input_compute_full_tree, 
    std::string input_linkage, 
    double input_distance_threshold, 
    bool input_compute_distance) : 
nclusters(input_nclusters), 
metric(input_metric),
connectivity(input_connectivity),
compute_full_tree(input_compute_full_tree),
linkage(input_linkage),
distance_threshold(input_distance_threshold),
compute_distance(input_compute_distance) {}

void Agglomerative::check_params(){

    if (!((nclusters == -1) != (distance_threshold == -1.0))) {
        throw std::invalid_argument(
            "Exactly one of n_clusters and "
            "distance_threshold has to be set, and the other "
            "needs to be None."
        );
    }
    if (distance_threshold != -1.0 && compute_full_tree != "true") {
        throw std::invalid_argument(
            "compute_full_tree must be True if distance_threshold is set."
        );
    }

    if (linkage == "ward" && metric != "euclidean") {
        throw std::invalid_argument(
            metric + " was provided as metric. Ward can only "
            "work with euclidean distances."
        );
    }

}

void Agglomerative::fit(MatrixXd& data) {

    check_params();

    int n_samples = data.rows();

    bool compute_full_tree_bool = false;
    if (connectivity.size() == 0) {
        compute_full_tree_bool = true;
    }
    if (compute_full_tree == "auto") {
        if (distance_threshold != -1.0)
            compute_full_tree_bool = true;
        else
            compute_full_tree_bool = nclusters < std::max(100, static_cast<int>(0.02 * n_samples));
    }

    int nclusters_alg = nclusters;
    if (compute_full_tree_bool) {
        nclusters_alg = -1;
    }

    double distance_threshold_alg = distance_threshold;

    bool return_distance = ((distance_threshold_alg != -1.0) || compute_distance);

    MatrixXd children;
    int n_connected_components;
    int n_leaves;
    VectorXi parents;
    MatrixXd distances;

    if (linkage == "ward") {
        std::tie(children, n_connected_components, n_leaves, parents, distances) = ward_tree(
            data, connectivity, nclusters_alg, return_distance
        );
    } else if (linkage == "complete" || linkage == "average" || linkage == "single") {
         std::tie(children, n_connected_components, n_leaves, parents, distances) = linkage_tree(
            data, connectivity, nclusters_alg, linkage, metric, return_distance);
    } else {
        throw std::invalid_argument(
            "Unknown linkage type " + linkage + ". "
            "Valid options are 'ward', 'complete', 'average', and 'single'."
        );
    }

    if (distance_threshold != -1.0) {
        nclusters = (distances.array() >= distance_threshold).count() + 1;
    }

    if (compute_full_tree_bool) {
        labels_ = hierarchy::hc_cut(nclusters, children, n_leaves);
    } else {
        VectorXi labels = hierarchy::hc_get_heads(parents);
        VectorXi unique_labels = unique(labels.head(n_samples));

        VectorXi indices(labels_.size());
        for (int i = 0; i < labels_.size(); ++i) {
            // std::lower_bound returns an iterator pointing to the first element that is not less than label
            // std::distance computes the number of elements between two iterators
            indices(i) = std::distance(unique_labels.data(), std::lower_bound(unique_labels.data(), unique_labels.data() + unique_labels.size(), labels_(i)));
        }
    }

    return;
}

int Agglomerative::connected_components_directed(
        const int* indices,
        const int* indptr,
        VectorXi& labels) {
    const int VOID = -1;
    const int END = -2;
    int N = labels.size();
    VectorXi SS = VectorXi::Constant(N, VOID), lowlinks = labels, stack_f = SS, stack_b = VectorXi::Constant(N, VOID);
    int SS_head = END, stack_head = END, index = 0, label = N - 1;

        for (int v = 0; v < N; ++v) {
        if (lowlinks[v] == VOID) {
            stack_head = v;
            stack_f[v] = END;
            stack_b[v] = END;
            while (stack_head != END) {
                v = stack_head;
                if (lowlinks[v] == VOID) {
                    lowlinks[v] = index;
                    index += 1;
                    for (int j = indptr[v]; j < indptr[v+1]; ++j) {
                        int w = indices[j];
                        if (lowlinks[w] == VOID) {
                            if (stack_f[w] != VOID) {
                                int f = stack_f[w];
                                int b = stack_b[w];
                                if (b != END) stack_f[b] = f;
                                if (f != END) stack_b[f] = b;
                            }
                            stack_f[w] = stack_head;
                            stack_b[w] = END;
                            stack_b[stack_head] = w;
                            stack_head = w;
                        }
                    }
                } else {
                    stack_head = stack_f[v];
                    if (stack_head >= 0) stack_b[stack_head] = END;
                    stack_f[v] = VOID;
                    stack_b[v] = VOID;
                    bool root = true;
                    int low_v = lowlinks[v];
                    for (int j = indptr[v]; j < indptr[v+1]; ++j) {
                        int low_w = lowlinks[indices[j]];
                        if (low_w < low_v) {
                            low_v = low_w;
                            root = false;
                        }
                    }
                    lowlinks[v] = low_v;
                    if (root) {
                        index -= 1;
                        while (SS_head != END && lowlinks[v] <= lowlinks[SS_head]) {
                            int w = SS_head;
                            SS_head = SS[w];
                            SS[w] = VOID;
                            labels[w] = label;
                            index -= 1;
                        }
                        labels[v] = label;
                        label -= 1;
                    } else {
                        SS[v] = SS_head;
                        SS_head = v;
                    }
                }
            }
        }
    }

    labels = -1 * labels;
    labels.array() += (N - 1);
    return (N - 1) - label;
}


int Agglomerative::connected_components_undirected(
    const int* indices1,
    const int* indptr1,
    const int* indices2,
    const int* indptr2,
    VectorXi& labels) {

    const int VOID = -1;
    const int END = -2;
    int N = labels.size();
    labels = VectorXi::Constant(N, VOID);
    int label = 0;
    VectorXi SS = labels;
    int SS_head = END;

    for (int v = 0; v < N; ++v) {
        if (labels[v] == VOID) {
            SS_head = v;
            SS[v] = END;

            while (SS_head != END) {
                v = SS_head;
                SS_head = SS[v];

                labels[v] = label;

                for (int j = indptr1[v]; j < indptr1[v+1]; ++j) {
                    int w = indices1[j];
                    if (SS[w] == VOID) {
                        SS[w] = SS_head;
                        SS_head = w;
                    }
                }
                for (int j = indptr2[v]; j < indptr2[v+1]; ++j) {
                    int w = indices2[j];
                    if (SS[w] == VOID) {
                        SS[w] = SS_head;
                        SS_head = w;
                    }
                }
            }
            label += 1;
        }
    }

    return label;
}


std::pair<int, VectorXi> Agglomerative::connected_components(
    const SparseMatrix<double>& connectivity, 
    bool directed, 
    std::string connection, 
    bool return_labels) {

    if (connection != "weak" && connection != "strong") {
        throw std::invalid_argument("connection must be 'weak' or 'strong'");
    }

    if (connection == "weak") {
        directed = false;
    }

    VectorXi labels(connectivity.rows());
    labels.setConstant(-1);

    int n_components;
    if (directed) {
        n_components = connected_components_directed(connectivity.innerIndexPtr(), connectivity.outerIndexPtr(), labels);
    } else {
        SparseMatrix<double> connectivity_T = connectivity.transpose();
        n_components = connected_components_undirected(connectivity.innerIndexPtr(), connectivity.outerIndexPtr(), connectivity.innerIndexPtr(), connectivity.outerIndexPtr(), labels);
    }

    if (return_labels) {
        return std::make_pair(n_components, labels);
    } else {
        return std::make_pair(n_components, VectorXi());
    }
}


SparseMatrix<double> Agglomerative::fix_connected_components(
    MatrixXd& X, 
    SparseMatrix<double>& connectivity, 
    int n_connected_components, 
    VectorXi& labels, 
    std::string affinity, 
    std::string mode) {

    if (metric == "precomputed" && connectivity.nonZeros() > 0) {
        throw std::runtime_error(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        );
    }

    for (int i = 0; i < n_connected_components; ++i) {
        std::vector<int> idx_i;
        for (int k = 0; k < labels.size(); ++k) {
            if (labels[k] == i) {
                idx_i.push_back(k);
            }
        }
        MatrixXd Xi = X.block(idx_i[0], 0, idx_i.size(), X.cols());

        for (int j = 0; j < i; ++j) {
            std::vector<int> idx_j;
            for (int k = 0; k < labels.size(); ++k) {
                if (labels[k] == j) {
                    idx_j.push_back(k);
                }
            }

            MatrixXd Xj = X.block(idx_j[0], 0, idx_j.size(), X.cols());


            MatrixXd D;
            if (metric == "precomputed") {
                D.resize(idx_i.size(), idx_j.size());
                for (int m = 0; m < idx_i.size(); ++m) {
                    for (int n = 0; n < idx_j.size(); ++n) {
                        D(m, n) = X(idx_i[m], idx_j[n]);
                    }
                }
            } else {
                MatrixXd ret(Xi.rows(), Xj.rows());
                if (Xi == Xj || Xj.size() == 0) {
                    // zeroing diagonal for euclidean norm.
                    for (int i = 0; i < ret.rows(); ++i) {
                        ret(i, i) = 0;
                    }
                }
                // Can just use euclidean_distances directly as other metrics are not implemented
                D = euclidean_distances(Xi, Xj, VectorXd(), VectorXd(), false);
            }

            int ii, jj;
            double min_val = D.minCoeff(&ii, &jj);
            if (mode == "connectivity") {
                connectivity.coeffRef(idx_i[ii], idx_j[jj]) = 1;
                connectivity.coeffRef(idx_j[jj], idx_i[ii]) = 1;
            } else if (mode == "distance") {
                connectivity.coeffRef(idx_i[ii], idx_j[jj]) = D(ii, jj);
                connectivity.coeffRef(idx_j[jj], idx_i[ii]) = D(ii, jj);
            } else {
                throw std::invalid_argument(
                    "Unknown mode=" + mode + ", should be one of ['connectivity', 'distance']."
                );
            }
        }
    }

    return connectivity;
}

std::pair<SparseMatrix<double>, int> Agglomerative::fix_connectivity(MatrixXd& X, SparseMatrix<double>& connectivity, std::string affinity) {
    int n_samples = X.rows();
    if (connectivity.rows() != n_samples || connectivity.cols() != n_samples) {
        std::cerr << "Wrong shape for connectivity matrix: " << connectivity.rows() << "x" << connectivity.cols()
                  << " when X is " << n_samples << "x" << X.cols() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Make the connectivity matrix symmetric:
    MatrixXd dense_connectivity = connectivity.toDense();
    MatrixXd symmetric_connectivity_dense = dense_connectivity + dense_connectivity.transpose();
    SparseMatrix<double> symmetric_connectivity = symmetric_connectivity_dense.sparseView();
    connectivity.swap(symmetric_connectivity);

    // Compute the number of nodes
    int n_connected_components;
    VectorXi labels;
    std::tie(n_connected_components, labels) = connected_components(connectivity);

    if (n_connected_components > 1) {
        std::cerr << "the number of connected components of the "
                  << "connectivity matrix is " << n_connected_components << " > 1. Completing it to avoid "
                  << "stopping the tree early." << std::endl;

        connectivity = fix_connected_components(X, connectivity, n_connected_components, labels, affinity, "connectivity");
    }

    return std::make_pair(connectivity, n_connected_components);
}




std::tuple<MatrixXd, int, int, VectorXi, MatrixXd> Agglomerative::ward_tree(
    MatrixXd X, 
    SparseMatrix<double>& connectivity, 
    int n_clusters, 
    bool return_distance) {

    
    // Check if the input has been seen before and return the result if it has
    MatrixXd input_identifier;
    if (connectivity.size() == 0) {
        input_identifier = X.array() * n_clusters;
    } else if (connectivity.rows() == X.rows()) {
        input_identifier = X.rowwise().mean().array().cwiseProduct(connectivity.toDense().rowwise().mean().array()) * n_clusters;
    } else {
        throw std::invalid_argument(
            "Connectivity matrix must have the same number of columns as the input matrix."
        );
    }

    input_identifier *= return_distance ? 1 : -1;
    auto it = memory.find(input_identifier);
    if (it != memory.end()) {
        return it->second;
    }

    std::tuple<MatrixXd, int, int, VectorXi, MatrixXd> result;

    if (X.cols() == 1) {
        X.resize(X.size(), 1);
    }
    int n_samples = X.rows();
    int n_features = X.cols();

    if (connectivity.size() == 0) {
        if (n_clusters > 0) {
            std::cerr << "Partial build of the tree is implemented "
                      << "only for structured clustering (i.e. with "
                      << "explicit connectivity). The algorithm "
                      << "will build the full tree and only "
                      << "retain the lower branches required "
                      << "for the specified number of clusters" << std::endl;
        }

        X = X.eval();

        MatrixXd out = hierarchy::ward(X);

        MatrixXd children_ = out.block(0, 0, out.rows(), 2).cast<double>();

        if (return_distance) {
            result = std::make_tuple(children_, 1, n_samples, VectorXi(), out.col(2));
        } else {
            result = std::make_tuple(children_, 1, n_samples, VectorXi(), MatrixXd());
        }        

    } else {
        int n_connected_components;
        std::tie(connectivity, n_connected_components) = fix_connectivity(X, connectivity, "euclidean");

        int n_nodes;
        if (n_clusters <= 0 ) {
            n_nodes = 2 * n_samples - 1;
        } else {
            if (n_clusters > n_samples) {
                std::cerr << "Cannot provide more clusters than samples. "
                        << n_clusters << " n_clusters was asked, and there are "
                        << n_samples << " samples." << std::endl;
                exit(EXIT_FAILURE);
            }
            n_nodes = 2 * n_samples - n_clusters;
        }

        VectorXi coord_row;
        VectorXi coord_col;
        std::vector<Eigen::VectorXi> A;

        for (int ind = 0; ind < connectivity.outerSize(); ++ind) {
            std::vector<int> temp_row;
            for (SparseMatrix<double>::InnerIterator it(connectivity, ind); it; ++it) {
                temp_row.push_back(it.row());
            }
            // Convert temp_row to Eigen::VectorXi and add it to A
            Eigen::VectorXi Ai = Eigen::Map<Eigen::VectorXi>(temp_row.data(), temp_row.size());
            A.push_back(Ai);

            // Filter temp_row to keep only the indices less than ind
            temp_row.erase(std::remove_if(temp_row.begin(), temp_row.end(), [ind](int i) { return i >= ind; }), temp_row.end());

            // Extend coord_row and coord_col
            coord_row.conservativeResize(coord_row.size() + temp_row.size());
            coord_row.tail(temp_row.size()).setConstant(ind);
            coord_col.conservativeResize(coord_col.size() + temp_row.size());
            for (int i = 0; i < temp_row.size(); ++i) {
                coord_col(coord_col.size() - temp_row.size() + i) = temp_row[i];
            }
        }

        VectorXd moments_1 = VectorXd::Zero(n_nodes);
        moments_1.head(n_samples).setOnes();

        MatrixXd moments_2 = MatrixXd::Zero(n_nodes, n_features);
        moments_2.block(0, 0, n_samples, n_features) = X;

        VectorXd inertia(coord_row.size());

        // Assuming compute_ward_dist is a function that modifies inertia
        hierarchy::compute_ward_dist(moments_1, moments_2, coord_row, coord_col, inertia);

        // Create a priority queue (heap) of tuples
        auto comp = [](const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) {
            return std::get<0>(a) > std::get<0>(b);
        };
        std::priority_queue<std::tuple<double, int, int>, std::vector<std::tuple<double, int, int>>, decltype(comp)> heap(comp);

        for (int i = 0; i < inertia.size(); ++i) {
            heap.push(std::make_tuple(inertia[i], coord_row[i], coord_col[i]));
        }

        VectorXi parent = VectorXi::LinSpaced(n_nodes, 0, n_nodes-1);
        VectorXi used_node = VectorXi::Ones(n_nodes);
        MatrixXd children;

        VectorXd distances;
        if (return_distance) {
            distances = VectorXd::Zero(n_nodes - n_samples);
        }

        VectorXi not_visited = VectorXi::Zero(n_nodes);

        // Recursive merge loop
        for (int k = n_samples; k < n_nodes; ++k) {
            // Identify the merge
            int i = 0; 
            int j = 0;
            double inert = 0.0;
            while (!heap.empty()) {
                auto [heap_inert, heap_i, heap_j] = heap.top();
                i = heap_i;
                j = heap_j;
                inert = heap_inert;
                heap.pop();
                if (used_node[i] == 1 && used_node[j] == 1) {
                    break;
                }
            }

            parent[i] = parent[j] = k;
            int currentSize = children.rows();
            children.conservativeResize(currentSize + 1, 2);
            children.row(currentSize) << i, j;
            used_node[i] = used_node[j] = false;
            if (return_distance) {  // Store inertia value
                distances[k - n_samples] = inert;
            }

            // Update the moments
            moments_1[k] = moments_1[i] + moments_1[j];
            moments_2.row(k) = moments_2.row(i) + moments_2.row(j);

            // Update the structure matrix A and the inertia matrix
            VectorXi coord_col;
            not_visited.setOnes();
            not_visited[k] = 0;
            hierarchy::get_parents(A[i], coord_col, parent, not_visited);
            hierarchy::get_parents(A[j], coord_col, parent, not_visited);
            for (int idx = 0; idx < coord_col.size(); ++idx) {
                int col = coord_col[idx];
                A[col].conservativeResize(A[col].size() + 1);
                A[col](A[col].size() - 1) = k;
            }

            A.push_back(coord_col);
            Eigen::VectorXi coord_row(coord_col.size());
            coord_row.fill(k);
            int n_additions = coord_row.size();
            Eigen::VectorXd ini(n_additions);

            hierarchy::compute_ward_dist(moments_1, moments_2, coord_row, coord_col, ini);

            for (int idx = 0; idx < n_additions; ++idx) {
                heap.push({ini[idx], k, coord_col[idx]});
            }
        }

        // Separate leaves in children (empty lists up to now)
        int n_leaves = n_samples;
        
        for (int i = 0; i < children.rows(); ++i) {
            children.row(i).reverseInPlace();
        }


        if (return_distance) {
            // 2 is scaling factor to compare w/ unstructured version
            distances = (2.0 * distances.array()).sqrt();
            result = std::make_tuple(children, 1, n_samples, VectorXi(), distances);
        } else {
            result = std::make_tuple(children, 1, n_samples, VectorXi(), MatrixXd());
        }  
    }

    // Store the result in the memory
    memory[input_identifier] = result;

    return result;
}

std::tuple<MatrixXd, int, int, VectorXi, MatrixXd> Agglomerative::linkage_tree(
    MatrixXd X, 
    SparseMatrix<double>& connectivity, 
    int n_clusters,
    std::string linkage,
    std::string affinity, 
    bool return_distance) {

    if (X.cols() == 1) {
        X.resize(X.size(), 1);
    }
    int n_samples = X.rows();
    int n_features = X.cols();

    if (affinity == "cosine" && (X.array().rowwise().sum() == 0).any()) {
        throw std::invalid_argument("Cosine affinity cannot be used when X contains zero vectors");
    }

    if (connectivity.nonZeros() == 0) {
        if (n_clusters > 0) {
            std::cerr << "Partial build of the tree is implemented "
                      << "only for structured clustering (i.e. with "
                      << "explicit connectivity). The algorithm "
                      << "will build the full tree and only "
                      << "retain the lower branches required "
                      << "for the specified number of clusters" << std::endl;
        }

        if (affinity == "precomputed") {
            if (X.rows() != X.cols()) {
                throw std::invalid_argument("Distance matrix should be square, got matrix of shape " + std::to_string(X.rows()) + "x" + std::to_string(X.cols()));
            }
            auto indices = triu_indices(X.rows());
            VectorXi i = indices.first;
            VectorXi j = indices.second;

            MatrixXd X_flattened(i.size(), 1);
            for (int k = 0; k < i.size(); ++k) {
                X_flattened(k, 0) = X(i(k), j(k));
            }
            X = X_flattened;
        } else if (affinity == "l2") {
            affinity = "euclidean";
        } else if (affinity == "l1" || affinity == "manhattan") {
            affinity = "cityblock";
        }

        MatrixXd out;
        if (linkage == "single" && affinity != "precomputed") {
            MatrixXd mst;
            DistanceMetric* metric = DistanceMetricFactory::getMetric(affinity);
            mst = hierarchy::mst_linkage_core(X, metric);
            delete metric;

            int column = 2; // The column index to sort by
            // Create a vector of indices
            std::vector<int> indices(mst.rows());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort the indices based on the specified column in mst
            std::sort(indices.begin(), indices.end(),
                [&mst, &column](int i1, int i2) {
                    return mst(i1, column) < mst(i2, column);
                });

            // Create a new matrix to hold the sorted rows
            MatrixXd sorted_mst(mst.rows(), mst.cols());

            // Fill the new matrix with the sorted rows
            for (int i = 0; i < indices.size(); i++) {
                sorted_mst.row(i) = mst.row(indices[i]);
            }

            out = hierarchy::single_linkage_label(sorted_mst);
        } else {
            out = hierarchy::linkage(X, linkage);
        }

        MatrixXd children_ = out.block(0, 0, out.rows(), 2).cast<double>();

        if (return_distance) {
            return std::make_tuple(children_, 1, n_samples, VectorXi(), out.col(2));
        } else {
            return std::make_tuple(children_, 1, n_samples, VectorXi(), MatrixXd());
        }      
    } else {
        int n_connected_components;
        std::tie(connectivity, n_connected_components) = fix_connectivity(X, connectivity, affinity);

        std::vector<Eigen::Triplet<double>> tripletList;
        for (int k=0; k<connectivity.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(connectivity,k); it; ++it) {
                tripletList.push_back(Eigen::Triplet<double>(it.row(),it.col(),it.value()));
            }
        }

        tripletList.erase(std::remove_if(tripletList.begin(), tripletList.end(),
            [](const Eigen::Triplet<double>& triplet) {
                return triplet.row() == triplet.col();
            }), tripletList.end());

        SparseMatrix<double> newConnectivity(connectivity.rows(), connectivity.cols());
        newConnectivity.setFromTriplets(tripletList.begin(), tripletList.end());
        connectivity = newConnectivity;

        VectorXd distances;
        Eigen::VectorXi row_indices(connectivity.nonZeros());
        Eigen::VectorXi col_indices(connectivity.nonZeros());
        int index = 0;

        for (int k=0; k<connectivity.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(connectivity,k); it; ++it) {
                col_indices(index) = it.col();
                row_indices(index) = it.row();
                index++;
            }
        }

        if (affinity == "precomputed") {
            MatrixXd distances(row_indices.size(), 1);

            for (int i = 0; i < row_indices.size(); ++i) {
                distances(i, 0) = X(row_indices(i), col_indices(i));
            }
        } else {
            MatrixXd X_rows(row_indices.size(), X.cols());
            MatrixXd X_cols(col_indices.size(), X.cols());
            for (int i = 0; i < row_indices.size(); ++i) {
                X_rows.row(i) = X.row(row_indices(i));
            }
            for (int i = 0; i < col_indices.size(); ++i) {
                X_cols.row(i) = X.row(col_indices(i));
            }
            distances = paired_distances(X_cols, X_rows, affinity);
        }

        tripletList.clear();
        int idx = 0;
        for (int k=0; k<connectivity.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(connectivity,k); it; ++it) {
                double new_value = distances(idx);
                tripletList.push_back(Eigen::Triplet<double>(it.row(), it.col(), new_value));
                idx++;
            }
        }
        connectivity.setFromTriplets(tripletList.begin(), tripletList.end());

        int n_nodes;
        if (n_clusters <= 0 ) {
            n_nodes = 2 * n_samples - 1;
        } else {
            if (n_clusters > n_samples) {
                std::cerr << "Cannot provide more clusters than samples. "
                        << n_clusters << " n_clusters was asked, and there are "
                        << n_samples << " samples." << std::endl;
                exit(EXIT_FAILURE);
            }
            n_nodes = 2 * n_samples - n_clusters;
        }

        if (linkage == "single") {
            // TODO:: Implement something similar to single_linkage_tree at a later date
            std::cerr << "Partial build of the tree is not implemented for single linkage" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (return_distance) {
            distances = Eigen::VectorXd::Zero(n_nodes - n_samples);
        }

        // Convert connectivity to a list of lists for efficient row access
        std::vector<std::list<int>> connectivityList(n_nodes);
        for (int k = 0; k < connectivity.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(connectivity, k); it; ++it) {
                connectivityList[it.row()].push_back(it.col());
            }
        }

        std::vector<std::map<int, double>> A(n_nodes);
        // std::priority_queue<hierarchy::WeightedEdge> inertia;
        auto compare = [](const hierarchy::WeightedEdge& a, const hierarchy::WeightedEdge& b) {
            return a.weight > b.weight;
        };
        std::priority_queue<hierarchy::WeightedEdge, std::vector<hierarchy::WeightedEdge>, decltype(compare)> inertia(compare);

        for (int k = 0; k < connectivity.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(connectivity, k); it; ++it) {
                A[it.row()][it.col()] = it.value();
                if (it.col() < it.row()) {
                    inertia.push(hierarchy::WeightedEdge(it.value(), it.row(), it.col()));
                }
            }
        }

        connectivityList.clear();

        std::vector<int> parent(n_nodes);
        std::iota(parent.begin(), parent.end(), 0);
        std::vector<int> used_node(n_nodes, 1);
        std::vector<std::pair<int, int>> children;


        // Recursive merge loop
        for (int k = n_samples; k < n_nodes; ++k) {
            // Identify the merge
            double weight = 0.0;
            int a = 0;
            int b = 0;
            hierarchy::WeightedEdge edge(weight, a, b);
            while (true) {
                edge = inertia.top();
                inertia.pop();
                if (used_node[edge.a] && used_node[edge.b]) {
                    break;
                }
            }

            int i = edge.a;
            int j = edge.b;

            if (return_distance) {
                // Store distances
                distances[k - n_samples] = edge.weight;
            }

            parent[i] = parent[j] = k;
            children.push_back(std::make_pair(i, j));

            // Keep track of the number of elements per cluster
            int n_i = used_node[i];
            int n_j = used_node[j];
            used_node[k] = n_i + n_j;
            used_node[i] = used_node[j] = 0;


            // Update the structure matrix A and the inertia matrix
            // A clever 'min', or 'max' operation between A[i] and A[j]
            std::map<int, double> coord_col;
            if (linkage == "complete") {
                coord_col = hierarchy::max_merge(A[i], A[j], used_node, n_i, n_j);
            } else if (linkage == "average") {
                coord_col = hierarchy::average_merge(A[i], A[j], used_node, n_i, n_j);
            } else {
                throw std::invalid_argument("Unknown linkage type " + linkage + ". Valid options are 'complete' and 'average'.");
            }

            for (const auto& pair : coord_col) {
                int col = pair.first;
                double d = pair.second;
                A[col][k] = d;
                // Here we use the information from coord_col (containing the
                // distances) to update the heap
                inertia.push(hierarchy::WeightedEdge(d, k, col));
            }
            A[k] = coord_col;
            // Clear A[i] and A[j] to save memory
            A[i].clear();
            A[j].clear();
        }

        // Separate leaves in children (empty lists up to now)
        int n_leaves = n_samples;

        // Convert children to a 2D array and reverse each pair
        Eigen::MatrixXd children_matrix(children.size(), 2);
        for (size_t i = 0; i < children.size(); ++i) {
            children_matrix(i, 0) = children[i].second;
            children_matrix(i, 1) = children[i].first;
        }

        // Get output in right format
        MatrixXd distances_matrix = distances;
        VectorXi parent_vector = Eigen::Map<VectorXi>(parent.data(), parent.size());

        if (return_distance) {
            return std::make_tuple(children_matrix, n_connected_components, n_leaves, parent_vector, distances_matrix);
        }
        return std::make_tuple(children_matrix, n_connected_components, n_leaves, parent_vector, MatrixXd());
    }
}