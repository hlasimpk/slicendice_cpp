
#include "extmath.h"
#include "dist_metrics.h"
#include "hierarchy.h"

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace hierarchy {
    std::set <std::string> linkage_methods_set = {
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward"
    };


    MatrixXd single(MatrixXd& X) {
        return linkage(X, "single");
    }

    MatrixXd complete(MatrixXd& X) {
        return linkage(X, "complete");
    }

    MatrixXd average(MatrixXd& X) {
        return linkage(X, "average");
    }

    MatrixXd weighted(MatrixXd& X) {
        return linkage(X, "weighted");
    }

    MatrixXd centroid(MatrixXd& X) {
        return linkage(X, "centroid");
    }

    MatrixXd median(MatrixXd& X) {
        return linkage(X, "median");
    }

    MatrixXd ward(MatrixXd& X) {
        return linkage(X, "ward");
    }

    VectorXd pdist(MatrixXd& X) {
        int n = X.rows();
        int out_size = (n * (n - 1)) / 2;
        VectorXd dm(out_size);

        int k = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                VectorXd diff = X.row(i) - X.row(j);
                double diff_sum = diff.array().square().sum();
                dm[k] = std::sqrt(diff_sum);
                k++;
            }
        }

        return dm;
    }

    MatrixXd linkage(MatrixXd X, std::string method) {

        if (linkage_methods_set.find(method) == linkage_methods_set.end()) {
            throw std::invalid_argument("Invalid method: " + method);
        }

        VectorXd dists;
        // Check for 1D input
        if (X.cols() == 1){
            try {
                if (X.cols() != 1) {
                    std::string msg = "Condensed distance matrix must have shape=1 (i.e. be one-dimensional).";
                    throw std::invalid_argument(msg);
                }
                int n = X.rows();
                int d = std::ceil(std::sqrt(n * 2));
                if (d * (d - 1) / 2 != n) {
                    std::string msg = "Length n of condensed distance matrix must be a binomial coefficient, i.e. there must be a k such that (k \\choose 2)=n)!";
                    throw std::invalid_argument(msg);
                }

                dists = X.col(0);
            } catch (std::exception& e) {
                throw;
                std::cerr << e.what() << std::endl;
            }
        } else {
            if (X.rows() == X.cols() && X.diagonal().isZero() && (X.array() >= 0).all() && X.isApprox(X.transpose())) {
                std::cerr << "Warning: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix" << std::endl;
            }
            dists = pdist(X);
        }

        if (!(dists.array().isFinite().all())) {
            throw std::invalid_argument("The condensed distance matrix must contain only finite values.");
        }

        int n = (std::sqrt(8 * dists.rows() + 1) + 1) / 2;

        MatrixXd out;
        if (method == "single") {
            out = mst_single_linkage(dists, n);
        } else if (method == "complete") {
            out = nn_chain(dists, n, 1);
        } else if (method == "average") {
            out = nn_chain(dists, n, 2);
        } else if (method == "weighted") {
            out = nn_chain(dists, n, 6);
        } else if (method == "ward") {
            out = nn_chain(dists, n, 5);
        } else if (method == "centroid") {
            out = fast_linkage(dists, n, 3);
        } else if (method == "median") {
            out = fast_linkage(dists, n, 4);
        }

        return out;
    }

    class LinkageUnionFind {
        VectorXi parent;
        VectorXi size;
        int next_label;

    public:
        LinkageUnionFind(int n) : parent(2 * n - 1), size(2 * n - 1), next_label(n) {
            for (int i = 0; i < 2 * n - 1; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        int merge(int x, int y) {
            parent[x] = next_label;
            parent[y] = next_label;
            int size = this->size[x] + this->size[y];
            this->size[next_label] = size;
            next_label++;
            return size;
        }

        int find(int x) {
            int p = x;

            while (parent[x] != x) {
                x = parent[x];
            }

            while (parent[p] != x) {
                int temp = parent[p];
                parent[p] = x;
                p = temp;
            }

            return x;
        }
    };

    void label(MatrixXd& Z, int n) {
        LinkageUnionFind uf(n);
        int x, y, x_root, y_root;
        for (int i = 0; i < n - 1; i++) {
            x = static_cast<int>(Z(i, 0));
            y = static_cast<int>(Z(i, 1));
            x_root = uf.find(x);
            y_root = uf.find(y);
            if (x_root < y_root) {
                Z(i, 0) = x_root;
                Z(i, 1) = y_root;
            } else {
                Z(i, 0) = y_root;
                Z(i, 1) = x_root;
            }
            Z(i, 3) = uf.merge(x_root, y_root);
        }
    }

    MatrixXd mst_single_linkage(VectorXd dists, int n) {
        MatrixXd Z = MatrixXd::Zero(n - 1, 4);
        VectorXi merged = VectorXi::Zero(n);
        VectorXd D = VectorXd::Constant(n, std::numeric_limits<double>::infinity());

        int i, k, x = 0, y = 0;
        double dist, current_min;

        for (k = 0; k < n - 1; k++) {
            current_min = std::numeric_limits<double>::infinity();
            merged[x] = 1;
            for (i = 0; i < n; i++) {
                if (merged[i] == 1) {
                    continue;
                }

                dist = dists[condensed_index(n, x, i)];
                if (D[i] > dist) {
                    D[i] = dist;
                }

                if (D[i] < current_min) {
                    y = i;
                    current_min = D[i];
                }
            }

            Z(k, 0) = x;
            Z(k, 1) = y;
            Z(k, 2) = current_min;
            x = y;
        }


        // Sort Z by cluster distances.
        VectorXi order = argsort(Z.col(2));
        MatrixXd Z_new(Z.rows(), Z.cols());
        for (int i = 0; i < Z.rows(); ++i) {
            Z_new.row(i) = Z.row(order[i]);
        }
        Z = Z_new;

        // Find correct cluster labels inplace.
        label(Z, n);

        return Z;
    }

    double _single(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return std::min(d_xi, d_yi);
    }

    double _complete(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return std::max(d_xi, d_yi);
    }

    double _average(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return (size_x * d_xi + size_y * d_yi) / (size_x + size_y);
    }

    double _centroid(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                    (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y));
    }

    double _median(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return sqrt(0.5 * (d_xi * d_xi + d_yi * d_yi) - 0.25 * d_xy * d_xy);
    }

    double _ward(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        double t = 1.0 / (size_x + size_y + size_i);
        return sqrt((size_i + size_x) * t * d_xi * d_xi +
                    (size_i + size_y) * t * d_yi * d_yi -
                    size_i * t * d_xy * d_xy);
    }

    double _weighted(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i) {
        return 0.5 * (d_xi + d_yi);
    }

    linkage_distance_update linkage_methods[] = {_single, _complete, _average, _centroid, _median, _ward, _weighted};

    int64_t condensed_index(int64_t n, int64_t i, int64_t j) {
        if (i < j) {
            return n * i - (i * (i + 1) / 2) + (j - i - 1);
        } else if (i > j) {
            return n * j - (j * (j + 1) / 2) + (i - j - 1);
        }
        return -1;
    }

    // Define a comparison function that compares values in v
    struct IndexComparator {
        const VectorXd& v;
        IndexComparator(const VectorXd& v): v(v) {}
        bool operator()(int a, int b) const { return v[a] < v[b]; }
    };

    VectorXi argsort(const VectorXd& v) {
        std::vector<int> indices(v.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::stable_sort(indices.begin(), indices.end(), IndexComparator(v));

        // Convert std::vector<int> to VectorXi
        VectorXi eigen_indices(v.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            eigen_indices[i] = indices[i];
        }
        return eigen_indices;
    }

    MatrixXd nn_chain(VectorXd dists, int n, int method) {
        MatrixXd Z = MatrixXd::Zero(n - 1, 4);
        VectorXd D = dists;  // Distances between clusters.
        VectorXi size = VectorXi::Ones(n);  // Sizes of clusters.

        linkage_distance_update new_dist = linkage_methods[method];

        // Variables to store neighbors chain.
        VectorXi cluster_chain(n);
        int chain_length = 0;

        int i, k, x, y = 0, nx, ny, ni;
        double dist, current_min;


        for (k = 0; k < n - 1; k++) {
            if (chain_length == 0) {
                chain_length = 1;
                for (i = 0; i < n; i++) {
                    if (size[i] > 0) {
                        cluster_chain[0] = i;
                        break;
                    }
                }
            }

            // Go through chain of neighbors until two mutual neighbors are found.
            while (true) {
                x = cluster_chain[chain_length - 1];

                // We want to prefer the previous element in the chain as the
                // minimum, to avoid potentially going in cycles.
                if (chain_length > 1) {
                    y = cluster_chain[chain_length - 2];
                    current_min = D[condensed_index(n, x, y)];
                } else {
                    current_min = std::numeric_limits<double>::infinity();
                }

                for (i = 0; i < n; i++) {
                    if (size[i] == 0 || x == i) {
                        continue;
                    }

                    int64_t cond_index = condensed_index(n, x, i);
                    if (cond_index < D.size()) {
                        dist = D[cond_index];
                        if (dist < current_min) {
                            current_min = dist;
                            y = i;
                        }
                    }
                }

                if (chain_length > 1 && y == cluster_chain[chain_length - 2]) {
                    break;
                }

                cluster_chain[chain_length] = y;
                chain_length++;
            }

            // Merge clusters x and y and pop them from stack.
            chain_length -= 2;

            // This is a convention used in fastcluster.
            if (x > y) {
                std::swap(x, y);
            }

            // get the original numbers of points in clusters x and y
            nx = size[x];
            ny = size[y];

            // Record the new node.
            Z(k, 0) = x;
            Z(k, 1) = y;
            Z(k, 2) = current_min;
            Z(k, 3) = nx + ny;
            size[x] = 0;  // Cluster x will be dropped.
            size[y] = nx + ny;  // Cluster y will be replaced with the new cluster

            // Update the distance matrix.
            for (i = 0; i < n; i++) {
                ni = size[i];
                if (ni == 0 || i == y) {
                    continue;
                }

                D[condensed_index(n, i, y)] = new_dist(
                    D[condensed_index(n, i, x)],
                    D[condensed_index(n, i, y)],
                    current_min, nx, ny, ni);
            }
        }

        // Sort Z by cluster distances.
        VectorXi order = argsort(Z.col(2));
        MatrixXd Z_new(Z.rows(), Z.cols());
        for (int i = 0; i < Z.rows(); ++i) {
            Z_new.row(i) = Z.row(order[i]);
        }
        Z = Z_new;

        // Find correct cluster labels inplace.
        label(Z, n);

        return Z;
    }

    std::pair<int, double> find_min_dist(int n, VectorXd& D, VectorXi& size, int x) {
        double current_min = std::numeric_limits<double>::infinity();
        int y = -1;
        double dist;

        for (int i = x + 1; i < n; ++i) {
            if (size[i] == 0) {
                continue;
            }

            dist = D[condensed_index(n, x, i)];
            if (dist < current_min) {
                current_min = dist;
                y = i;
            }
        }

        return std::make_pair(y, current_min);
    }

    class Heap {
        public:
            Heap(VectorXd values) : values(values), size(values.size()) {
                index_by_key.resize(size);
                key_by_index.resize(size);
                std::iota(index_by_key.begin(), index_by_key.end(), 0);
                std::iota(key_by_index.begin(), key_by_index.end(), 0);

                for (int i = size / 2 - 1; i >= 0; --i) {
                    sift_down(i);
                }
            }

            std::pair<int, double> get_min() {
                return {key_by_index[0], values[0]};
            }

            void remove_min() {
                swap(0, size - 1);
                --size;
                sift_down(0);
            }

            void change_value(int key, double value) {
                int index = index_by_key[key];
                double old_value = values[index];
                values[index] = value;
                if (value < old_value) {
                    sift_up(index);
                } else {
                    sift_down(index);
                }
            }

        private:
            std::vector<int> index_by_key;
            std::vector<int> key_by_index;
            Eigen::VectorXd values;
            int size;

            void sift_up(int index) {
                int parent = Heap::parent(index);
                while (index > 0 && values[parent] > values[index]) {
                    swap(index, parent);
                    index = parent;
                    parent = Heap::parent(index);
                }
            }

            void sift_down(int index) {
                int child = Heap::left_child(index);
                while (child < size) {
                    if (child + 1 < size && values[child + 1] < values[child]) {
                        ++child;
                    }

                    if (values[index] > values[child]) {
                        swap(index, child);
                        index = child;
                        child = Heap::left_child(index);
                    } else {
                        break;
                    }
                }
            }

            static inline int left_child(int parent) {
                return 2 * parent + 1;
            }

            static inline int parent(int child) {
                return (child - 1) / 2;
            }

            void swap(int i, int j) {
                std::swap(values[i], values[j]);
                std::swap(key_by_index[i], key_by_index[j]);
                index_by_key[key_by_index[i]] = i;
                index_by_key[key_by_index[j]] = j;
            }
    };

    MatrixXd fast_linkage(VectorXd dists, int n, int method) {
        MatrixXd Z = MatrixXd::Zero(n - 1, 4);
        VectorXd D = dists;  // copy
        VectorXi size = VectorXi::Ones(n);
        VectorXi cluster_id = VectorXi::LinSpaced(n, 0, n-1);
        VectorXi neighbor = VectorXi::Zero(n - 1);
        VectorXd min_dist = VectorXd::Zero(n - 1);

        linkage_distance_update new_dist = linkage_methods[method];

        int x = 0, y = 0, z;
        int nx, ny, nz;
        int id_x, id_y;
        double dist = 0;

        for (int x = 0; x < n - 1; ++x) {
            std::pair<int, double> pair = find_min_dist(n, D, size, x);
            neighbor[x] = pair.first;
            min_dist[x] = pair.second;
        }
        Heap min_dist_heap(min_dist);

        for (int k = 0; k < n - 1; ++k) {
            for (int i = 0; i < n - k; ++i) {
                std::pair<int, double> pair = min_dist_heap.get_min();
                x = pair.first;
                dist = pair.second;
                y = neighbor[x];

                if (dist == D[condensed_index(n, x, y)]) {
                    break;
                }

                pair = find_min_dist(n, D, size, x);
                y = pair.first;
                dist = pair.second;
                neighbor[x] = y;
                min_dist[x] = dist;
                min_dist_heap.change_value(x, dist);
            }
            min_dist_heap.remove_min();

            id_x = cluster_id[x];
            id_y = cluster_id[y];
            nx = size[x];
            ny = size[y];

            if (id_x > id_y) {
                std::swap(id_x, id_y);
            }

            Z(k, 0) = id_x;
            Z(k, 1) = id_y;
            Z(k, 2) = dist;
            Z(k, 3) = nx + ny;

            size[x] = 0;
            size[y] = nx + ny;
            cluster_id[y] = n + k;

            // Update the distance matrix.
            for (int z = 0; z < n; ++z) {
                int nz = size[z];
                if (nz == 0 || z == y) {
                    continue;
                }

                D[condensed_index(n, z, y)] = new_dist(
                    D[condensed_index(n, z, x)], D[condensed_index(n, z, y)],
                    dist, nx, ny, nz);
            }

            // Reassign neighbor candidates from x to y.
            for (int z = 0; z < x; ++z) {
                if (size[z] > 0 && neighbor[z] == x) {
                    neighbor[z] = y;
                }
            }

            // Update lower bounds of distance.
            for (int z = 0; z < y; ++z) {
                if (size[z] == 0) {
                    continue;
                }

                dist = D[condensed_index(n, z, y)];
                if (dist < min_dist[z]) {
                    neighbor[z] = y;
                    min_dist[z] = dist;
                    min_dist_heap.change_value(z, dist);
                }
            }

            // Find nearest neighbor for y.
            if (y < n - 1) {
                std::pair<int, double> pair = find_min_dist(n, D, size, y);
                int z = pair.first;
                dist = pair.second;
                if (z != -1) {
                    neighbor[y] = z;
                    min_dist[y] = dist;
                    min_dist_heap.change_value(y, dist);
                }
            }
        }

        return Z;
    }

    void compute_ward_dist(
        const VectorXd& m_1,
        const MatrixXd& m_2,
        const VectorXi& coord_row,
        const VectorXi& coord_col,
        VectorXd& res
    ) {
        int size_max = coord_row.size();
        int n_features = m_2.cols();
        double pa, n;
        int row, col;

        for (int i = 0; i < size_max; ++i) {
            row = coord_row[i];
            col = coord_col[i];
            n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col]);
            pa = 0.;
            for (int j = 0; j < n_features; ++j) {
                pa += pow((m_2(row, j) / m_1[row] - m_2(col, j) / m_1[col]), 2);
            }
            res[i] = pa * n;
        }
    }

    void get_parents(const VectorXi& nodes, VectorXi& heads, const VectorXi& parents, VectorXi& notVisited) {
        int parent, node;

        for (int i = 0; i < nodes.size(); ++i) {
            node = nodes[i];
            parent = parents[node];
            while (parent != node) {
                node = parent;
                parent = parents[node];
            }
            if (notVisited[node]) {
                notVisited[node] = 0;
                heads.conservativeResize(heads.size() + 1);
                heads[heads.size() - 1] = node;
            }
        }
    }

    struct greater {
        template<class T>
        bool operator()(T const &a, T const &b) const { return a > b; }
    };

    VectorXi hc_cut(int n_clusters, MatrixXd children, int n_leaves) {
        if (n_clusters > n_leaves) {
            throw std::invalid_argument(
                "Cannot extract more clusters than samples: " +
                std::to_string(n_clusters) + " clusters were given for a tree with " +
                std::to_string(n_leaves) + " leaves."
            );
        }

        // In this function, we store nodes as a heap to avoid recomputing
        // the max of the nodes: the first element is always the smallest
        // We use negated indices as heaps work on smallest elements, and we
        // are interested in largest elements
        // children.row(children.rows()-1) is the root of the tree
        std::priority_queue<int, std::vector<int>, greater> nodes;
        nodes.push(-(children.row(children.rows()-1).maxCoeff() + 1));
        for (int i = 0; i < n_clusters - 1; ++i) {
            // As we have a heap, nodes.top() is the largest element
            VectorXi these_children = children.row(-nodes.top() - n_leaves).cast<int>();;
            // Insert the 2 children and remove the largest node
            nodes.push(-these_children(0));
            nodes.push(-these_children(1));
            nodes.pop();
        }

        VectorXi label = VectorXi::Zero(n_leaves);
        int i = 0;
        while (!nodes.empty()) {
            VectorXi indices = hc_get_descendent(-nodes.top(), children, n_leaves);
            for (int j = 0; j < indices.size(); ++j) {
                label(indices(j)) = i;
            }
            nodes.pop();
            i++;
        }

        return label;
    }

    VectorXi hc_get_descendent(int node, MatrixXd children, int n_leaves) {
        std::vector<int> ind = {node};
        if (node < n_leaves) {
            return VectorXi::Map(ind.data(), ind.size());
        }
        std::vector<int> descendent;

        // It is actually faster to do the accounting of the number of
        // elements in the list ourselves: size() is a lengthy operation on a
        // linked list
        int n_indices = 1;

        while (n_indices) {
            int i = ind.back();
            ind.pop_back();
            if (i < n_leaves) {
                descendent.push_back(i);
                --n_indices;
            } else {
                VectorXi children_nodes = children.row(i - n_leaves).cast<int>();
                ind.insert(ind.end(), children_nodes.data(), children_nodes.data() + children_nodes.size());
                ++n_indices;
            }
        }
        return VectorXi::Map(descendent.data(), descendent.size());
    }

    VectorXi hc_get_heads(VectorXi parents, bool copy) {
        if (copy) {
            parents = parents; // This creates a copy of parents
        }
        int size = parents.size();

        // Start from the top of the tree and go down
        for (int node0 = size - 1; node0 >= 0; --node0) {
            int node = node0;
            int parent = parents(node);
            while (parent != node) {
                parents(node0) = parent;
                node = parent;
                parent = parents(node);
            }
        }
        return parents;
    }

    MatrixXd mst_linkage_core(const MatrixXd& raw_data, DistanceMetric* dist_metric){
        int n_samples = raw_data.rows();
        std::vector<bool> in_tree(n_samples, false);
        MatrixXd result = MatrixXd::Zero(n_samples - 1, 3);

        int current_node = 0;
        int new_node;
        int num_features = raw_data.cols();

        double right_value;
        double left_value;
        double new_distance;

        VectorXd current_distances = VectorXd::Constant(n_samples, std::numeric_limits<double>::infinity());

        for (int i = 0; i < n_samples - 1; ++i) {
            in_tree[current_node] = true;

            new_distance = std::numeric_limits<double>::infinity();
            new_node = 0;

            for (int j = 0; j < n_samples; ++j) {
                if (in_tree[j]) {
                    continue;
                }

                right_value = current_distances[j];
                left_value = dist_metric->dist(raw_data.row(current_node), raw_data.row(j));

                if (left_value < right_value) {
                    current_distances[j] = left_value;
                }

                if (current_distances[j] < new_distance) {
                    new_distance = current_distances[j];
                    new_node = j;
                }
            }

            result(i, 0) = current_node;
            result(i, 1) = new_node;
            result(i, 2) = new_distance;
            current_node = new_node;
        }

        return result;
    }

    class UnionFind {
        public:
            UnionFind(int N) : parent(2 * N - 1, -1), next_label(N), size(2 * N - 1, 0) {
                std::fill(size.begin(), size.begin() + N, 1);
            }

            void union_sets(int m, int n) {
                parent[m] = next_label;
                parent[n] = next_label;
                size[next_label] = size[m] + size[n];
                next_label++;
            }

            int fast_find(int n) {
                int p = n;
                // find the highest node in the linkage graph so far
                while (n < parent.size() && parent[n] != -1) {
                    n = parent[n];
                }
                // provide a shortcut up to the highest node
                while (p < parent.size() && parent[p] != n) {
                    int temp = p;
                    p = parent[p];
                    parent[temp] = n;
                }
                return n;
            }

            int get_size(int n) {
                return size[n];
            }

        private:
            std::vector<int> parent;
            int next_label;
            std::vector<int> size;
    };

    MatrixXd single_linkage_label(MatrixXd& L) {
        if (L.block(0, 0, L.rows(), 2).minCoeff() < 0 || L.block(0, 0, L.rows(), 2).maxCoeff() >= 2 * L.rows() + 1) {
            throw std::invalid_argument("Input MST array is not a validly formatted MST array");
        }

        VectorXd weights = L.col(2);
        if (!std::is_sorted(weights.data(), weights.data() + weights.size())) {
            throw std::invalid_argument("Input MST array must be sorted by weight");
        }

        MatrixXd result_arr = MatrixXd::Zero(L.rows(), 4);
        UnionFind U(L.rows() + 1);

        for (int index = 0; index < L.rows(); ++index) {
            int left = static_cast<int>(L(index, 0));
            int right = static_cast<int>(L(index, 1));
            double delta = L(index, 2);

            int left_cluster = U.fast_find(left);
            int right_cluster = U.fast_find(right);

            result_arr(index, 0) = left_cluster;
            result_arr(index, 1) = right_cluster;
            result_arr(index, 2) = delta;
            result_arr(index, 3) = U.get_size(left_cluster) + U.get_size(right_cluster);

            U.union_sets(left_cluster, right_cluster);
        }

        return result_arr;
    }

    WeightedEdge::WeightedEdge(double weight, int a, int b) : weight(weight), a(a), b(b) {}

    bool WeightedEdge::operator<(const WeightedEdge& other) const {
        return weight < other.weight;
    }

    bool WeightedEdge::operator<=(const WeightedEdge& other) const {
        return weight <= other.weight;
    }

    bool WeightedEdge::operator==(const WeightedEdge& other) const {
        return weight == other.weight;
    }

    bool WeightedEdge::operator!=(const WeightedEdge& other) const {
        return weight != other.weight;
    }

    bool WeightedEdge::operator>(const WeightedEdge& other) const {
        return weight > other.weight;
    }

    bool WeightedEdge::operator>=(const WeightedEdge& other) const {
        return weight >= other.weight;
    }

    std::ostream& operator<<(std::ostream& os, const WeightedEdge& edge) {
        os << "WeightedEdge(weight=" << edge.weight << ", a=" << edge.a << ", b=" << edge.b << ")";
        return os;
    }

    std::map<int, double> max_merge(const std::map<int, double>& a, const std::map<int, double>& b, const std::vector<int>& mask, int n_a, int n_b) {

         std::map<int, double> out_obj;

        // First copy a into out_obj
        for (const auto& pair : a) {
            int key = pair.first;
            if (mask[key]) {
                out_obj[key] = pair.second;
            }
        }

        // Then merge b into out_obj
        for (const auto& pair : b) {
            int key = pair.first;
            double value = pair.second;
            if (mask[key]) {
                auto out_it = out_obj.find(key);
                if (out_it == out_obj.end()) {
                    // Key not found
                    out_obj[key] = value;
                } else {
                    out_it->second = std::max(out_it->second, value);
                }
            }
        }

        return out_obj;
    }

    std::map<int, double> average_merge(
        const std::map<int, double>& a,
        const std::map<int, double>& b,
        const std::vector<int>& mask,
        int n_a,
        int n_b
    ) {
        std::map<int, double> out_obj;
        double n_out = n_a + n_b;

        // First copy a into out_obj
        for (const auto& pair : a) {
            int key = pair.first;
            if (mask[key]) {
                out_obj[key] = pair.second;
            }
        }

        // Then merge b into out_obj
        for (const auto& pair : b) {
            int key = pair.first;
            double value = pair.second;
            if (mask[key]) {
                auto out_it = out_obj.find(key);
                if (out_it == out_obj.end()) {
                    // Key not found
                    out_obj[key] = value;
                } else {
                    out_it->second = (n_a * out_it->second + n_b * value) / n_out;
                }
            }
        }

        return out_obj;
    }

} // namespace hierachy