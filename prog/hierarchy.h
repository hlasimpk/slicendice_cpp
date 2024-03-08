#include "dist_metrics.h"

#include <Eigen/Dense>
#include <map>
#include <vector>

using Eigen::MatrixXd;

namespace hierarchy {
    typedef double (*linkage_distance_update)(double, double, double, int, int, int);

    MatrixXd single(MatrixXd& X);

    double _single(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd complete(MatrixXd& X);

    double _complete(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd average(MatrixXd& X);

    double _average(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd weighted(MatrixXd& X);

    double _weighted(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd centroid(MatrixXd& X);

    double _centroid(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd median(MatrixXd& X);

    double _median(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    MatrixXd ward(MatrixXd& X);

    double _ward(double d_xi, double d_yi, double d_xy, int size_x, int size_y, int size_i);

    void label(MatrixXd& Z, int n);

    VectorXd pdist(MatrixXd& X);

    int64_t condensed_index(int64_t n, int64_t i, int64_t j);

    VectorXi argsort(const VectorXd& v);

    MatrixXd linkage(MatrixXd X, std::string method);

    MatrixXd mst_single_linkage(VectorXd dists, int n);

    MatrixXd nn_chain(VectorXd dists, int n, int method);

    MatrixXd fast_linkage(VectorXd dists, int n, int method);

    void compute_ward_dist(
        const VectorXd& m_1,
        const MatrixXd& m_2,
        const VectorXi& coord_row,
        const VectorXi& coord_col,
        VectorXd& res);

    void get_parents(
        const VectorXi& nodes, 
        VectorXi& heads, 
        const VectorXi& parents, 
        VectorXi& notVisited);

    VectorXi hc_cut(int n_clusters, MatrixXd children, int n_leaves);

    VectorXi hc_get_descendent(int node, MatrixXd children, int n_leaves);

    VectorXi hc_get_heads(VectorXi parents, bool copy=true);

    MatrixXd mst_linkage_core(const MatrixXd& raw_data, DistanceMetric* dist_metric);

    MatrixXd single_linkage_label(MatrixXd& L);

    class WeightedEdge {
    public:
        double weight;
        int a;
        int b;
        WeightedEdge(double weight, int a, int b);

        bool operator<(const WeightedEdge& other) const;
        bool operator<=(const WeightedEdge& other) const;
        bool operator==(const WeightedEdge& other) const;
        bool operator!=(const WeightedEdge& other) const;
        bool operator>(const WeightedEdge& other) const;
        bool operator>=(const WeightedEdge& other) const;

        friend std::ostream& operator<<(std::ostream& os, const WeightedEdge& edge);

    };

    std::map<int, double> max_merge(const std::map<int, double>& a, const std::map<int, double>& b, const std::vector<int>& mask, int n_a, int n_b);

    std::map<int, double> average_merge(const std::map<int, double>& a, const std::map<int, double>& b, const std::vector<int>& mask, int n_a, int n_b);

} // namespace hierarchy