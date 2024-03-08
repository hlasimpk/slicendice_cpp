#ifndef DIST_METRICS_H
#define DIST_METRICS_H

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <unordered_map>


class DistanceMetric {
public:
    virtual ~DistanceMetric() = default;
    virtual double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) = 0;
};

class EuclideanDistance : public DistanceMetric {
    public:
        EuclideanDistance(double input_p = 2);
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist_to_dist(double rdist);
        double dist_to_rdist(double dist);
    private:
        int p;
};

class MinkowskiDistance : public DistanceMetric {
    public:
        MinkowskiDistance(double p = 2, const Eigen::VectorXd& w = Eigen::VectorXd());
        void validate_data(const Eigen::MatrixXd& X);
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist_to_dist(double rdist);
        double dist_to_rdist(double dist);

    private:
        double p;
        Eigen::VectorXd w;
        int size;
};

class ManhattanDistance : public DistanceMetric {
    public:
        ManhattanDistance(double input_p = 1);
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
    private:
        int p;
};

class ChebyshevDistance : public DistanceMetric {
    public:
        ChebyshevDistance();
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
};

class SEuclideanDistance : public DistanceMetric {
    public:
        SEuclideanDistance(const Eigen::MatrixXd& V);
        void validate_data(const Eigen::MatrixXd& X);
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
        double rdist_to_dist(double rdist);
        double dist_to_rdist(double dist);
    private:
        Eigen::MatrixXd vec;
        int size;
        int p;
};

class MahalanobisDistance : public DistanceMetric {
    public:
        MahalanobisDistance(const Eigen::MatrixXd& V);
        double dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2);
    private:
        Eigen::MatrixXd mat;
        Eigen::VectorXd buffer;
        int size;
};

class DistanceMetricFactory {
public:
    typedef std::variant<
    std::function<DistanceMetric*()>,
    std::function<DistanceMetric*(double)>,
    std::function<DistanceMetric*(double, const Eigen::MatrixXd&)>,
    std::function<DistanceMetric*(const Eigen::MatrixXd&)>
    > FactoryMethod;
    static std::unordered_map<std::string, FactoryMethod> metricMapping;
    static DistanceMetric* getMetric(std::string& metric, 
                                     const Eigen::MatrixXd& V = Eigen::MatrixXd(), 
                                     double p = -1, 
                                     const Eigen::VectorXd& w = Eigen::VectorXd()) {

        // In Minkowski special cases, return more efficient methods
        if (metric == "minowski" || metric == "p") {
            if (p == 1 && w.size() == 0) {
                metric = "manhattan";
            } else if ((p == 2 || -1) && w.size() == 0) {
                metric = "euclidean";
            }
        }

        auto it = metricMapping.find(metric);
        if (it == metricMapping.end()) {
            throw std::invalid_argument("Unrecognized metric '" + metric + "'");
        }
        return std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::function<DistanceMetric*()>>) {
                return arg();
            } else if constexpr (std::is_same_v<T, std::function<DistanceMetric*(double)>>) {
                if (p < 0) {
                    if (metric == "euclidean" || metric == "l2") {
                        return arg(2);
                    } else if (metric == "manhattan" || metric == "l1" || metric == "cityblock") {
                        return arg(1);
                    } else {
                        throw std::invalid_argument("Metric requires a p value");
                    }
                } else {
                    return arg(p);
                }
            } else if constexpr (std::is_same_v<T, std::function<DistanceMetric*(double, const Eigen::MatrixXd&)>>) {
                return arg(p, w);
            } else if constexpr (std::is_same_v<T, std::function<DistanceMetric*(const Eigen::MatrixXd&)>>) {
                return arg(V);
            } else {
                throw std::logic_error("Unrecognized metric factory method type");
            }
        }, it->second);
    }
};

Eigen::VectorXd paired_cosine_distances(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

Eigen::VectorXd paired_manhattan_distances(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

Eigen::VectorXd paired_euclidean_distances(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

Eigen::VectorXd paired_distances(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::string& metric = "euclidean");

#endif // DIST_METRICS_H