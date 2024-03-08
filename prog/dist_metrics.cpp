#include "dist_metrics.h"

#include <cmath>
#include <Eigen/Dense>

#include <iostream>

EuclideanDistance::EuclideanDistance(double input_p) : p(input_p) {}

double EuclideanDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    double sum = 0.0;
    for (int i = 0; i < x1.size(); ++i) {
        double diff = x1(i) - x2(i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double EuclideanDistance::rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    double sum = 0.0;
    for (int i = 0; i < x1.size(); ++i) {
        double diff = x1(i) - x2(i);
        sum += diff * diff;
    }
    return sum;
}

double EuclideanDistance::rdist_to_dist(double rdist) {
    return std::sqrt(rdist);
}

double EuclideanDistance::dist_to_rdist(double dist) {
    return dist * dist;
}

MinkowskiDistance::MinkowskiDistance(double input_p, const Eigen::VectorXd& input_w) : p(input_p), w(input_w), size(input_w.size()) {
    if (p <= 0) {
            throw std::invalid_argument("p must be greater than 0");
        }
        if (std::isinf(p)) {
            throw std::invalid_argument("MinkowskiDistance requires finite p. For p=inf, use ChebyshevDistance.");
        }
        if ((w.array() < 0).any()) {
            throw std::invalid_argument("w cannot contain negative weights");
        }
}

void MinkowskiDistance::validate_data(const Eigen::MatrixXd& X) {
    if (size > 0 && X.cols() != size) {
        throw std::invalid_argument("MinkowskiDistance: the size of w must match the number of features.");
    }
}

double MinkowskiDistance::rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    Eigen::VectorXd x1_vec = Eigen::VectorXd::Map(x1.data(), x1.size());
    Eigen::VectorXd x2_vec = Eigen::VectorXd::Map(x2.data(), x2.size());

    double d = 0;
    if (size > 0) {
        for (int j = 0; j < size; ++j) {
            d += (w[j] * std::pow(std::abs(x1_vec[j] - x2_vec[j]), p));
        }
    } else {
        for (int j = 0; j < x1_vec.size(); ++j) {
            d += (std::pow(std::abs(x1_vec[j] - x2_vec[j]), p));
        }
    }
    return d;
}

double MinkowskiDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    return std::pow(rdist(x1, x2), 1. / p);
}


double MinkowskiDistance::rdist_to_dist(double rdist) {
    return std::pow(rdist, 1. / p);
}

double MinkowskiDistance::dist_to_rdist(double dist) {
    return std::pow(dist, p);
}


ManhattanDistance::ManhattanDistance(double input_p) : p(input_p) {}

double ManhattanDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    Eigen::VectorXd x1_vec = Eigen::VectorXd::Map(x1.data(), x1.size());
    Eigen::VectorXd x2_vec = Eigen::VectorXd::Map(x2.data(), x2.size());

    if (x1_vec.size() != x2_vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }
    double d = 0;
    for (int j = 0; j < x1_vec.size(); ++j) {
        d += std::abs(x1_vec[j] - x2_vec[j]);
    }
    return d;
}


ChebyshevDistance::ChebyshevDistance(){}

double ChebyshevDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    Eigen::VectorXd x1_vec = Eigen::VectorXd::Map(x1.data(), x1.size());
    Eigen::VectorXd x2_vec = Eigen::VectorXd::Map(x2.data(), x2.size());

    if (x1_vec.size() != x2_vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }
    double d = 0;
    for (int j = 0; j < x1_vec.size(); ++j) {
        d = std::max(d, std::abs(x1_vec[j] - x2_vec[j]));
    }
    return d;
}

SEuclideanDistance::SEuclideanDistance(const Eigen::MatrixXd& V) : vec(V), size(V.size()), p(2) {} 

void SEuclideanDistance::validate_data(const Eigen::MatrixXd& X) {
    if (X.cols() != size) {
        throw std::invalid_argument("SEuclidean dist: size of V does not match");
    }
}

double SEuclideanDistance::rdist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    double d = 0;
    for (int j = 0; j < size; ++j) {
        double tmp = x1(j) - x2(j);
        d += (tmp * tmp / vec(j));
    }
    return d;
}

double SEuclideanDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    return std::sqrt(rdist(x1, x2));
}

double SEuclideanDistance::rdist_to_dist(double rdist) {
    return std::sqrt(rdist);
}

double SEuclideanDistance::dist_to_rdist(double dist) {
    return dist * dist;
}

MahalanobisDistance::MahalanobisDistance(const Eigen::MatrixXd& V) : mat(V.inverse()) {
    size = mat.rows();
    buffer = Eigen::VectorXd::Zero(size);
}

double MahalanobisDistance::dist(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2) {
    Eigen::VectorXd x1_vec = Eigen::VectorXd::Map(x1.data(), x1.size());
    Eigen::VectorXd x2_vec = Eigen::VectorXd::Map(x2.data(), x2.size());

    if (x1_vec.size() != x2_vec.size() || x1_vec.size() != size) {
        throw std::invalid_argument("Vectors must be of the same size as V.");
    }
    for (int i = 0; i < size; ++i) {
        buffer(i) = x1_vec[i] - x2_vec[i];
    }
    double d = buffer.transpose() * mat * buffer;
    return std::sqrt(d);
}


// Define the mapping from metric names to factory methods
std::unordered_map<std::string, DistanceMetricFactory::FactoryMethod> DistanceMetricFactory::metricMapping = {
    {"euclidean", [](double p) { return new EuclideanDistance(p); }},
    {"l2", [](double p) { return new EuclideanDistance(p); }},
    {"minkowski", [](double p, const Eigen::MatrixXd& w) { return new MinkowskiDistance(p, w); }},
    {"p", [](double p, const Eigen::MatrixXd& w) { return new MinkowskiDistance(p, w); }},
    {"manhattan", [](double p) { return new ManhattanDistance(p); }},
    {"cityblock", [](double p) { return new ManhattanDistance(p); }},
    {"l1", [](double p) { return new ManhattanDistance(p); }},
    {"seuclidean", [](const Eigen::MatrixXd& V) { return new SEuclideanDistance(V); }},
    {"chebyshev", []() { return new ChebyshevDistance(); }},
    {"infinity", []() { return new ChebyshevDistance(); }},
    {"mahalanobis", [](const Eigen::MatrixXd& V) { return new MahalanobisDistance(V); }},
};

Eigen::VectorXd cosine_distance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
    Eigen::MatrixXd x_normalized = x.rowwise().normalized();
    Eigen::MatrixXd y_normalized = y.rowwise().normalized();
    Eigen::MatrixXd diff = x_normalized - y_normalized;
    return (0.5 * diff.rowwise().squaredNorm()).colwise().sum();
}

Eigen::VectorXd manhattan_distance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
    Eigen::MatrixXd diff = x - y;
    return diff.cwiseAbs().rowwise().sum();
}

Eigen::VectorXd euclidean_distance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
    Eigen::MatrixXd diff = x - y;
    return diff.rowwise().norm();
}

Eigen::VectorXd paired_distances(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::string& metric) {
    if (metric == "cosine") {
        return cosine_distance(x, y);
    } else if (metric == "euclidean" || metric == "l2") {
        return euclidean_distance(x, y);
    } else if (metric == "l1" || metric == "manhattan" || metric == "cityblock") {
        return manhattan_distance(x, y);
    } else {
        throw std::invalid_argument("Unknown distance " + metric);
    }
}