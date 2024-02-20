#include <vector>

typedef std::vector<std::vector< double > > Vector3D;

Vector3D transpose (Vector3D data);

Vector3D dot_product (const Vector3D& X, const Vector3D& Y);

std::vector<double> row_norms (Vector3D data, bool squared = false);

Vector3D euclidean_distances (Vector3D X, Vector3D Y, std::vector<double> x_squared_norms, std::vector<double> y_squared_norms, bool squared = false);

std::vector<double> calculate_mean (const Vector3D& X);

std::vector<double> calculate_variance (const Vector3D& X);