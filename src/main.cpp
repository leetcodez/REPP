#include <iostream>
#include <cmath>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace arma;
using namespace std;

int main() {
    mat dataset;
    bool loaded = mlpack::data::Load("../data/clean_gurugram_data.csv", dataset, true);
    
    if (!loaded) return -1;

    rowvec responses = dataset.row(dataset.n_rows - 1);
    dataset.shed_row(dataset.n_rows - 1);

    mlpack::LinearRegression lr(dataset, responses, 0.1);
    double mse = lr.ComputeError(dataset, responses);

    mat newProperty(9, 1);
    newProperty(0, 0) = 5.0;     // location id
    newProperty(1, 0) = 4200.0;  // sqft
    newProperty(2, 0) = 2.0;     // age
    newProperty(3, 0) = 0.0;     // dwarka exp
    newProperty(4, 0) = 7.0;     // dist nh48
    newProperty(5, 0) = 1.0;     // rapid metro
    newProperty(6, 0) = 2.0;     // luxury tier
    newProperty(7, 0) = 1.0;     // penthouse
    newProperty(8, 0) = 1.0;     // premium view

    rowvec prediction;
    lr.Predict(newProperty, prediction);
    
    long predicted_price = (long)prediction[0];

    double r = 0.12 / 12.0;
    int m = 5 * 12;
    double sip = predicted_price / (double)m; 
    long opp_cost = sip * ((std::pow(1 + r, m) - 1) / r) * (1 + r);

    cout << "Predicted Price: Rs. " << predicted_price << endl;
    cout << "SIP Cost (5Y):   Rs. " << opp_cost << endl;

    return 0;
}