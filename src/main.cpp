#include <iostream>
#include <cmath>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp> // The LARS (Lasso) Algorithm

using namespace arma;
using namespace std;

int main() {
    cout << "=========================================" << endl;
    cout << " v3.0 Elastic Net Hybrid Engine " << endl;
    cout << "=========================================" << endl;

    mat dataset;
    bool loaded = mlpack::data::Load("../data/clean_gurugram_data.csv", dataset, true);
    if (!loaded) return -1;

    rowvec responses = dataset.row(dataset.n_rows - 1);
    dataset.shed_row(dataset.n_rows - 1);

    // 1. TRAIN MODEL A: Ridge Linear Regression (L2 Penalty = 0.1)
    cout << "[INFO] Training Ridge Regressor (L2 Penalty)..." << endl;
    mlpack::LinearRegression lr(dataset, responses, 0.1);

    // 2. TRAIN MODEL B: LARS / Lasso Regression (L1 Penalty)
    // LARS uses Cholesky decomposition to perfectly handle sparse categorical features
    cout << "[INFO] Training LARS Regressor (L1 Feature Selection)..." << endl;
    bool useCholesky = true;
    double lambda1 = 0.1; // L1 Penalty
    double lambda2 = 0.0; // Elastic Net parameter
    mlpack::LARS lars(useCholesky, lambda1, lambda2);
    lars.Train(dataset, responses);

    cout << "[SUCCESS] Elastic Net Ensemble Trained Successfully!" << endl;

    // --- LIVE INFERENCE V3.0 ---
    // Target: Golf Course Road Penthouse (9 Features)
    mat newProperty(9, 1);
    newProperty(0, 0) = 5.0;     // Location ID 
    newProperty(1, 0) = 4200.0;  // SqFt
    newProperty(2, 0) = 2.0;     // Age
    newProperty(3, 0) = 0.0;     // Dwarka Exp
    newProperty(4, 0) = 7.0;     // Dist NH48
    newProperty(5, 0) = 1.0;     // Rapid Metro
    newProperty(6, 0) = 3.0;     // Luxury Tier (3 = Ultra-Luxury)
    newProperty(7, 0) = 1.0;     // Penthouse
    newProperty(8, 0) = 1.0;     // Premium View

    // Get Predictions from both models
    rowvec lr_pred, lars_pred;
    lr.Predict(newProperty, lr_pred);
    lars.Predict(newProperty, lars_pred);

    long ridge_price = (long)lr_pred[0];
    long lasso_price = (long)lars_pred[0];

    // THE ENSEMBLE: 50% Ridge, 50% Lasso (Elastic Net Blending)
    long final_ensemble_price = (ridge_price + lasso_price) / 2;

    // Financial Analysis
    double r = 0.12 / 12.0;
    int m = 5 * 12;
    double sip = final_ensemble_price / (double)m; 
    long opp_cost = sip * ((std::pow(1 + r, m) - 1) / r) * (1 + r);

    cout << "\n--- PROPERTY INFERENCE & ANALYSIS ---" << endl;
    cout << "Ridge Regression Estimate:   Rs. " << ridge_price << endl;
    cout << "Lasso (LARS) Estimate:       Rs. " << lasso_price << endl;
    cout << "-----------------------------------------" << endl;
    cout << "ELASTIC NET VALUATION:       Rs. " << final_ensemble_price << endl;
    cout << "SIP Opportunity Cost (5Y):   Rs. " << opp_cost << endl;

    return 0;
}