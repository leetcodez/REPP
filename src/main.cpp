#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp>

using namespace arma;
using namespace std;

class RealEstateEnsemble {
private:
    mlpack::LinearRegression ols;
    mlpack::LinearRegression ridge;
    mlpack::LARS lasso;
    mlpack::LARS elastic_net;
    mlpack::BayesianLinearRegression blr;
    
    vec feature_means;
    vec feature_stdevs;

    // Z-Score Normalization
    void scaleData(mat& data) {
        if (feature_means.is_empty() || feature_stdevs.is_empty()) {
            throw std::logic_error("Scaler not initialized. Call train() first.");
        }
        for (size_t i = 0; i < data.n_rows; ++i) {
            if (feature_stdevs(i) > 0) {
                data.row(i) = (data.row(i) - feature_means(i)) / feature_stdevs(i);
            }
        }
    }

    double calculateMSE(const rowvec& predictions, const rowvec& actual) {
        if (predictions.n_elem != actual.n_elem || actual.n_elem == 0) return -1.0;
        return accu(square(predictions - actual)) / actual.n_elem;
    }

public:
    RealEstateEnsemble() : lasso(true, 0.1, 0.0), elastic_net(true, 0.1, 0.1), blr(true) {}

    void train(mat trainData, const rowvec& trainResponses) {
        cout << "[INFO] Initializing feature scalers..." << endl;
        // Compute statistics for each feature (row)
        feature_means = mean(trainData, 1);
        feature_stdevs = stddev(trainData, 0, 1);

        cout << "[INFO] Scaling training features..." << endl;
        scaleData(trainData);

        // 1. OLS Linear Regression (with tiny L2 penalty to prevent singularity matrix issues)
        cout << "[INFO] Training OLS Linear Regression..." << endl;
        ols = mlpack::LinearRegression(trainData, trainResponses, 0.001); 

        // 2. Ridge Regression
        cout << "[INFO] Training Ridge Regression (L2)..." << endl;
        ridge = mlpack::LinearRegression(trainData, trainResponses, 0.5);

        // 3. Lasso Regression
        cout << "[INFO] Training Lasso Regression (L1)..." << endl;
        lasso.Train(trainData, trainResponses);

        // 4. Elastic Net
        cout << "[INFO] Training Elastic Net (L1 & L2)..." << endl;
        elastic_net.Train(trainData, trainResponses);

        // 5. Bayesian Linear Regression
        cout << "[INFO] Training Bayesian Linear Regression..." << endl;
        blr.Train(trainData, trainResponses);
        
        cout << "[SUCCESS] Ensemble of 5 models trained successfully!" << endl;
    }

    void evaluate(mat testData, const rowvec& testResponses) {
        // Must scale test data using training statistics to avoid data leakage
        scaleData(testData);

        rowvec p_ols, p_ridge, p_lasso, p_elastic, p_blr;
        ols.Predict(testData, p_ols);
        ridge.Predict(testData, p_ridge);
        lasso.Predict(testData, p_lasso);
        elastic_net.Predict(testData, p_elastic);
        blr.Predict(testData, p_blr);

        cout << "\n--- PERFORMANCE CRITERIA (MSE on Unseen Test Set) ---" << endl;
        cout << scientific << setprecision(4);
        cout << "OLS Regression:       " << calculateMSE(p_ols, testResponses) << endl;
        cout << "Ridge Regression:     " << calculateMSE(p_ridge, testResponses) << endl;
        cout << "Lasso Regression:     " << calculateMSE(p_lasso, testResponses) << endl;
        cout << "Elastic Net:          " << calculateMSE(p_elastic, testResponses) << endl;
        cout << "Bayesian Regression:  " << calculateMSE(p_blr, testResponses) << endl;
        cout << fixed << setprecision(0); // Reset for prices
    }

    long predict(mat newProperty) {
        scaleData(newProperty);

        rowvec p_ols, p_ridge, p_lasso, p_elastic, p_blr;
        ols.Predict(newProperty, p_ols);
        ridge.Predict(newProperty, p_ridge);
        lasso.Predict(newProperty, p_lasso);
        elastic_net.Predict(newProperty, p_elastic);
        blr.Predict(newProperty, p_blr);

        long est_ols = (long)p_ols[0];
        long est_ridge = (long)p_ridge[0];
        long est_lasso = (long)p_lasso[0];
        long est_elastic = (long)p_elastic[0];
        long est_blr = (long)p_blr[0];

        cout << "\n--- PROPERTY INFERENCE ---" << endl;
        cout << "OLS Estimate:            Rs. " << est_ols << endl;
        cout << "Ridge Estimate:          Rs. " << est_ridge << endl;
        cout << "Lasso Estimate:          Rs. " << est_lasso << endl;
        cout << "Elastic Net Estimate:    Rs. " << est_elastic << endl;
        cout << "Bayesian Estimate:       Rs. " << est_blr << endl;
        cout << "-----------------------------------------" << endl;

        return (est_ols + est_ridge + est_lasso + est_elastic + est_blr) / 5;
    }
};

int main() {
    cout << "=========================================" << endl;
    cout << " v5.0 Robust ML Ensemble Engine" << endl;
    cout << "=========================================" << endl;

    mat dataset;
    if (!mlpack::data::Load("../data/clean_gurugram_data.csv", dataset, true)) {
        cerr << "[ERROR] Failed to load dataset! Ensure clean_gurugram_data.csv exists." << endl;
        return -1;
    }

    // Extract targets (responses) which are located in the last row
    rowvec responses = dataset.row(dataset.n_rows - 1);
    dataset.shed_row(dataset.n_rows - 1);

    // Split data into 80% training and 20% testing
    mat trainData, testData;
    rowvec trainResponses, testResponses;
    mlpack::data::Split(dataset, responses, trainData, testData, trainResponses, testResponses, 0.2, true);

    cout << "[INFO] Dataset loaded and split: Train (" << trainData.n_cols << "), Test (" << testData.n_cols << ")." << endl;

    // Initialize and train our ensemble wrapper
    RealEstateEnsemble ensemble;
    ensemble.train(trainData, trainResponses);

    // Evaluate on test data
    ensemble.evaluate(testData, testResponses);

    // Inference on a new unscaled property matrix (9 features x 1 column)
    mat newProperty(9, 1);
    newProperty(0, 0) = 5.0;     // Location ID 
    newProperty(1, 0) = 4200.0;  // SqFt
    newProperty(2, 0) = 2.0;     // Age
    newProperty(3, 0) = 0.0;     // Dwarka Exp
    newProperty(4, 0) = 7.0;     // Dist NH48
    newProperty(5, 0) = 1.0;     // Rapid Metro
    newProperty(6, 0) = 3.0;     // Luxury Tier
    newProperty(7, 0) = 1.0;     // Penthouse
    newProperty(8, 0) = 1.0;     // Premium View

    long final_ensemble_price = ensemble.predict(newProperty);

    // Calculate Opportunity Cost (SIP comparison)
    double r = 0.12 / 12.0; // 12% annual return
    int m = 5 * 12; // 5 years
    double sip = final_ensemble_price / (double)m; 
    long opp_cost = sip * ((std::pow(1 + r, m) - 1) / r) * (1 + r);

    cout << "ENSEMBLE VALUATION:      Rs. " << final_ensemble_price << endl;
    cout << "SIP Opportunity Cost:    Rs. " << opp_cost << " (5Y @ 12%)" << endl;
    cout << "=========================================" << endl;
    
    return 0;
}