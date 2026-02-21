#include <iostream>
#include <cmath> // Needed for the SIP power calculation
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace arma;
using namespace std;

int main() {
    cout << "=========================================" << endl;
    cout << " ML Project Engine: Gurugram Micro-Market" << endl;
    cout << "=========================================" << endl;
    
    mat dataset;
    bool loaded = mlpack::data::Load("../data/clean_gurugram_data.csv", dataset, true);
    
    if (!loaded) {
        cout << "Error: Could not load the dataset." << endl;
        return -1;
    }
    cout << "[INFO] Loaded " << dataset.n_cols << " properties from Gurugram." << endl;

    // Separate the Target Variable (Price)
    rowvec responses = dataset.row(dataset.n_rows - 1);
    dataset.shed_row(dataset.n_rows - 1);

    cout << "[INFO] Training Multiple Linear Regression model..." << endl;

    // Train the Model
    mlpack::LinearRegression lr(dataset, responses);

    // Evaluate the Model
    double mse = lr.ComputeError(dataset, responses);
    cout << "[SUCCESS] Model trained flawlessly! Mean Squared Error: " << mse << endl;

    // --- LIVE INFERENCE ---
    // Now we only pass 3 features: Location ID, SqFt, Age
    mat newProperty(3, 1);
    newProperty(0, 0) = 2.0;       // Location ID 
    newProperty(1, 0) = 1500.0;    // Carpet Area in SqFt
    newProperty(2, 0) = 5.0;       // Age in Years

    rowvec prediction;
    lr.Predict(newProperty, prediction);
    
    long predicted_price = (long)prediction[0];

    // --- FINANCIAL ANALYSIS ENGINE ---
    // Calculate the opportunity cost dynamically based on the model's prediction
    double annual_rate = 0.12;
    int months = 5 * 12; // 5 years
    double monthly_rate = annual_rate / 12.0;
    
    double estimated_monthly_sip = predicted_price / (double)months; 
    long opportunity_cost = estimated_monthly_sip * ((std::pow(1 + monthly_rate, months) - 1) / monthly_rate) * (1 + monthly_rate);

    cout << "\n--- PROPERTY INFERENCE & ANALYSIS ---" << endl;
    cout << "Predicted Property Price:            Rs. " << predicted_price << endl;
    cout << "Nifty50 Opportunity Cost (5 Years):  Rs. " << opportunity_cost << endl;
    
    if (opportunity_cost > predicted_price) {
        cout << "Verdict: The SIP would have generated Rs. " << (opportunity_cost - predicted_price) << " more in value." << endl;
    }

    return 0;
}