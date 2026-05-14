{
  "name": "repp-microservice",
  "version": "5.0.0",
  "description": "Real Estate Price Predictor Microservice",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "express": "^4.18.2"
  }
}dillo-dev \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Python dependencies for data cleaning
RUN pip3 install pandas numpy

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Run the Python data cleaning script to generate data matrices
RUN python3 clean_data.py

# Compile the C++ ML engine binary
RUN mkdir -p build && cd build && cmake .. && make

# Install Node.js dependencies
RUN npm install

# Expose API port
EXPOSE 3000

# Start the Node.js Express server
CMD ["node", "server.js"]
m(), 10);
        if (isNaN(price)) {
            console.error(`Invalid output from predictor: ${stdout}`);
            return res.status(500).json({ error: 'Invalid prediction result.' });
        }
        
        res.json({ price });
    });
});

app.listen(PORT, () => {
    console.log(`REPP Microservice running on port ${PORT}`);
});
 each feature (row)
        feature_means = mean(trainData, 1);
        feature_stdevs = stddev(trainData, 0, 1);

        scaleData(trainData);

        // 1. OLS Linear Regression (with tiny L2 penalty)
        ols = mlpack::LinearRegression(trainData, trainResponses, 0.001); 

        // 2. Ridge Regression
        ridge = mlpack::LinearRegression(trainData, trainResponses, 0.5);

        // 3. Lasso Regression
        lasso.Train(trainData, trainResponses);

        // 4. Elastic Net
        elastic_net.Train(trainData, trainResponses);

        // 5. Bayesian Linear Regression
        blr.Train(trainData, trainResponses);
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

        return (est_ols + est_ridge + est_lasso + est_elastic + est_blr) / 5;
    }
};

int main(int argc, char* argv[]) {
    // Disable mlpack logging to stdout/stderr to ensure a clean integer output
    mlpack::Log::Info.ignoreInput = true;
    mlpack::Log::Warn.ignoreInput = true;

    if (argc != 10) {
        cerr << "Error: Expected exactly 9 property features as arguments." << endl;
        return -1;
    }

    mat dataset;
    // Attempt to load from the project root (when run via Node server)
    if (!mlpack::data::Load("data/clean_gurugram_data.csv", dataset, false)) {
        // Fallback for execution from the build directory
        if (!mlpack::data::Load("../data/clean_gurugram_data.csv", dataset, false)) {
            cerr << "Error: Failed to load dataset." << endl;
            return -1;
        }
    }

    // Extract targets (responses) which are located in the last row
    rowvec responses = dataset.row(dataset.n_rows - 1);
    dataset.shed_row(dataset.n_rows - 1);

    // Train the ensemble
    RealEstateEnsemble ensemble;
    ensemble.train(dataset, responses);

    // Parse the 9 command line arguments into a matrix
    mat newProperty(9, 1);
    for (int i = 0; i < 9; ++i) {
        newProperty(i, 0) = std::stod(argv[i + 1]);
    }

    // Predict and output ONLY the final blended integer price
    long final_ensemble_price = ensemble.predict(newProperty);
    cout << final_ensemble_price << endl;
    
    return 0;
}

