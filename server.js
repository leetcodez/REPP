const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
    const features = req.body.features; // Expecting an array of 9 numerical features

    if (!features || !Array.isArray(features) || features.length !== 9) {
        return res.status(400).json({ error: 'Please provide an array of exactly 9 features.' });
    }

    // Join features with spaces for command line arguments
    const args = features.join(' ');
    
    exec(`./build/predictor ${args}`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing predictor: ${error}`);
            console.error(`Stderr: ${stderr}`);
            return res.status(500).json({ error: 'Prediction failed.' });
        }
        
        const price = parseInt(stdout.trim(), 10);
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