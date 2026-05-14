FROM ubuntu:22.04

# Avoid tzdata prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install core C++ dependencies, Python, and utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libmlpack-dev \
    libarmadillo-dev \
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