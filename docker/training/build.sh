#!/bin/bash

# Build the Rho training container.

set -e

echo "Building Rho training container..."

# Change to the project root directory
cd "$(dirname "$0")/../.."

# Build the Docker image
docker build -t rho-training:latest -f docker/training/Dockerfile .
