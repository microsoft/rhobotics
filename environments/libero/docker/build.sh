#!/bin/bash

# Build script for ALKU Libero container
# This script builds the Docker image for ALKU Libero development

set -e

echo "Building ALKU Libero container..."

# Change to the project root directory
cd "$(dirname "$0")/../../.."

# Build the Docker image
docker build -t rho-libero:latest -f environments/libero/docker/Dockerfile .
