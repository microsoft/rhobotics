#!/bin/bash

# Build the Rho LIBERO container.

set -e

echo "Building Rho LIBERO container..."

# Change to the project root directory
cd "$(dirname "$0")/../../.."

# Build the Docker image
docker build -t rho-libero:latest -f environments/libero/docker/Dockerfile .
