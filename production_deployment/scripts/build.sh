#!/bin/bash
set -e

echo "🏗️  Building Gaudi 3 Scale v0.5.0"

# Build Docker image
docker build -t gaudi3-scale:0.5.0 .
docker tag gaudi3-scale:0.5.0 gaudi3-scale:latest

echo "✅ Build completed successfully"
