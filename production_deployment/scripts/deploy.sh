#!/bin/bash
set -e

echo "🚀 Deploying Gaudi 3 Scale to production"

# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.json
kubectl apply -f k8s/deployment.json
kubectl apply -f k8s/service.json
kubectl apply -f k8s/hpa.json

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/gaudi3-scale -n gaudi3-scale

# Check deployment status
kubectl get pods -n gaudi3-scale
kubectl get svc -n gaudi3-scale

echo "✅ Deployment completed successfully"
