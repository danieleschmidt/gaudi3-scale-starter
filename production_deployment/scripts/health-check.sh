#!/bin/bash

echo "🏥 Checking Gaudi 3 Scale health..."

# Check if pods are running
RUNNING_PODS=$(kubectl get pods -n gaudi3-scale --field-selector=status.phase=Running --no-headers | wc -l)
TOTAL_PODS=$(kubectl get pods -n gaudi3-scale --no-headers | wc -l)

echo "📊 Pod Status: $RUNNING_PODS/$TOTAL_PODS running"

# Check service endpoints
if kubectl get endpoints gaudi3-scale-service -n gaudi3-scale | grep -q "none"; then
    echo "❌ No healthy endpoints"
    exit 1
else
    echo "✅ Service endpoints healthy"
fi

# Check HPA status
HPA_STATUS=$(kubectl get hpa gaudi3-scale-hpa -n gaudi3-scale -o jsonpath='{.status.conditions[?(@.type=="AbleToScale")].status}')
if [ "$HPA_STATUS" = "True" ]; then
    echo "✅ HPA is functioning"
else
    echo "⚠️  HPA status: $HPA_STATUS"
fi

echo "✅ Health check completed"
