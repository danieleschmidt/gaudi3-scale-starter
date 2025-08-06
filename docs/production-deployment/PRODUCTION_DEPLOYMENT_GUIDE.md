# Gaudi 3 Scale Production Deployment Guide

This comprehensive guide covers the complete production deployment of the Gaudi 3 Scale platform, including infrastructure setup, application deployment, monitoring, security, and operational procedures.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Deployment](#infrastructure-deployment)
3. [Application Deployment](#application-deployment)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Security Configuration](#security-configuration)
6. [Backup and Recovery](#backup-and-recovery)
7. [Operational Procedures](#operational-procedures)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

## Prerequisites

### Required Tools

```bash
# Install required CLI tools
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
```

### Cloud Provider Setup

#### AWS Setup
```bash
# Configure AWS CLI
aws configure

# Create S3 bucket for Terraform state
aws s3 mb s3://gaudi3-scale-terraform-state

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name gaudi3-scale-terraform-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

### Environment Variables

```bash
export AWS_REGION="us-west-2"
export CLUSTER_NAME="gaudi3-scale-production"
export DOMAIN="gaudi3scale.com"
export ENVIRONMENT="production"
```

## Infrastructure Deployment

### 1. Terraform Infrastructure

```bash
# Navigate to Terraform directory
cd deployment/terraform/environments/production

# Initialize Terraform
terraform init \
  -backend-config="bucket=gaudi3-scale-terraform-state" \
  -backend-config="key=production/terraform.tfstate" \
  -backend-config="region=us-west-2" \
  -backend-config="dynamodb_table=gaudi3-scale-terraform-lock"

# Plan deployment
terraform plan -var-file="production.tfvars" -out=production.tfplan

# Apply infrastructure
terraform apply production.tfplan
```

### 2. EKS Cluster Configuration

```bash
# Update kubeconfig
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 3. Install Essential Add-ons

```bash
# AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=$CLUSTER_NAME \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# EBS CSI Driver
kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"

# Cluster Autoscaler
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=$CLUSTER_NAME \
  --set awsRegion=$AWS_REGION
```

## Application Deployment

### 1. Namespace and RBAC

```bash
# Apply namespace and RBAC
kubectl apply -f security/rbac/production-rbac.yaml
kubectl apply -f deployment/k8s/base/namespace.yaml
```

### 2. Secrets Configuration

```bash
# Create secrets (replace with actual values)
kubectl create secret generic gaudi3-scale-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@host:5432/db" \
  --from-literal=REDIS_URL="redis://host:6379/0" \
  --from-literal=SECRET_KEY="your-secret-key" \
  --from-literal=JWT_SECRET="your-jwt-secret" \
  --from-literal=WANDB_API_KEY="your-wandb-key" \
  --namespace=gaudi3-scale
```

### 3. Helm Deployment

```bash
# Add Helm repository (if published)
helm repo add gaudi3-scale https://charts.gaudi3scale.com
helm repo update

# Deploy using Helm
helm install gaudi3-scale deployment/helm/gaudi3-scale \
  --namespace gaudi3-scale \
  --values deployment/helm/gaudi3-scale/values-production.yaml \
  --set image.tag=$(git rev-parse HEAD) \
  --set app.environment=production \
  --set api.ingress.enabled=true \
  --set api.ingress.hosts[0].host=api.gaudi3scale.com \
  --wait --timeout=600s
```

### 4. Verify Deployment

```bash
# Check deployment status
kubectl get pods -n gaudi3-scale
kubectl get services -n gaudi3-scale
kubectl get ingress -n gaudi3-scale

# Check application health
kubectl port-forward service/gaudi3-scale-api 8080:80 -n gaudi3-scale &
curl http://localhost:8080/health
```

## Monitoring and Observability

### 1. Prometheus and Grafana

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/production/prometheus/values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values monitoring/production/grafana/values.yaml
```

### 2. Configure Dashboards

```bash
# Import Grafana dashboards
kubectl create configmap gaudi3-scale-dashboards \
  --from-file=monitoring/production/grafana/dashboards/ \
  --namespace=monitoring

# Apply dashboard configuration
kubectl label configmap gaudi3-scale-dashboards \
  grafana_dashboard=1 \
  --namespace=monitoring
```

### 3. Set up Alerting

```bash
# Apply Prometheus rules
kubectl apply -f monitoring/production/prometheus/alerts.yml

# Configure Alertmanager
kubectl create configmap alertmanager-config \
  --from-file=monitoring/production/alertmanager/alertmanager.yml \
  --namespace=monitoring
```

## Security Configuration

### 1. Network Policies

```bash
# Apply network policies
kubectl apply -f security/policies/network-policies.yaml
```

### 2. Pod Security Standards

```bash
# Enable Pod Security Standards
kubectl label namespace gaudi3-scale \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### 3. Secrets Management

For production environments, use external secret management:

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets-system \
  --create-namespace

# Configure AWS Secrets Manager integration
kubectl apply -f security/secrets/secret-store.yaml
kubectl apply -f security/secrets/external-secrets.yaml
```

## Backup and Recovery

### 1. Configure Backup Automation

```bash
# Make backup script executable
chmod +x backup-recovery/scripts/backup-automation.sh

# Create backup CronJob
kubectl apply -f backup-recovery/manifests/backup-cronjob.yaml
```

### 2. Test Backup System

```bash
# Run manual backup
./backup-recovery/scripts/backup-automation.sh backup

# Verify backup integrity
./backup-recovery/scripts/backup-automation.sh health-check
```

### 3. Disaster Recovery Test

```bash
# Test database restore
./backup-recovery/scripts/backup-automation.sh restore-db \
  /backup/database/postgres-20240101-120000.sql.gz
```

## Operational Procedures

### Daily Operations

1. **Health Checks**
   ```bash
   # Check cluster health
   kubectl get nodes
   kubectl get pods --all-namespaces | grep -v Running
   
   # Check application health
   curl https://api.gaudi3scale.com/health
   ```

2. **Monitor Key Metrics**
   - API response times and error rates
   - HPU utilization and memory usage
   - Training job success rates
   - Resource consumption

3. **Log Review**
   ```bash
   # Check application logs
   kubectl logs -f deployment/gaudi3-scale-api -n gaudi3-scale
   kubectl logs -f deployment/gaudi3-scale-trainer -n gaudi3-scale
   ```

### Weekly Operations

1. **Security Updates**
   ```bash
   # Update base images
   docker pull vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habana-torch:latest
   
   # Rebuild and deploy
   docker build -f Dockerfile.production -t gaudi3-scale:latest .
   helm upgrade gaudi3-scale deployment/helm/gaudi3-scale \
     --set image.tag=latest
   ```

2. **Performance Review**
   - Review Grafana dashboards
   - Analyze training performance trends
   - Check resource utilization patterns

3. **Backup Verification**
   ```bash
   # Verify recent backups
   aws s3 ls s3://gaudi3-scale-backups/daily/ --recursive
   ```

### Monthly Operations

1. **Cost Optimization Review**
   ```bash
   # Check AWS costs
   aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-02-01 \
     --granularity MONTHLY --metrics BlendedCost
   ```

2. **Security Audit**
   - Review access logs
   - Update certificates
   - Review RBAC permissions

3. **Disaster Recovery Test**
   - Test backup restoration
   - Verify failover procedures

## Troubleshooting

### Common Issues

#### 1. Pods Stuck in Pending State

```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name> -n gaudi3-scale

# Check for resource constraints
kubectl top nodes
kubectl top pods -n gaudi3-scale
```

#### 2. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n gaudi3-scale --sort-by=memory

# Scale down if necessary
kubectl scale deployment gaudi3-scale-api --replicas=2 -n gaudi3-scale
```

#### 3. Training Job Failures

```bash
# Check HPU availability
kubectl exec -it <trainer-pod> -n gaudi3-scale -- hl-smi

# Check training logs
kubectl logs <trainer-pod> -n gaudi3-scale --tail=100

# Check persistent volume access
kubectl exec -it <trainer-pod> -n gaudi3-scale -- df -h /app/data
```

#### 4. API Latency Issues

```bash
# Check API metrics
curl https://api.gaudi3scale.com/metrics

# Check database connections
kubectl exec -it deployment/postgres -n database -- psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check Redis performance
kubectl exec -it deployment/redis -n cache -- redis-cli info stats
```

### Log Analysis

```bash
# Aggregated log analysis
kubectl logs -l app.kubernetes.io/name=gaudi3-scale -n gaudi3-scale --since=1h | \
  grep ERROR | wc -l

# Check for memory leaks
kubectl logs <pod-name> -n gaudi3-scale | grep -i "memory\|oom"

# Check for performance issues
kubectl logs <pod-name> -n gaudi3-scale | grep -i "slow\|timeout\|latency"
```

## Maintenance

### Planned Maintenance Windows

1. **Kubernetes Updates**
   - Schedule during low-traffic periods
   - Update worker nodes one by one
   - Test thoroughly in staging first

2. **Application Updates**
   ```bash
   # Rolling update
   helm upgrade gaudi3-scale deployment/helm/gaudi3-scale \
     --set image.tag=new-version \
     --wait
   ```

3. **Database Maintenance**
   - Schedule during maintenance windows
   - Always backup before maintenance
   - Test queries in staging first

### Emergency Procedures

1. **Scale Down for Emergency**
   ```bash
   kubectl scale deployment gaudi3-scale-api --replicas=1 -n gaudi3-scale
   kubectl scale deployment gaudi3-scale-trainer --replicas=0 -n gaudi3-scale
   ```

2. **Emergency Rollback**
   ```bash
   helm rollback gaudi3-scale -n gaudi3-scale
   ```

3. **Incident Response**
   - Check monitoring dashboards
   - Review recent deployments
   - Check infrastructure status
   - Coordinate with team via incident channel

## Performance Optimization

### CPU and Memory Optimization

```bash
# Check resource usage patterns
kubectl top pods -n gaudi3-scale --containers

# Adjust resource requests/limits
helm upgrade gaudi3-scale deployment/helm/gaudi3-scale \
  --set api.resources.requests.cpu=1000m \
  --set api.resources.requests.memory=2Gi
```

### HPU Optimization

```bash
# Check HPU utilization
kubectl exec -it <trainer-pod> -n gaudi3-scale -- hl-smi -q

# Monitor HPU metrics
kubectl port-forward service/gaudi3-scale-trainer 9200:8080 -n gaudi3-scale &
curl http://localhost:9200/hpu-metrics
```

### Storage Optimization

```bash
# Check storage usage
kubectl exec -it <pod-name> -n gaudi3-scale -- df -h

# Clean up old checkpoints
kubectl exec -it deployment/gaudi3-scale-trainer -n gaudi3-scale -- \
  find /app/models -name "*.ckpt" -mtime +7 -delete
```

## Support and Documentation

- **Internal Documentation**: `/docs` directory
- **API Documentation**: `https://api.gaudi3scale.com/docs`
- **Monitoring Dashboards**: `https://grafana.gaudi3scale.com`
- **Issue Tracking**: GitHub Issues
- **Emergency Contact**: [Your emergency contact information]

---

**Note**: This guide should be regularly updated as the platform evolves. Always test procedures in staging environments before applying to production.