# Gaudi 3 Scale Operational Runbook

This runbook provides step-by-step procedures for common operational tasks, incident response, and maintenance activities for the Gaudi 3 Scale production environment.

## Table of Contents

1. [Emergency Response](#emergency-response)
2. [Health Check Procedures](#health-check-procedures)
3. [Scaling Operations](#scaling-operations)
4. [Deployment Procedures](#deployment-procedures)
5. [Backup and Recovery](#backup-and-recovery)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Maintenance Procedures](#maintenance-procedures)

## Emergency Response

### Severity Levels

- **P0 (Critical)**: Complete system outage, data loss
- **P1 (High)**: Major feature unavailable, significant performance degradation
- **P2 (Medium)**: Minor feature issues, moderate performance impact
- **P3 (Low)**: Cosmetic issues, minimal impact

### P0 - Critical Incident Response

#### Immediate Actions (0-5 minutes)
```bash
# 1. Acknowledge the alert
echo "P0 incident acknowledged at $(date)" >> /var/log/incidents.log

# 2. Check system status
kubectl get pods -n gaudi3-scale
kubectl get nodes

# 3. Check API health
curl -f https://api.gaudi3scale.com/health || echo "API DOWN"

# 4. Notify incident response team
./scripts/notify-incident.sh "P0" "System outage detected"

# 5. Start incident bridge
# [Coordinate via your incident management system]
```

#### Assessment Phase (5-15 minutes)
```bash
# Check infrastructure status
terraform show -json | jq '.values.root_module.resources[] | select(.type == "aws_eks_cluster") | .values.status'

# Check recent deployments
helm list -n gaudi3-scale
kubectl rollout history deployment/gaudi3-scale-api -n gaudi3-scale

# Check resource usage
kubectl top nodes
kubectl top pods -n gaudi3-scale

# Check logs for errors
kubectl logs -l app.kubernetes.io/name=gaudi3-scale -n gaudi3-scale --since=30m | grep -i error
```

#### Immediate Mitigation
```bash
# If API is down - attempt restart
kubectl rollout restart deployment/gaudi3-scale-api -n gaudi3-scale

# If resource exhaustion - scale down non-critical services
kubectl scale deployment gaudi3-scale-trainer --replicas=0 -n gaudi3-scale

# If recent deployment issue - rollback
helm rollback gaudi3-scale -n gaudi3-scale

# Enable maintenance mode (if available)
kubectl patch configmap gaudi3-scale-config \
  -p '{"data":{"maintenance_mode":"true"}}' \
  -n gaudi3-scale
```

### P1 - High Priority Response

#### Response Actions (0-30 minutes)
```bash
# 1. Assess impact
kubectl describe pods -l app.kubernetes.io/name=gaudi3-scale -n gaudi3-scale

# 2. Check specific service health
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- curl -f http://localhost:8000/health

# 3. Review monitoring dashboards
# Navigate to Grafana: https://grafana.gaudi3scale.com

# 4. Scale up if performance issue
kubectl scale deployment gaudi3-scale-api --replicas=5 -n gaudi3-scale
```

## Health Check Procedures

### Daily Health Checks

#### System Health Check Script
```bash
#!/bin/bash
# daily-health-check.sh

echo "=== Gaudi 3 Scale Daily Health Check ==="
echo "Date: $(date)"

# Kubernetes cluster health
echo "1. Cluster Health:"
kubectl get nodes --no-headers | awk '{print $1, $2}'

# Namespace resource status
echo "2. Pod Status:"
kubectl get pods -n gaudi3-scale --no-headers | \
  awk '{print $1, $3}' | grep -v Running || echo "All pods running"

# Storage health
echo "3. Storage Status:"
kubectl get pvc -n gaudi3-scale --no-headers | \
  awk '{print $1, $2, $4}'

# API health
echo "4. API Health:"
if curl -sf https://api.gaudi3scale.com/health > /dev/null; then
    echo "API: HEALTHY"
else
    echo "API: UNHEALTHY"
fi

# Database connectivity
echo "5. Database Connectivity:"
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- \
  python -c "
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print('Database: CONNECTED')
    conn.close()
except:
    print('Database: CONNECTION FAILED')
"

# Training system status
echo "6. Training System:"
training_pods=$(kubectl get pods -l app.kubernetes.io/component=trainer -n gaudi3-scale --no-headers | wc -l)
echo "Training pods: $training_pods"

echo "=== Health Check Complete ==="
```

### Application Health Endpoints

```bash
# API health check
curl https://api.gaudi3scale.com/health

# Detailed health with dependencies
curl https://api.gaudi3scale.com/health/detailed

# Readiness check
curl https://api.gaudi3scale.com/health/ready

# Metrics endpoint
curl https://api.gaudi3scale.com/metrics
```

## Scaling Operations

### Manual Scaling

#### Scale API Service
```bash
# Scale up during high load
kubectl scale deployment gaudi3-scale-api --replicas=8 -n gaudi3-scale

# Verify scaling
kubectl get pods -l app.kubernetes.io/component=api -n gaudi3-scale

# Monitor during scaling
watch kubectl top pods -n gaudi3-scale
```

#### Scale Training Workers
```bash
# Scale up for batch training
kubectl scale deployment gaudi3-scale-trainer --replicas=3 -n gaudi3-scale

# Check HPU availability
kubectl get nodes -l gaudi.habana.ai/gaudi=true

# Monitor training metrics
kubectl exec deployment/gaudi3-scale-trainer -n gaudi3-scale -- \
  curl http://localhost:8080/metrics | grep training_
```

### Auto-scaling Configuration

#### Update HPA Settings
```bash
# Modify CPU threshold
kubectl patch hpa gaudi3-scale-api-hpa -n gaudi3-scale \
  -p '{"spec":{"targetCPUUtilizationPercentage":60}}'

# Modify replica limits
kubectl patch hpa gaudi3-scale-api-hpa -n gaudi3-scale \
  -p '{"spec":{"maxReplicas":15}}'

# Check HPA status
kubectl describe hpa gaudi3-scale-api-hpa -n gaudi3-scale
```

## Deployment Procedures

### Standard Deployment

#### Pre-deployment Checklist
```bash
# 1. Verify staging deployment
helm test gaudi3-scale -n gaudi3-scale-staging

# 2. Check resource availability
kubectl describe nodes | grep -A 5 "Allocated resources"

# 3. Backup current state
helm get values gaudi3-scale -n gaudi3-scale > backup-values-$(date +%Y%m%d).yaml

# 4. Review changes
git diff HEAD~1 HEAD --name-only

# 5. Notify team
./scripts/notify-deployment.sh "Starting deployment of version $(git rev-parse HEAD)"
```

#### Deployment Process
```bash
# 1. Deploy with canary strategy
helm upgrade gaudi3-scale deployment/helm/gaudi3-scale \
  --namespace gaudi3-scale \
  --set image.tag=$(git rev-parse HEAD) \
  --set deployment.strategy=canary \
  --set deployment.canary.weight=20 \
  --wait --timeout=600s

# 2. Monitor canary deployment
kubectl get pods -l version=canary -n gaudi3-scale
kubectl logs -l version=canary -n gaudi3-scale --tail=50

# 3. Run smoke tests
curl https://api.gaudi3scale.com/health
./tests/smoke-tests.sh

# 4. Promote to full deployment
helm upgrade gaudi3-scale deployment/helm/gaudi3-scale \
  --namespace gaudi3-scale \
  --set deployment.strategy=rolling \
  --wait --timeout=600s

# 5. Verify deployment
kubectl rollout status deployment/gaudi3-scale-api -n gaudi3-scale
```

### Emergency Rollback

```bash
# 1. Identify rollback target
helm history gaudi3-scale -n gaudi3-scale

# 2. Rollback to previous version
helm rollback gaudi3-scale 1 -n gaudi3-scale

# 3. Verify rollback
kubectl get pods -n gaudi3-scale
curl https://api.gaudi3scale.com/health

# 4. Monitor after rollback
kubectl logs -f deployment/gaudi3-scale-api -n gaudi3-scale
```

## Backup and Recovery

### Manual Backup

```bash
# 1. Run full backup
./backup-recovery/scripts/backup-automation.sh backup

# 2. Verify backup completion
ls -la /backup/daily/$(date +%Y%m%d-*)

# 3. Test backup integrity
./backup-recovery/scripts/backup-automation.sh health-check

# 4. Upload to S3 (if not automatic)
aws s3 sync /backup/daily/$(date +%Y%m%d-*) s3://gaudi3-scale-backups/daily/$(date +%Y%m%d-*)
```

### Recovery Procedures

#### Database Recovery
```bash
# 1. Stop application
kubectl scale deployment gaudi3-scale-api --replicas=0 -n gaudi3-scale

# 2. Restore database
./backup-recovery/scripts/backup-automation.sh restore-db \
  /backup/database/postgres-20240101-120000.sql.gz

# 3. Verify database integrity
kubectl exec deployment/postgres -n database -- \
  psql -c "SELECT count(*) FROM training_jobs;"

# 4. Restart application
kubectl scale deployment gaudi3-scale-api --replicas=3 -n gaudi3-scale
```

#### Model Checkpoint Recovery
```bash
# 1. List available backups
aws s3 ls s3://gaudi3-scale-backups/models/

# 2. Download checkpoint backup
aws s3 cp s3://gaudi3-scale-backups/models/20240101/models-20240101.tar.gz /tmp/

# 3. Restore to training pod
kubectl cp /tmp/models-20240101.tar.gz gaudi3-scale-trainer-xxx:/app/models/
kubectl exec gaudi3-scale-trainer-xxx -n gaudi3-scale -- \
  tar xzf /app/models/models-20240101.tar.gz -C /app/models/

# 4. Verify checkpoint integrity
kubectl exec gaudi3-scale-trainer-xxx -n gaudi3-scale -- \
  python -c "import torch; print(torch.load('/app/models/latest.ckpt').keys())"
```

## Monitoring and Alerting

### Alert Response Procedures

#### High API Latency Alert
```bash
# 1. Check current latency
curl -o /dev/null -s -w "Time: %{time_total}s\n" https://api.gaudi3scale.com/health

# 2. Check API pod metrics
kubectl top pods -l app.kubernetes.io/component=api -n gaudi3-scale

# 3. Check database performance
kubectl exec deployment/postgres -n database -- \
  psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;"

# 4. Scale up if needed
kubectl scale deployment gaudi3-scale-api --replicas=6 -n gaudi3-scale

# 5. Check for improvement
# Monitor Grafana dashboard for 10 minutes
```

#### HPU Memory High Alert
```bash
# 1. Check HPU memory usage
kubectl exec deployment/gaudi3-scale-trainer -n gaudi3-scale -- hl-smi

# 2. Check for memory leaks
kubectl exec deployment/gaudi3-scale-trainer -n gaudi3-scale -- \
  python -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB')
"

# 3. Restart training pod if necessary
kubectl delete pod -l app.kubernetes.io/component=trainer -n gaudi3-scale

# 4. Monitor recovery
kubectl logs -f deployment/gaudi3-scale-trainer -n gaudi3-scale
```

#### Training Job Failed Alert
```bash
# 1. Check failed jobs
kubectl get jobs -l app.kubernetes.io/component=training -n gaudi3-scale

# 2. Get failure details
kubectl describe job <failed-job-name> -n gaudi3-scale

# 3. Check logs
kubectl logs job/<failed-job-name> -n gaudi3-scale --tail=100

# 4. Restart job if safe
kubectl delete job <failed-job-name> -n gaudi3-scale
# Redeploy job with fix
```

## Troubleshooting Guide

### Pod Issues

#### Pod Stuck in Pending
```bash
# 1. Check pod events
kubectl describe pod <pod-name> -n gaudi3-scale

# 2. Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# 3. Check for scheduling constraints
kubectl get pods -o wide -n gaudi3-scale | grep Pending

# 4. Check persistent volume availability
kubectl get pv | grep Available

# 5. Force reschedule if needed
kubectl delete pod <pod-name> -n gaudi3-scale
```

#### Pod CrashLoopBackOff
```bash
# 1. Check pod logs
kubectl logs <pod-name> -n gaudi3-scale --previous

# 2. Check resource limits
kubectl describe pod <pod-name> -n gaudi3-scale | grep -A 10 "Limits"

# 3. Check environment variables
kubectl exec <pod-name> -n gaudi3-scale -- env | sort

# 4. Debug interactively
kubectl exec -it <pod-name> -n gaudi3-scale -- /bin/bash

# 5. Check for image issues
docker pull <image-name> && docker run --rm -it <image-name> /bin/bash
```

### Network Issues

#### Service Not Accessible
```bash
# 1. Check service endpoints
kubectl get endpoints -n gaudi3-scale

# 2. Check service configuration
kubectl describe service gaudi3-scale-api -n gaudi3-scale

# 3. Test internal connectivity
kubectl run debug-pod --image=busybox -it --rm -- wget -qO- http://gaudi3-scale-api.gaudi3-scale.svc.cluster.local/health

# 4. Check ingress configuration
kubectl describe ingress -n gaudi3-scale

# 5. Check network policies
kubectl describe networkpolicy -n gaudi3-scale
```

### Performance Issues

#### High Memory Usage
```bash
# 1. Check memory usage by pod
kubectl top pods -n gaudi3-scale --sort-by=memory

# 2. Check for memory leaks
kubectl exec <pod-name> -n gaudi3-scale -- \
  python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# 3. Generate heap dump (Python)
kubectl exec <pod-name> -n gaudi3-scale -- \
  python -c "
import tracemalloc
tracemalloc.start()
# Run some operations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"

# 4. Restart if necessary
kubectl rollout restart deployment/<deployment-name> -n gaudi3-scale
```

## Maintenance Procedures

### Planned Maintenance

#### Kubernetes Cluster Update
```bash
# 1. Pre-maintenance checklist
echo "Starting maintenance at $(date)" >> /var/log/maintenance.log
kubectl get nodes --no-headers | awk '{print $1, $2, $5}'

# 2. Drain nodes one by one
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# 3. Update node
# [Perform cloud provider specific update]

# 4. Uncordon node
kubectl uncordon <node-name>

# 5. Verify node health
kubectl get nodes
kubectl describe node <node-name> | grep Conditions -A 5

# 6. Repeat for remaining nodes
# [Continue with next node]
```

#### Application Maintenance
```bash
# 1. Scale down training jobs
kubectl scale deployment gaudi3-scale-trainer --replicas=0 -n gaudi3-scale

# 2. Enable maintenance mode
kubectl patch configmap gaudi3-scale-config \
  -p '{"data":{"maintenance_mode":"true"}}' \
  -n gaudi3-scale

# 3. Perform maintenance tasks
# [Database updates, file system cleanup, etc.]

# 4. Disable maintenance mode
kubectl patch configmap gaudi3-scale-config \
  -p '{"data":{"maintenance_mode":"false"}}' \
  -n gaudi3-scale

# 5. Scale back up
kubectl scale deployment gaudi3-scale-trainer --replicas=2 -n gaudi3-scale
```

### Database Maintenance

#### Vacuum and Analyze
```bash
# 1. Check database size and performance
kubectl exec deployment/postgres -n database -- \
  psql -c "SELECT pg_size_pretty(pg_database_size('gaudi3_scale'));"

# 2. Run vacuum analyze
kubectl exec deployment/postgres -n database -- \
  psql -c "VACUUM ANALYZE;"

# 3. Check for bloated tables
kubectl exec deployment/postgres -n database -- \
  psql -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# 4. Update statistics
kubectl exec deployment/postgres -n database -- \
  psql -c "ANALYZE;"
```

### Log Rotation and Cleanup

```bash
# 1. Check log sizes
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- \
  du -sh /app/logs/*

# 2. Archive old logs
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- \
  tar czf /app/logs/archive-$(date +%Y%m%d).tar.gz /app/logs/*.log

# 3. Clean old logs
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- \
  find /app/logs -name "*.log" -mtime +7 -delete

# 4. Restart log rotation
kubectl exec deployment/gaudi3-scale-api -n gaudi3-scale -- \
  kill -USR1 1
```

## Emergency Contacts

- **On-call Engineer**: [Your on-call system]
- **Infrastructure Team**: [Infrastructure team contact]
- **Database Team**: [Database team contact]
- **Security Team**: [Security team contact]
- **Management Escalation**: [Management contact]

## Documentation Updates

This runbook should be updated whenever:
- New deployment procedures are implemented
- System architecture changes
- New monitoring alerts are added
- Incident response procedures are modified

**Last Updated**: [Current Date]
**Next Review Date**: [Review Date]