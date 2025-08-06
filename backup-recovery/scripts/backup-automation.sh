#!/bin/bash
# Gaudi 3 Scale Production Backup Automation Script
# Handles database backups, model checkpoints, configuration backups, and logs

set -euo pipefail

# Configuration
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/backup}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-gaudi3-scale-backups}"
NAMESPACE="${NAMESPACE:-gaudi3-scale}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres.database.svc.cluster.local}"
POSTGRES_DB="${POSTGRES_DB:-gaudi3_scale}"
POSTGRES_USER="${POSTGRES_USER:-gaudi3}"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${BACKUP_BASE_DIR}/backup.log"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "${BACKUP_BASE_DIR}/backup.log" >&2
    exit 1
}

# Create backup directories
setup_backup_dirs() {
    log "Setting up backup directories"
    mkdir -p "${BACKUP_BASE_DIR}"/{database,models,configs,logs,k8s}
    mkdir -p "${BACKUP_BASE_DIR}/daily/${TIMESTAMP}"
}

# Database backup
backup_database() {
    log "Starting database backup"
    local backup_file="${BACKUP_BASE_DIR}/database/postgres-${TIMESTAMP}.sql"
    
    # Create database dump
    kubectl exec -n database deployment/postgres -- pg_dump \
        -h "${POSTGRES_HOST}" \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        --no-password \
        --verbose \
        --clean \
        --if-exists \
        --format=custom > "${backup_file}"
    
    if [[ $? -eq 0 ]]; then
        log "Database backup completed: ${backup_file}"
        # Compress backup
        gzip "${backup_file}"
        log "Database backup compressed"
    else
        error "Database backup failed"
    fi
}

# Model checkpoints backup
backup_models() {
    log "Starting model checkpoints backup"
    local backup_dir="${BACKUP_BASE_DIR}/models/${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Sync model checkpoints from persistent volume
    kubectl exec -n "${NAMESPACE}" deployment/gaudi3-scale-trainer -- \
        tar czf - -C /app/models . > "${backup_dir}/models-${TIMESTAMP}.tar.gz"
    
    if [[ $? -eq 0 ]]; then
        log "Model checkpoints backup completed: ${backup_dir}"
    else
        error "Model checkpoints backup failed"
    fi
}

# Configuration backup
backup_configs() {
    log "Starting configuration backup"
    local backup_dir="${BACKUP_BASE_DIR}/configs/${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Backup Kubernetes resources
    kubectl get all,configmaps,secrets,pvc,ingress -n "${NAMESPACE}" -o yaml > \
        "${backup_dir}/k8s-resources-${TIMESTAMP}.yaml"
    
    # Backup Helm values
    helm get values gaudi3-scale -n "${NAMESPACE}" > \
        "${backup_dir}/helm-values-${TIMESTAMP}.yaml"
    
    # Backup custom configurations
    kubectl exec -n "${NAMESPACE}" deployment/gaudi3-scale-api -- \
        tar czf - -C /app/config . > "${backup_dir}/app-configs-${TIMESTAMP}.tar.gz"
    
    log "Configuration backup completed: ${backup_dir}"
}

# Logs backup
backup_logs() {
    log "Starting logs backup"
    local backup_dir="${BACKUP_BASE_DIR}/logs/${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Collect logs from all pods
    for pod in $(kubectl get pods -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}'); do
        kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers=true > \
            "${backup_dir}/${pod}-${TIMESTAMP}.log" 2>/dev/null || true
    done
    
    # Compress logs
    tar czf "${backup_dir}/pod-logs-${TIMESTAMP}.tar.gz" -C "${backup_dir}" *.log
    rm "${backup_dir}"/*.log
    
    log "Logs backup completed: ${backup_dir}"
}

# Persistent volumes backup
backup_persistent_volumes() {
    log "Starting persistent volumes backup"
    local backup_dir="${BACKUP_BASE_DIR}/volumes/${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # Get list of PVCs
    local pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}')
    
    for pvc in ${pvcs}; do
        log "Backing up PVC: ${pvc}"
        
        # Create a backup pod to access the PVC
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: backup-${pvc}-${TIMESTAMP}
  namespace: ${NAMESPACE}
spec:
  containers:
  - name: backup
    image: alpine:latest
    command: ["/bin/sh", "-c", "sleep 3600"]
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: ${pvc}
  restartPolicy: Never
EOF
        
        # Wait for pod to be ready
        kubectl wait --for=condition=Ready pod/backup-${pvc}-${TIMESTAMP} -n "${NAMESPACE}" --timeout=300s
        
        # Create backup
        kubectl exec backup-${pvc}-${TIMESTAMP} -n "${NAMESPACE}" -- \
            tar czf - -C /data . > "${backup_dir}/${pvc}-${TIMESTAMP}.tar.gz"
        
        # Clean up backup pod
        kubectl delete pod backup-${pvc}-${TIMESTAMP} -n "${NAMESPACE}"
        
        log "PVC backup completed: ${pvc}"
    done
}

# Upload to S3
upload_to_s3() {
    log "Starting upload to S3"
    
    if command -v aws &> /dev/null; then
        # Sync to S3
        aws s3 sync "${BACKUP_BASE_DIR}/daily/${TIMESTAMP}" \
            "s3://${S3_BUCKET}/daily/${TIMESTAMP}" \
            --storage-class STANDARD_IA \
            --server-side-encryption AES256
        
        # Create lifecycle policy for automated cleanup
        cat > /tmp/lifecycle-policy.json <<EOF
{
    "Rules": [{
        "ID": "gaudi3-scale-backup-lifecycle",
        "Status": "Enabled",
        "Filter": {"Prefix": "daily/"},
        "Transitions": [
            {
                "Days": 30,
                "StorageClass": "GLACIER"
            },
            {
                "Days": 90,
                "StorageClass": "DEEP_ARCHIVE"
            }
        ],
        "Expiration": {
            "Days": 2555
        }
    }]
}
EOF
        
        aws s3api put-bucket-lifecycle-configuration \
            --bucket "${S3_BUCKET}" \
            --lifecycle-configuration file:///tmp/lifecycle-policy.json
        
        log "S3 upload completed"
    else
        log "AWS CLI not available, skipping S3 upload"
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log "Starting cleanup of old backups"
    
    # Remove local backups older than retention period
    find "${BACKUP_BASE_DIR}" -type f -mtime +${RETENTION_DAYS} -delete
    find "${BACKUP_BASE_DIR}" -type d -empty -delete
    
    log "Cleanup completed"
}

# Verify backup integrity
verify_backups() {
    log "Starting backup verification"
    
    # Verify database backup
    if [[ -f "${BACKUP_BASE_DIR}/database/postgres-${TIMESTAMP}.sql.gz" ]]; then
        gunzip -t "${BACKUP_BASE_DIR}/database/postgres-${TIMESTAMP}.sql.gz"
        if [[ $? -eq 0 ]]; then
            log "Database backup verification: PASSED"
        else
            error "Database backup verification: FAILED"
        fi
    fi
    
    # Verify tar.gz files
    for file in $(find "${BACKUP_BASE_DIR}" -name "*.tar.gz" -mtime -1); do
        tar -tzf "${file}" > /dev/null
        if [[ $? -eq 0 ]]; then
            log "Archive verification PASSED: ${file}"
        else
            error "Archive verification FAILED: ${file}"
        fi
    done
    
    log "Backup verification completed"
}

# Send notification
send_notification() {
    local status=$1
    local message="Gaudi 3 Scale backup ${status} at $(date)"
    
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"${message}\"}" \
            "${SLACK_WEBHOOK_URL}" || log "Failed to send Slack notification"
    fi
    
    if [[ -n "${EMAIL_RECIPIENT:-}" ]]; then
        echo "${message}" | mail -s "Gaudi 3 Scale Backup ${status}" "${EMAIL_RECIPIENT}" || \
            log "Failed to send email notification"
    fi
}

# Main backup function
main_backup() {
    log "Starting Gaudi 3 Scale production backup"
    
    setup_backup_dirs
    
    # Run backups in parallel where possible
    backup_database &
    backup_configs &
    wait
    
    backup_models
    backup_logs
    backup_persistent_volumes
    
    verify_backups
    upload_to_s3
    cleanup_old_backups
    
    log "Gaudi 3 Scale backup completed successfully"
    send_notification "completed successfully"
}

# Restore function
restore_database() {
    local backup_file=$1
    
    if [[ ! -f "${backup_file}" ]]; then
        error "Backup file not found: ${backup_file}"
    fi
    
    log "Starting database restore from: ${backup_file}"
    
    # Decompress if needed
    if [[ "${backup_file}" == *.gz ]]; then
        gunzip -c "${backup_file}" | kubectl exec -i -n database deployment/postgres -- \
            pg_restore -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" --verbose
    else
        kubectl exec -i -n database deployment/postgres -- \
            pg_restore -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" --verbose < "${backup_file}"
    fi
    
    log "Database restore completed"
}

# Health check
health_check() {
    log "Running backup system health check"
    
    # Check required tools
    local required_tools=("kubectl" "helm" "tar" "gzip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            error "Required tool not found: ${tool}"
        fi
    done
    
    # Check Kubernetes connectivity
    kubectl cluster-info > /dev/null || error "Cannot connect to Kubernetes cluster"
    
    # Check namespace exists
    kubectl get namespace "${NAMESPACE}" > /dev/null || error "Namespace not found: ${NAMESPACE}"
    
    # Check backup directory permissions
    if [[ ! -w "${BACKUP_BASE_DIR}" ]]; then
        error "Cannot write to backup directory: ${BACKUP_BASE_DIR}"
    fi
    
    log "Health check passed"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
  backup          Run full backup (default)
  restore-db      Restore database from backup file
  health-check    Run backup system health check
  cleanup         Clean up old backups only

Options:
  --retention-days DAYS    Set backup retention period (default: 30)
  --s3-bucket BUCKET      Set S3 bucket for backups
  --namespace NAMESPACE   Set Kubernetes namespace (default: gaudi3-scale)
  --help                  Show this help message

Examples:
  $0 backup
  $0 restore-db /backup/database/postgres-20240101-120000.sql.gz
  $0 health-check
  $0 cleanup --retention-days 7

EOF
}

# Parse command line arguments
case "${1:-backup}" in
    backup)
        health_check
        main_backup
        ;;
    restore-db)
        if [[ -z "${2:-}" ]]; then
            error "Backup file path required for restore"
        fi
        restore_database "$2"
        ;;
    health-check)
        health_check
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    --help|help)
        usage
        exit 0
        ;;
    *)
        error "Unknown command: $1"
        ;;
esac