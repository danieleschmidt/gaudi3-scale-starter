# Incident Response Playbook

Comprehensive incident response procedures for Gaudi 3 Scale infrastructure.

## Incident Classification

### Severity Levels

**P0 - Critical (< 15 min response)**
- Complete system outage
- Data loss or corruption
- Security breach
- HPU hardware failure

**P1 - High (< 1 hour response)**
- Significant performance degradation
- Training pipeline failures
- Monitoring system down
- Partial service unavailability

**P2 - Medium (< 4 hours response)**
- Minor performance issues
- Non-critical feature failures
- Documentation errors
- Dependency vulnerabilities

**P3 - Low (< 24 hours response)**
- Feature requests
- Minor bugs
- Optimization opportunities
- General inquiries

## Response Procedures

### P0 - Critical Incidents

1. **Immediate Actions (0-15 minutes)**
   ```bash
   # Check system status
   make monitor-up
   docker ps -a
   hl-smi
   
   # Stop training if necessary
   docker-compose down gaudi-trainer
   
   # Preserve logs
   docker logs gaudi3-trainer > incident-$(date +%Y%m%d-%H%M%S).log
   ```

2. **Assessment (15-30 minutes)**
   - Identify root cause
   - Assess impact scope
   - Determine recovery strategy
   - Notify stakeholders

3. **Recovery (30+ minutes)**
   - Execute recovery plan
   - Verify system functionality
   - Resume normal operations
   - Document incident

### P1 - High Priority Incidents

1. **Initial Response (0-60 minutes)**
   ```bash
   # Collect diagnostic information
   ./scripts/collect_diagnostics.sh
   
   # Check resource utilization
   htop
   df -h
   free -m
   
   # Review recent logs
   docker logs --tail 100 gaudi3-trainer
   ```

2. **Investigation**
   - Analyze logs and metrics
   - Reproduce issue if possible
   - Identify potential fixes

3. **Resolution**
   - Apply temporary workaround
   - Implement permanent fix
   - Test thoroughly
   - Deploy solution

## Common Incident Scenarios

### HPU Hardware Issues

**Symptoms:**
- `hl-smi` shows errors
- Training crashes with HPU errors
- Temperature alerts firing

**Response:**
```bash
# Check HPU status
hl-smi

# Check system logs
dmesg | grep -i habana
journalctl -u habana-driver

# Restart HPU services if needed
sudo systemctl restart habana-driver

# Verify HPU availability
python -c "import habana_frameworks.torch; print(habana_frameworks.torch.hpu.is_available())"
```

**Escalation:** Contact Habana support if hardware failure suspected.

### Training Pipeline Failures

**Symptoms:**
- Training stops unexpectedly
- Loss becomes NaN
- Out of memory errors

**Response:**
```bash
# Check training logs
docker logs gaudi3-trainer | tail -50

# Monitor resource usage
docker stats gaudi3-trainer

# Check data integrity
python -c "from gaudi3_scale.data import validate_dataset; validate_dataset()"

# Restart with reduced batch size
export BATCH_SIZE=16
docker-compose restart gaudi-trainer
```

### Performance Degradation

**Symptoms:**
- Slow training throughput
- High memory usage
- CPU bottlenecks

**Response:**
```bash
# Profile training performance
python -m cProfile -o profile.stats train.py

# Check for data loading bottlenecks
htop | grep python

# Monitor HPU utilization
watch -n 1 hl-smi

# Optimize if needed
export PT_HPU_LAZY_MODE=1
export PT_HPU_GRAPH_COMPILER_OPT_LEVEL=3
```

### Security Incidents

**Symptoms:**
- Unauthorized access attempts
- Suspicious network activity
- Security tool alerts

**Response:**
```bash
# Check access logs
sudo journalctl -u ssh | grep Failed

# Monitor network connections
ss -tuln

# Run security scan
bandit -r src/
safety check

# Check for malware
sudo rkhunter --check
```

**Escalation:** Immediately notify security team for any confirmed breaches.

## Diagnostic Tools

### System Diagnostics Script

```bash
#!/bin/bash
# scripts/collect_diagnostics.sh

DIAG_DIR="diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p $DIAG_DIR

echo "Collecting system diagnostics..."

# System information
uname -a > $DIAG_DIR/system_info.txt
uptime > $DIAG_DIR/uptime.txt
df -h > $DIAG_DIR/disk_usage.txt
free -m > $DIAG_DIR/memory_usage.txt

# HPU information
if command -v hl-smi &> /dev/null; then
    hl-smi > $DIAG_DIR/hpu_status.txt
fi

# Docker information
docker ps -a > $DIAG_DIR/containers.txt
docker images > $DIAG_DIR/images.txt
docker system df > $DIAG_DIR/docker_usage.txt

# Logs
docker logs gaudi3-trainer > $DIAG_DIR/trainer_logs.txt 2>&1
docker logs gaudi3-prometheus > $DIAG_DIR/prometheus_logs.txt 2>&1
docker logs gaudi3-grafana > $DIAG_DIR/grafana_logs.txt 2>&1

# Network information
ss -tuln > $DIAG_DIR/network.txt
ip addr show > $DIAG_DIR/interfaces.txt

# Process information
ps aux > $DIAG_DIR/processes.txt

echo "Diagnostics collected in $DIAG_DIR/"
tar -czf ${DIAG_DIR}.tar.gz $DIAG_DIR/
echo "Archive created: ${DIAG_DIR}.tar.gz"
```

### Performance Profiling

```python
# scripts/profile_training.py
import cProfile
import pstats
import io
from gaudi3_scale.trainer import GaudiTrainer

def profile_training_step():
    """Profile a single training step."""
    trainer = GaudiTrainer()
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run training step
    batch = create_mock_batch()
    loss = trainer.training_step(batch, 0)
    
    pr.disable()
    
    # Generate report
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    # Save to file
    with open('training_profile.txt', 'w') as f:
        f.write(s.getvalue())
    
    print("Profile saved to training_profile.txt")

if __name__ == "__main__":
    profile_training_step()
```

## Recovery Procedures

### Database Recovery

```bash
# Backup current state
sudo cp -r /var/lib/docker/volumes/ /backup/docker-volumes-$(date +%Y%m%d)/

# Stop services
docker-compose down

# Restore from backup
sudo cp -r /backup/docker-volumes-latest/ /var/lib/docker/volumes/

# Restart services
docker-compose up -d

# Verify recovery
make test
```

### Configuration Recovery

```bash
# Reset to last known good configuration
git checkout HEAD~1 -- docker-compose.yml
git checkout HEAD~1 -- .github/workflows/

# Restart services with old config
docker-compose down
docker-compose up -d

# Test functionality
make test-integration
```

## Post-Incident Activities

### Incident Report Template

```markdown
# Incident Report: [YYYY-MM-DD] [Brief Description]

## Summary
- **Incident ID:** INC-YYYYMMDD-001
- **Severity:** P0/P1/P2/P3
- **Start Time:** YYYY-MM-DD HH:MM UTC
- **End Time:** YYYY-MM-DD HH:MM UTC
- **Duration:** X hours Y minutes
- **Affected Systems:** [List systems]

## Impact
- **Users Affected:** X users
- **Service Degradation:** [Description]
- **Business Impact:** [Description]

## Root Cause
[Detailed root cause analysis]

## Timeline
- **HH:MM** - Initial alert received
- **HH:MM** - Investigation started
- **HH:MM** - Root cause identified
- **HH:MM** - Fix implemented
- **HH:MM** - Service restored

## Actions Taken
1. [Action 1]
2. [Action 2]
3. [Action 3]

## Lessons Learned
- [Lesson 1]
- [Lesson 2]

## Follow-up Actions
- [ ] [Action item 1] - Assigned to [Person] - Due [Date]
- [ ] [Action item 2] - Assigned to [Person] - Due [Date]
```

### Retrospective Process

1. **Schedule retrospective** within 24 hours of resolution
2. **Gather participants** - all involved team members
3. **Review timeline** and actions taken
4. **Identify improvements** in processes and tools
5. **Create action items** with owners and due dates
6. **Update documentation** and procedures

## Contacts and Escalation

### Internal Contacts
- **On-call Engineer:** [Phone/Slack]
- **Team Lead:** [Phone/Email]
- **Security Team:** [Phone/Email]
- **Infrastructure Team:** [Phone/Email]

### External Contacts
- **Habana Support:** [Support portal/Phone]
- **Cloud Provider Support:** [Portal/Phone]
- **Monitoring Service:** [Portal/Phone]

### Escalation Matrix

| Time | P0 | P1 | P2 | P3 |
|------|----|----|----|---------|
| 0-15m | On-call | On-call | - | - |
| 15m-1h | Team Lead | On-call | On-call | - |
| 1h-4h | Management | Team Lead | On-call | On-call |
| 4h+ | Executive | Management | Team Lead | On-call |

## Continuous Improvement

### Monitoring Enhancements
- Add new alerts based on incident patterns
- Improve detection accuracy
- Reduce false positive rates
- Enhance dashboard visibility

### Process Improvements
- Update runbooks based on lessons learned
- Automate common recovery procedures
- Improve communication templates
- Enhance training materials

### Tool Improvements
- Upgrade monitoring tools
- Add new diagnostic capabilities
- Improve automation scripts
- Enhance reporting systems
