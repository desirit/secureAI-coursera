# SecureAI Monitoring Stack

Real Prometheus + Grafana setup for AI security monitoring.

## Quick Start

### Prerequisites
- Docker Desktop installed
- Ports 3000, 8000, 9090 available

### Start Stack

```bash
docker-compose up -d
```

### Access

- **Grafana:** http://localhost:3000 (admin / secureai123)
- **Prometheus:** http://localhost:9090
- **Metrics:** http://localhost:8004/metrics

### Simulate Attack

```bash
docker-compose down
SIMULATE_ATTACK=true docker-compose up -d
```

### Stop Stack

```bash
docker-compose down
```

## What's Included

- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **Metrics Exporter** - Generates AI security metrics
- **7 Alert Rules** - Automated security alerts
- **Pre-configured Dashboard** - AI security monitoring

## Metrics Available

- `ai_query_total` - Query counter by user type
- `ai_auth_failures_total` - Authentication failures
- `ai_model_accuracy` - Model accuracy gauge
- `ai_response_time_seconds` - Response time histogram
- `ai_prompt_injection_attempts_total` - Injection attempts
- `ai_adversarial_detected_total` - Adversarial inputs
- `ai_data_leakage_events_total` - Data leakage events

## Alert Rules

1. **HighQueryRate** - >100 queries/min
2. **AuthenticationFailures** - >10 failures/5min
3. **ModelAccuracyDrop** - <85% accuracy
4. **PromptInjectionAttempts** - >5 attempts/5min
5. **AdversarialInputsDetected** - >3 inputs/5min
6. **DataLeakageEvent** - Any leakage
7. **HighResponseTime** - >1 second

## Troubleshooting

**Ports in use?**
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000
lsof -i :9090
```

**No data in Grafana?**
- Check metrics: `curl http://localhost:8004/metrics`
- Check Prometheus targets: http://localhost:9090/targets
- Verify datasource in Grafana settings

**Containers not starting?**
```bash
# Check logs
docker-compose logs prometheus
docker-compose logs grafana
docker-compose logs metrics-exporter

# Restart clean
docker-compose down -v
docker-compose up -d
```

## For Demo Recording

**Start before recording:**
```bash
docker-compose up -d
# Wait 30 seconds for startup
```

**Show in demo:**
1. Open http://localhost:3000
2. Login: admin / secureai123
3. Show AI Security Monitoring dashboard
4. Optionally simulate attack

**Cleanup after:**
```bash
docker-compose down
```

## Production Use

For production deployment:
1. Change default passwords
2. Enable HTTPS
3. Set up persistent storage
4. Configure alertmanager
5. Add authentication

---

**Time to start:** ~30 seconds
**Disk space:** ~500 MB (Docker images)
**Memory:** ~1 GB (all services)
