"""
SecureAI Metrics Exporter
Exports AI system security metrics to Prometheus
"""

from prometheus_client import start_http_server, Counter, Gauge, Histogram
import random
import time
import os

# Environment variables
SIMULATE_ATTACK = os.getenv('SIMULATE_ATTACK', 'false').lower() == 'true'

# Prometheus metrics
query_counter = Counter('ai_query_total', 'Total AI queries', ['user_type'])
auth_failures = Counter('ai_auth_failures_total', 'Authentication failures')
model_accuracy = Gauge('ai_model_accuracy', 'Model accuracy')
response_time = Histogram('ai_response_time_seconds', 'API response time')
prompt_injection_attempts = Counter('ai_prompt_injection_attempts_total', 'Prompt injection attempts')
adversarial_detected = Counter('ai_adversarial_detected_total', 'Adversarial inputs detected')
data_leakage_events = Counter('ai_data_leakage_events_total', 'Data leakage events')

def generate_normal_metrics():
    """Generate normal baseline metrics"""
    # Normal query rate: 20-50 per minute
    for _ in range(random.randint(20, 50)):
        query_counter.labels(user_type='normal').inc()

    # Very few auth failures
    if random.random() < 0.1:
        auth_failures.inc()

    # High accuracy
    model_accuracy.set(random.uniform(0.92, 0.96))

    # Good response time
    response_time.observe(random.uniform(0.1, 0.3))

    # Minimal security events
    if random.random() < 0.05:
        prompt_injection_attempts.inc()

    if random.random() < 0.03:
        adversarial_detected.inc()

def generate_attack_metrics():
    """Generate metrics during simulated attack"""
    # High query rate (model extraction pattern)
    for _ in range(random.randint(200, 400)):
        query_counter.labels(user_type='attacker').inc()

    # More auth failures
    for _ in range(random.randint(5, 15)):
        auth_failures.inc()

    # Degraded accuracy
    model_accuracy.set(random.uniform(0.75, 0.85))

    # Slower response time
    response_time.observe(random.uniform(0.3, 0.8))

    # Many security events
    for _ in range(random.randint(5, 15)):
        prompt_injection_attempts.inc()

    for _ in range(random.randint(3, 10)):
        adversarial_detected.inc()

def main():
    """Main metrics generation loop"""
    # Start metrics server on port 8000
    start_http_server(8000)
    print("SecureAI Metrics Exporter started on port 8000")
    print(f"Attack simulation: {SIMULATE_ATTACK}")

    # Initial baseline
    model_accuracy.set(0.94)

    # Metrics generation loop
    while True:
        if SIMULATE_ATTACK:
            generate_attack_metrics()
        else:
            generate_normal_metrics()

        # Wait 5 seconds (matches Prometheus scrape interval)
        time.sleep(5)

if __name__ == '__main__':
    main()
