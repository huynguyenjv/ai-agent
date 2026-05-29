# Implementation Plan: Prometheus + Grafana Monitoring

**Status:** 📋 PLANNED  
**Priority:** Medium  
**Estimated effort:** 2-3 hours

---

## 1. Overview

Thêm Prometheus metrics endpoint và Grafana dashboard cho AI Agent monitoring.

```
┌─────────────┐    scrape/15s   ┌────────────┐    query    ┌─────────────┐
│  AI Agent   │ ◄────────────── │ Prometheus │ ◄────────── │   Grafana   │
│  :8080      │                 │  :9090     │             │   :3000     │
└─────────────┘                 └────────────┘             └─────────────┘
       │                              │                          │
       │ /metrics (prometheus fmt)    │ tsdb storage             │ dashboards
       │                              │ alertmanager             │ alerts
```

---

## 2. Metrics to Expose

### 2.1 Request Metrics
```python
# Counter
ai_agent_requests_total{intent, model, status}  # total requests
ai_agent_tool_calls_total{tool_name}            # tool usage

# Histogram
ai_agent_request_duration_seconds{intent}       # latency distribution
ai_agent_ttft_seconds{intent}                   # time to first token

# Gauge
ai_agent_active_requests                        # concurrent requests
ai_agent_session_count                          # active sessions
```

### 2.2 Token Metrics
```python
# Counter
ai_agent_tokens_total{type="input|output", model}

# Histogram
ai_agent_tokens_per_request{type="input|output"}
```

### 2.3 Validation Metrics
```python
# Counter
ai_agent_validation_warnings_total{warning_type}
ai_agent_validation_passed_total{intent}
ai_agent_validation_failed_total{intent}
```

### 2.4 RAG Metrics
```python
# Histogram
ai_agent_rag_chunks_per_request
ai_agent_rag_search_duration_seconds

# Counter
ai_agent_rag_cache_hits_total
ai_agent_rag_cache_misses_total
```

### 2.5 System Metrics
```python
# Gauge
ai_agent_info{version}                          # version info
ai_agent_uptime_seconds
```

---

## 3. Implementation Steps

### Step 1: Add prometheus-client dependency
```bash
# requirements.txt
prometheus-client>=0.20.0
```

### Step 2: Create metrics module
```python
# server/metrics/prometheus.py

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Counters
REQUEST_COUNT = Counter(
    'ai_agent_requests_total',
    'Total requests',
    ['intent', 'model', 'status']
)

TOOL_CALLS = Counter(
    'ai_agent_tool_calls_total',
    'Tool calls',
    ['tool_name']
)

TOKEN_COUNT = Counter(
    'ai_agent_tokens_total',
    'Tokens processed',
    ['type', 'model']
)

# Histograms
REQUEST_LATENCY = Histogram(
    'ai_agent_request_duration_seconds',
    'Request latency',
    ['intent'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

TTFT = Histogram(
    'ai_agent_ttft_seconds',
    'Time to first token',
    ['intent'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5]
)

# Gauges
ACTIVE_REQUESTS = Gauge(
    'ai_agent_active_requests',
    'Active concurrent requests'
)

# Info
VERSION_INFO = Info(
    'ai_agent',
    'AI Agent version info'
)
VERSION_INFO.info({'version': '2.0.0'})
```

### Step 3: Add metrics endpoint
```python
# server/routers/metrics.py (update existing)

from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@router.get("/metrics/prometheus")
async def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### Step 4: Instrument request handler
```python
# server/routers/chat.py (update)

from server.metrics.prometheus import (
    REQUEST_COUNT, REQUEST_LATENCY, TTFT,
    ACTIVE_REQUESTS, TOKEN_COUNT, TOOL_CALLS
)

async def _stream_response(...):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # ... existing code ...
        
        # On first token
        TTFT.labels(intent=intent).observe(ttft_seconds)
        
        # On tool call
        for tc in tool_calls:
            TOOL_CALLS.labels(tool_name=tc['function']['name']).inc()
        
        # On complete
        REQUEST_COUNT.labels(
            intent=intent,
            model=model,
            status='success'
        ).inc()
        REQUEST_LATENCY.labels(intent=intent).observe(time.time() - start_time)
        TOKEN_COUNT.labels(type='input', model=model).inc(input_tokens)
        TOKEN_COUNT.labels(type='output', model=model).inc(output_tokens)
        
    except Exception as e:
        REQUEST_COUNT.labels(intent=intent, model=model, status='error').inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()
```

### Step 5: Add Prometheus config
```yaml
# deploy/prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-agent'
    static_configs:
      - targets: ['ai-agent:8080']
    metrics_path: /metrics/prometheus

alerting:
  alertmanagers:
    - static_configs:
        - targets: []  # Add alertmanager if needed
```

### Step 6: Add Grafana dashboard
```json
// deploy/grafana/dashboards/ai-agent.json
{
  "title": "AI Agent Dashboard",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [{
        "expr": "rate(ai_agent_requests_total[5m])"
      }]
    },
    {
      "title": "Latency P95",
      "type": "graph",
      "targets": [{
        "expr": "histogram_quantile(0.95, rate(ai_agent_request_duration_seconds_bucket[5m]))"
      }]
    },
    {
      "title": "Token Usage",
      "type": "graph",
      "targets": [{
        "expr": "rate(ai_agent_tokens_total[5m])"
      }]
    },
    {
      "title": "Intent Distribution",
      "type": "piechart",
      "targets": [{
        "expr": "sum by (intent) (ai_agent_requests_total)"
      }]
    },
    {
      "title": "Tool Usage",
      "type": "bar",
      "targets": [{
        "expr": "sum by (tool_name) (ai_agent_tool_calls_total)"
      }]
    },
    {
      "title": "Active Requests",
      "type": "stat",
      "targets": [{
        "expr": "ai_agent_active_requests"
      }]
    }
  ]
}
```

### Step 7: Update docker-compose
```yaml
# docker-compose.yml (add services)

services:
  # ... existing services ...

  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./deploy/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    volumes:
      - ./deploy/grafana/provisioning:/etc/grafana/provisioning
      - ./deploy/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

volumes:
  prometheus_data:
  grafana_data:
```

---

## 4. File Structure

```
ai-agent/
├── server/
│   └── metrics/
│       ├── __init__.py
│       ├── models.py          # existing
│       ├── counter.py         # existing
│       └── prometheus.py      # NEW
│
└── deploy/
    ├── prometheus/
    │   └── prometheus.yml     # NEW
    │
    └── grafana/
        ├── provisioning/
        │   ├── datasources/
        │   │   └── prometheus.yml
        │   └── dashboards/
        │       └── default.yml
        └── dashboards/
            └── ai-agent.json  # NEW
```

---

## 5. Alerts (Optional)

```yaml
# deploy/prometheus/alerts.yml

groups:
  - name: ai-agent
    rules:
      - alert: HighErrorRate
        expr: rate(ai_agent_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(ai_agent_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 30s"

      - alert: LowTokenThroughput
        expr: rate(ai_agent_tokens_total{type="output"}[5m]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low token output rate"
```

---

## 6. Acceptance Criteria

- [ ] `/metrics/prometheus` endpoint returns valid Prometheus format
- [ ] All request metrics are tracked (count, latency, tokens)
- [ ] Tool usage metrics work
- [ ] Grafana dashboard shows real-time data
- [ ] Docker-compose starts all services
- [ ] Existing `/metrics` JSON endpoint still works

---

## 7. References

- [prometheus-client Python](https://github.com/prometheus/client_python)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
- [Prometheus Naming Conventions](https://prometheus.io/docs/practices/naming/)

---

*Plan created: 2026-05-28*
