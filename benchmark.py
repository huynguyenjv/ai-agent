"""
AI Agent API Benchmark - Test performance của endpoint /v1/chat/completions

Metrics:
- Latency (min, max, avg, P50, P95, P99)
- Throughput (requests/second)
- Concurrent request handling
- Error rate
- Time to first token (streaming)

Usage:
    python benchmark.py                    # Run with defaults
    python benchmark.py --concurrent 10    # 10 concurrent requests
    python benchmark.py --requests 50      # Total 50 requests
    python benchmark.py --html             # Generate HTML report
"""

import argparse
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: int
    success: bool
    status_code: int
    latency_ms: float
    time_to_first_token_ms: Optional[float] = None
    total_tokens: int = 0
    error: Optional[str] = None
    response_preview: str = ""


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    timestamp: str
    base_url: str
    total_requests: int
    concurrent_requests: int
    # Success metrics
    successful_requests: int
    failed_requests: int
    error_rate: float
    # Latency metrics (ms)
    latency_min: float
    latency_max: float
    latency_avg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    # Time to first token (streaming)
    ttft_min: Optional[float] = None
    ttft_max: Optional[float] = None
    ttft_avg: Optional[float] = None
    ttft_p95: Optional[float] = None
    # Throughput
    total_time_seconds: float = 0
    requests_per_second: float = 0
    # Details
    results: list[RequestResult] = field(default_factory=list)


class APIBenchmark:
    """Benchmark suite for AI Agent API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Test prompts với độ phức tạp khác nhau
        self.test_prompts = [
            # Simple
            {
                "messages": [
                    {"role": "user", "content": "Generate a simple unit test for a UserService.createUser method"}
                ],
                "complexity": "simple",
            },
            # Medium
            {
                "messages": [
                    {"role": "system", "content": "You are a Java test generator."},
                    {"role": "user", "content": "Generate unit tests for OrderService with methods: createOrder, cancelOrder, getOrderById. Use JUnit5 and Mockito."}
                ],
                "complexity": "medium",
            },
            # Complex
            {
                "messages": [
                    {"role": "system", "content": "You are a senior Java developer writing comprehensive unit tests."},
                    {"role": "user", "content": """Generate complete unit tests for PaymentService:
- processPayment(PaymentRequest request) - validates, calls gateway, saves transaction
- refundPayment(UUID transactionId) - validates status, calls gateway refund
- getPaymentHistory(UUID userId) - returns paginated results
Include edge cases, error handling, and mock all dependencies."""}
                ],
                "complexity": "complex",
            },
        ]
    
    async def run(
        self,
        total_requests: int = 20,
        concurrent_requests: int = 5,
        use_streaming: bool = True,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run the benchmark."""
        if verbose:
            print("=" * 70)
            print("🚀 AI AGENT API BENCHMARK")
            print("=" * 70)
            print(f"  Target:      {self.base_url}")
            print(f"  Requests:    {total_requests}")
            print(f"  Concurrent:  {concurrent_requests}")
            print(f"  Streaming:   {use_streaming}")
            print("=" * 70)
        
        # Check health first
        if verbose:
            print("\n[1/3] Checking API health...")
        
        health_ok = await self._check_health()
        if not health_ok:
            print("❌ API health check failed! Make sure server is running on", self.base_url)
            return self._empty_result(total_requests, concurrent_requests)
        
        if verbose:
            print("  ✓ API is healthy")
        
        # Run benchmark
        if verbose:
            print(f"\n[2/3] Running {total_requests} requests ({concurrent_requests} concurrent)...")
        
        start_time = time.time()
        results = await self._run_concurrent(
            total_requests=total_requests,
            concurrent_requests=concurrent_requests,
            use_streaming=use_streaming,
            verbose=verbose,
        )
        total_time = time.time() - start_time
        
        # Calculate metrics
        if verbose:
            print("\n[3/3] Calculating metrics...")
        
        benchmark_result = self._calculate_metrics(
            results=results,
            total_requests=total_requests,
            concurrent_requests=concurrent_requests,
            total_time=total_time,
        )
        
        if verbose:
            self._print_summary(benchmark_result)
        
        return benchmark_result
    
    async def _check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def _run_concurrent(
        self,
        total_requests: int,
        concurrent_requests: int,
        use_streaming: bool,
        verbose: bool,
    ) -> list[RequestResult]:
        """Run requests with concurrency control."""
        semaphore = asyncio.Semaphore(concurrent_requests)
        results = []
        completed = 0
        
        async def make_request(request_id: int) -> RequestResult:
            nonlocal completed
            async with semaphore:
                result = await self._single_request(
                    request_id=request_id,
                    use_streaming=use_streaming,
                )
                completed += 1
                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"  [{completed:3d}/{total_requests}] {status} "
                          f"Request #{request_id:3d} - {result.latency_ms:7.1f}ms "
                          f"{'(TTFT: ' + str(int(result.time_to_first_token_ms)) + 'ms)' if result.time_to_first_token_ms else ''}")
                return result
        
        # Create all tasks
        tasks = [make_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    async def _single_request(
        self,
        request_id: int,
        use_streaming: bool,
    ) -> RequestResult:
        """Make a single request to the API."""
        # Rotate through test prompts
        prompt_data = self.test_prompts[request_id % len(self.test_prompts)]
        
        payload = {
            "model": "qwen2.5-coder",
            "messages": prompt_data["messages"],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": use_streaming,
        }
        
        start_time = time.time()
        time_to_first_token = None
        total_tokens = 0
        response_text = ""
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if use_streaming:
                    # Streaming request
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status_code != 200:
                            return RequestResult(
                                request_id=request_id,
                                success=False,
                                status_code=response.status_code,
                                latency_ms=(time.time() - start_time) * 1000,
                                error=f"HTTP {response.status_code}",
                            )
                        
                        first_chunk = True
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                if first_chunk:
                                    time_to_first_token = (time.time() - start_time) * 1000
                                    first_chunk = False
                                
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk and chunk["choices"]:
                                        delta = chunk["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        response_text += content
                                        total_tokens += 1
                                except json.JSONDecodeError:
                                    pass
                        
                        latency = (time.time() - start_time) * 1000
                        
                        return RequestResult(
                            request_id=request_id,
                            success=True,
                            status_code=200,
                            latency_ms=latency,
                            time_to_first_token_ms=time_to_first_token,
                            total_tokens=total_tokens,
                            response_preview=response_text[:100] + "..." if len(response_text) > 100 else response_text,
                        )
                else:
                    # Non-streaming request
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status_code != 200:
                        return RequestResult(
                            request_id=request_id,
                            success=False,
                            status_code=response.status_code,
                            latency_ms=latency,
                            error=f"HTTP {response.status_code}",
                        )
                    
                    data = response.json()
                    content = ""
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content", "")
                    
                    return RequestResult(
                        request_id=request_id,
                        success=True,
                        status_code=200,
                        latency_ms=latency,
                        response_preview=content[:100] + "..." if len(content) > 100 else content,
                    )
                    
        except httpx.TimeoutException:
            return RequestResult(
                request_id=request_id,
                success=False,
                status_code=0,
                latency_ms=(time.time() - start_time) * 1000,
                error="Timeout",
            )
        except Exception as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                status_code=0,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def _calculate_metrics(
        self,
        results: list[RequestResult],
        total_requests: int,
        concurrent_requests: int,
        total_time: float,
    ) -> BenchmarkResult:
        """Calculate benchmark metrics."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Latency metrics
        latencies = [r.latency_ms for r in successful]
        if not latencies:
            latencies = [0]
        
        latencies_sorted = sorted(latencies)
        
        def percentile(data: list, p: float) -> float:
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])
        
        # TTFT metrics
        ttft_values = [r.time_to_first_token_ms for r in successful if r.time_to_first_token_ms]
        ttft_sorted = sorted(ttft_values) if ttft_values else []
        
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            base_url=self.base_url,
            total_requests=total_requests,
            concurrent_requests=concurrent_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=len(failed) / total_requests * 100 if total_requests > 0 else 0,
            latency_min=min(latencies),
            latency_max=max(latencies),
            latency_avg=statistics.mean(latencies),
            latency_p50=percentile(latencies_sorted, 50),
            latency_p95=percentile(latencies_sorted, 95),
            latency_p99=percentile(latencies_sorted, 99),
            ttft_min=min(ttft_values) if ttft_values else None,
            ttft_max=max(ttft_values) if ttft_values else None,
            ttft_avg=statistics.mean(ttft_values) if ttft_values else None,
            ttft_p95=percentile(ttft_sorted, 95) if ttft_sorted else None,
            total_time_seconds=total_time,
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            results=results,
        )
    
    def _empty_result(self, total_requests: int, concurrent_requests: int) -> BenchmarkResult:
        """Return empty result when benchmark cannot run."""
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            base_url=self.base_url,
            total_requests=total_requests,
            concurrent_requests=concurrent_requests,
            successful_requests=0,
            failed_requests=total_requests,
            error_rate=100.0,
            latency_min=0,
            latency_max=0,
            latency_avg=0,
            latency_p50=0,
            latency_p95=0,
            latency_p99=0,
            total_time_seconds=0,
            requests_per_second=0,
        )
    
    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS")
        print("=" * 70)
        
        # Success/Error
        success_color = "✓" if result.error_rate < 5 else "⚠" if result.error_rate < 20 else "✗"
        print(f"\n  {success_color} Success Rate:    {100 - result.error_rate:.1f}% ({result.successful_requests}/{result.total_requests})")
        print(f"    Error Rate:      {result.error_rate:.1f}%")
        
        # Latency
        print(f"\n  ⏱️  LATENCY (ms)")
        print(f"    Min:             {result.latency_min:,.1f}")
        print(f"    Max:             {result.latency_max:,.1f}")
        print(f"    Avg:             {result.latency_avg:,.1f}")
        print(f"    P50 (median):    {result.latency_p50:,.1f}")
        print(f"    P95:             {result.latency_p95:,.1f}")
        print(f"    P99:             {result.latency_p99:,.1f}")
        
        # TTFT
        if result.ttft_avg:
            print(f"\n  🚀 TIME TO FIRST TOKEN (ms)")
            print(f"    Min:             {result.ttft_min:,.1f}")
            print(f"    Max:             {result.ttft_max:,.1f}")
            print(f"    Avg:             {result.ttft_avg:,.1f}")
            print(f"    P95:             {result.ttft_p95:,.1f}")
        
        # Throughput
        print(f"\n  📈 THROUGHPUT")
        print(f"    Total Time:      {result.total_time_seconds:.2f}s")
        print(f"    Requests/sec:    {result.requests_per_second:.2f}")
        
        # Errors
        errors = [r for r in result.results if not r.success]
        if errors:
            print(f"\n  ❌ ERRORS ({len(errors)})")
            error_types = {}
            for e in errors:
                error_types[e.error] = error_types.get(e.error, 0) + 1
            for error, count in error_types.items():
                print(f"    - {error}: {count}")
        
        print("\n" + "=" * 70)
    
    def save_json(self, result: BenchmarkResult, output_path: str = "benchmark_results.json"):
        """Save results to JSON."""
        data = {
            "timestamp": result.timestamp,
            "config": {
                "base_url": result.base_url,
                "total_requests": result.total_requests,
                "concurrent_requests": result.concurrent_requests,
            },
            "summary": {
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "error_rate_percent": result.error_rate,
                "total_time_seconds": result.total_time_seconds,
                "requests_per_second": result.requests_per_second,
            },
            "latency_ms": {
                "min": result.latency_min,
                "max": result.latency_max,
                "avg": result.latency_avg,
                "p50": result.latency_p50,
                "p95": result.latency_p95,
                "p99": result.latency_p99,
            },
            "time_to_first_token_ms": {
                "min": result.ttft_min,
                "max": result.ttft_max,
                "avg": result.ttft_avg,
                "p95": result.ttft_p95,
            } if result.ttft_avg else None,
            "requests": [
                {
                    "id": r.request_id,
                    "success": r.success,
                    "status_code": r.status_code,
                    "latency_ms": r.latency_ms,
                    "ttft_ms": r.time_to_first_token_ms,
                    "error": r.error,
                }
                for r in result.results
            ],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📄 JSON results saved to: {output_path}")
    
    def generate_html_report(self, result: BenchmarkResult, output_path: str = "benchmark_report.html"):
        """Generate HTML report."""
        
        # Prepare chart data
        latencies = [r.latency_ms for r in result.results]
        ttfts = [r.time_to_first_token_ms or 0 for r in result.results]
        request_ids = list(range(len(result.results)))
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .config-badge {{
            display: inline-block;
            background: var(--bg-card);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            margin: 0.5rem 0.25rem;
            font-size: 0.85rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .metric-card h3 {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value.success {{ color: var(--accent-green); }}
        .metric-value.warning {{ color: var(--accent-yellow); }}
        .metric-value.error {{ color: var(--accent-red); }}
        .metric-value.info {{ color: var(--accent-blue); }}
        
        .metric-detail {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .latency-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .latency-item {{
            text-align: center;
            padding: 0.75rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        
        .latency-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }}
        
        .latency-value {{
            font-size: 1.25rem;
            font-weight: 600;
        }}
        
        .chart-container {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .chart-container h3 {{
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 300px;
        }}
        
        .percentile-bar {{
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }}
        
        .percentile-label {{
            width: 60px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .percentile-track {{
            flex: 1;
            height: 24px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        
        .percentile-fill {{
            height: 100%;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            transition: width 0.5s ease;
        }}
        
        .error-list {{
            margin-top: 1rem;
        }}
        
        .error-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
        }}
        
        footer {{
            text-align: center;
            margin-top: 3rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 AI Agent Benchmark Report</h1>
            <p class="timestamp">{result.timestamp}</p>
            <div>
                <span class="config-badge">📍 {result.base_url}</span>
                <span class="config-badge">📊 {result.total_requests} requests</span>
                <span class="config-badge">⚡ {result.concurrent_requests} concurrent</span>
            </div>
        </header>
        
        <div class="metrics-grid">
            <!-- Success Rate -->
            <div class="metric-card">
                <h3>✅ Success Rate</h3>
                <div class="metric-value {'success' if result.error_rate < 5 else 'warning' if result.error_rate < 20 else 'error'}">
                    {100 - result.error_rate:.1f}%
                </div>
                <div class="metric-detail">
                    {result.successful_requests} successful / {result.failed_requests} failed
                </div>
            </div>
            
            <!-- Throughput -->
            <div class="metric-card">
                <h3>📈 Throughput</h3>
                <div class="metric-value info">{result.requests_per_second:.2f}</div>
                <div class="metric-detail">requests per second</div>
                <div class="metric-detail" style="margin-top: 0.5rem;">
                    Total time: {result.total_time_seconds:.2f}s
                </div>
            </div>
            
            <!-- Average Latency -->
            <div class="metric-card">
                <h3>⏱️ Average Latency</h3>
                <div class="metric-value {'success' if result.latency_avg < 5000 else 'warning' if result.latency_avg < 15000 else 'error'}">
                    {result.latency_avg:,.0f}ms
                </div>
                <div class="metric-detail">
                    Min: {result.latency_min:,.0f}ms / Max: {result.latency_max:,.0f}ms
                </div>
            </div>
            
            <!-- P95 Latency -->
            <div class="metric-card">
                <h3>📊 P95 Latency</h3>
                <div class="metric-value {'success' if result.latency_p95 < 10000 else 'warning' if result.latency_p95 < 30000 else 'error'}">
                    {result.latency_p95:,.0f}ms
                </div>
                <div class="metric-detail">
                    95th percentile response time
                </div>
            </div>
'''
        
        # TTFT card if available
        if result.ttft_avg:
            html += f'''
            <!-- Time to First Token -->
            <div class="metric-card">
                <h3>🚀 Time to First Token</h3>
                <div class="metric-value {'success' if result.ttft_avg < 1000 else 'warning' if result.ttft_avg < 3000 else 'error'}">
                    {result.ttft_avg:,.0f}ms
                </div>
                <div class="metric-detail">
                    P95: {result.ttft_p95:,.0f}ms
                </div>
            </div>
'''
        
        html += '''
        </div>
        
        <!-- Latency Distribution -->
        <div class="chart-container">
            <h3>📊 Latency Percentiles</h3>
'''
        
        max_latency = result.latency_max or 1
        percentiles = [
            ("P50", result.latency_p50, "#3b82f6"),
            ("P95", result.latency_p95, "#eab308"),
            ("P99", result.latency_p99, "#ef4444"),
            ("Max", result.latency_max, "#a855f7"),
        ]
        
        for label, value, color in percentiles:
            width = min(100, (value / max_latency) * 100)
            html += f'''
            <div class="percentile-bar">
                <span class="percentile-label">{label}</span>
                <div class="percentile-track">
                    <div class="percentile-fill" style="width: {width}%; background: {color};">
                        {value:,.0f}ms
                    </div>
                </div>
            </div>
'''
        
        html += '''
        </div>
        
        <!-- Latency Chart -->
        <div class="chart-container">
            <h3>📈 Latency Over Time</h3>
            <div class="chart-wrapper">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
'''
        
        # Errors section
        errors = [r for r in result.results if not r.success]
        if errors:
            error_counts = {}
            for e in errors:
                error_counts[e.error or "Unknown"] = error_counts.get(e.error or "Unknown", 0) + 1
            
            html += '''
        <div class="chart-container">
            <h3>❌ Errors</h3>
            <div class="error-list">
'''
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                html += f'''
                <div class="error-item">
                    <span>{error}</span>
                    <span style="color: var(--accent-red);">{count} occurrences</span>
                </div>
'''
            html += '''
            </div>
        </div>
'''
        
        html += f'''
        <footer>
            <p>Generated by AI Agent Benchmark Tool</p>
        </footer>
    </div>
    
    <script>
        // Latency chart
        const ctx = document.getElementById('latencyChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(request_ids)},
                datasets: [
                    {{
                        label: 'Latency (ms)',
                        data: {json.dumps(latencies)},
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.3,
                    }},
                    {{
                        label: 'TTFT (ms)',
                        data: {json.dumps(ttfts)},
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        fill: true,
                        tension: 0.3,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#94a3b8' }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Request #', color: '#94a3b8' }},
                        ticks: {{ color: '#94a3b8' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Time (ms)', color: '#94a3b8' }},
                        ticks: {{ color: '#94a3b8' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"📊 HTML report saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark AI Agent API")
    parser.add_argument("--url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--requests", "-n", type=int, default=20, help="Total number of requests")
    parser.add_argument("--concurrent", "-c", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--json", type=str, help="Save JSON results to file")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--html-output", type=str, default="benchmark_report.html", help="HTML output file")
    
    args = parser.parse_args()
    
    benchmark = APIBenchmark(
        base_url=args.url,
        timeout=args.timeout,
    )
    
    result = await benchmark.run(
        total_requests=args.requests,
        concurrent_requests=args.concurrent,
        use_streaming=not args.no_stream,
        verbose=not args.quiet,
    )
    
    if args.json:
        benchmark.save_json(result, args.json)
    
    if args.html:
        benchmark.generate_html_report(result, args.html_output)


if __name__ == "__main__":
    asyncio.run(main())



