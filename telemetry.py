import time
import json
import random

class SystemTracer:
    def __init__(self):
        self.trace_log = []

    def execute_with_tracing(self, task_name, simulated_fail=False, optimized=False):
        start_time = time.time()
        # Mock token usage based on whether it's optimized
        tokens = random.randint(120, 150) if optimized else random.randint(350, 400)
        trace_data = {"task": task_name, "tokens_used": tokens, "tool_calls": 3 if not optimized else 1}

        try:
            print(f"🔍 [TRACE START] Executing: {task_name}...")
            time.sleep(1.2) # Simulate network call
            
            if simulated_fail and not optimized:
                time.sleep(2.0)
                raise ConnectionError("Primary Tool Timeout (HTTP 504) - High Latency Detected")

            latency = round(time.time() - start_time, 2)
            trace_data.update({"status": "SUCCESS", "latency_sec": latency, "error": "None"})

        except Exception as e:
            latency = round(time.time() - start_time, 2)
            trace_data.update({"status": "FAILED", "latency_sec": latency, "error": str(e)})
            
        self.trace_log.append(trace_data)
        return trace_data

# --- RUNNING THE DEMONSTRATION ---
tracer = SystemTracer()

print("\n" + "="*50 + "\n PHASE 1: CONTROLLED FAILURE (BEFORE)\n" + "="*50)
run_1 = tracer.execute_with_tracing("Competitor Analysis", simulated_fail=True)
print(json.dumps(run_1, indent=2))

print("\n" + "="*50 + "\n PHASE 2: AUTO-DIAGNOSIS & ROOT CAUSE ANALYSIS\n" + "="*50)
if run_1["status"] == "FAILED":
    print("⚠️ ALERT: Trace flagged a failure.")
    print(f"🔎 Root Cause: {run_1['error']}")
    print("🛠️ System Action: Applying optimization. Switching to Fallback API and enabling response caching...")
    time.sleep(1.5)

print("\n" + "="*50 + "\n PHASE 3: OPTIMIZED EXECUTION (AFTER)\n" + "="*50)
run_2 = tracer.execute_with_tracing("Competitor Analysis", simulated_fail=True, optimized=True)
print(json.dumps(run_2, indent=2))

print("\n" + "="*50 + "\n 📊 TRACE METRICS: BEFORE VS AFTER\n" + "="*50)
print(f"Execution Time : {run_1['latency_sec']}s  -->  {run_2['latency_sec']}s")
print(f"Tool Calls     : {run_1['tool_calls']}       -->  {run_2['tool_calls']}")
print(f"Token Usage    : {run_1['tokens_used']}     -->  {run_2['tokens_used']}")
print(f"Task Success   : {run_1['status']}  -->  {run_2['status']}")