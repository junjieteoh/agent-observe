"""
Query runs example for agent-observe.

Shows how to query stored run data for analysis and debugging.
"""

from agent_observe.sinks.sqlite_sink import SQLiteSink

# Connect to the database
sink = SQLiteSink(path=".riff/observe.db")
sink.initialize()

# Query recent runs
print("=== Recent Runs ===")
runs = sink.get_runs(limit=10)
for run in runs:
    print(f"  {run['name']}: {run['status']} (risk: {run['risk_score']}, mode: {run.get('capture_mode', 'unknown')})")

# Query high-risk runs
print("\n=== High Risk Runs (score >= 50) ===")
risky = sink.get_runs(min_risk=50)
for run in risky:
    print(f"  {run['run_id'][:8]}... risk={run['risk_score']} tags={run.get('eval_tags', [])}")

# Query runs with policy violations
print("\n=== Runs with Policy Violations ===")
blocked = sink.get_runs(tag="POLICY_VIOLATION")
for run in blocked:
    print(f"  {run['name']}: {run['policy_violations']} violations")

# Get details for a specific run
if runs:
    run_id = runs[0]["run_id"]
    print(f"\n=== Details for run {run_id[:8]}... ===")

    # Get spans (tool/model calls)
    spans = sink.get_spans(run_id)
    print(f"  Spans: {len(spans)}")
    for span in spans:
        print(f"    - {span['kind']}/{span['name']}: {span['status']}")

    # Get events
    events = sink.get_events(run_id)
    print(f"  Events: {len(events)}")
    for event in events:
        print(f"    - {event['type']}")

sink.close()
