"""Temporary smoke test — delete after verifying."""
from bamboo.extractors.panda_job_data_aggregator import PandaJobDataAggregator
from bamboo.models.graph_element import JobFeatureNode, NodeType, RelationType

assert NodeType.JOB_FEATURE.value == "Job_Feature"
assert RelationType.HAS_JOB_PATTERN.value == "has_job_pattern"

agg = PandaJobDataAggregator()
r = agg.aggregate([
    {"jobStatus": "failed", "computingSite": "AGLT2", "pilotErrorCode": "1099",
     "pilotErrorDiag": "pilot: lost heartbeat", "cpuConsumptionTime": 3720},
    {"jobStatus": "failed", "computingSite": "AGLT2", "pilotErrorCode": "1099",
     "pilotErrorDiag": "pilot: lost heartbeat", "cpuConsumptionTime": 4200},
    {"jobStatus": "finished", "computingSite": "BNL-OSG2", "cpuConsumptionTime": 1800},
])
print("feature_items:", r.feature_items)
print("error_signals:", r.error_signals)
print("context_texts:", r.context_texts)
assert r.total_jobs == 3
assert r.failed_jobs == 2
assert "pilot:1099" in r.error_signals
assert "pilot: lost heartbeat" in r.context_texts
attr_names = [fi[0] for fi in r.feature_items]
assert "computingSite" in attr_names
assert "site_failure_rate" in attr_names
assert "cpuConsumptionTime" in attr_names
print("ALL OK")
