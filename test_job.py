"""Quick smoke test for the job data extension."""
from bamboo.models.graph_element import JobFeatureNode, NodeType, RelationType
from bamboo.extractors.job_data_aggregator import JobDataAggregator

assert NodeType.JOB_FEATURE.value == "Job_Feature", NodeType.JOB_FEATURE
assert RelationType.HAS_JOB_PATTERN.value == "has_job_pattern", RelationType.HAS_JOB_PATTERN

n = JobFeatureNode(
    name="computingSite=AGLT2(73%)",
    attribute="computingSite",
    value="AGLT2(73%)",
    job_count=110,
)
assert n.node_type == NodeType.JOB_FEATURE

agg = JobDataAggregator()
jobs = [
    {"jobStatus": "failed", "computingSite": "AGLT2", "errorCode": "1099",
     "errorDiag": "pilot: lost heartbeat", "cpuConsumptionTime": 3720, "pilotVersion": "pilot-3.7.2"},
    {"jobStatus": "failed", "computingSite": "AGLT2", "errorCode": "1099",
     "errorDiag": "pilot: lost heartbeat", "cpuConsumptionTime": 4200, "pilotVersion": "pilot-3.7.2"},
    {"jobStatus": "finished", "computingSite": "BNL-OSG2", "cpuConsumptionTime": 1800, "pilotVersion": "pilot-3.7.2"},
]
r = agg.aggregate(jobs)

print("feature_items:", r.feature_items)
print("error_signals:", r.error_signals)
print("component_signals:", r.component_signals)
print("context_texts:", r.context_texts)
print("total_jobs:", r.total_jobs, "failed_jobs:", r.failed_jobs)

assert r.total_jobs == 3
assert r.failed_jobs == 2
assert r.error_signals == ["1099"]
assert r.context_texts == ["pilot: lost heartbeat"]
assert ("pilot-3.7.2", "PanDA pilot") in r.component_signals

# AGLT2 dominates overall (2/3 = 66%) — expect computingSite feature
attr_names = [fi[0] for fi in r.feature_items]
assert "computingSite" in attr_names, f"expected computingSite in {attr_names}"
# AGLT2 100% failure rate → site_failure_rate feature
assert "site_failure_rate" in attr_names, f"expected site_failure_rate in {attr_names}"
# cpuConsumptionTime bucket: 3720s and 4200s → 1-6h (both); AGLT2 counts 2/3 ≥ 10%
assert "cpuConsumptionTime" in attr_names, f"expected cpuConsumptionTime in {attr_names}"

print("ALL ASSERTIONS PASSED")

