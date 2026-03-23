"""Tests for bamboo.utils.sanitize."""

from bamboo.utils.sanitize import (
    SENSITIVE_TASK_KEYS,
    pseudonymise,
    pseudonymise_dict,
    sanitize_for_llm,
)

# ...existing code...


class TestJobDataSensitiveFields:
    """prodUserName and prodUserID appear in job records and must be pseudonymised."""

    def test_prod_user_name_in_sensitive_keys(self):
        assert "prodUserName" in SENSITIVE_TASK_KEYS

    def test_prod_user_id_in_sensitive_keys(self):
        assert "prodUserID" in SENSITIVE_TASK_KEYS

    def test_job_dict_pseudonymised(self):
        job = {
            "jobID": 99,
            "prodUserName": "atlas_prod",
            "prodUserID": "42",
            "computingSite": "CERN",
        }
        result = pseudonymise_dict(job)
        assert result["jobID"] == 99
        assert result["computingSite"] == "CERN"
        assert result["prodUserName"] != "atlas_prod"
        assert result["prodUserID"] != "42"
        assert "atlas_prod" not in str(list(result.values()))
        assert "42" not in str([result["prodUserName"], result["prodUserID"]])

    def test_aggregator_strips_sensitive_fields(self):
        """PandaJobDataAggregator must pseudonymise sensitive fields before aggregation."""
        from bamboo.agents.extractors.panda_job_data_aggregator import PandaJobDataAggregator

        jobs = [
            {
                "jobID": i,
                "prodUserName": "atlas_prod",
                "prodUserID": "42",
                "jobStatus": "failed",
                "computingSite": "CERN",
                "processingType": "simul",
            }
            for i in range(5)
        ]
        agg = PandaJobDataAggregator()
        result = agg.aggregate(jobs)

        # Sensitive values must not appear anywhere in the aggregated output
        all_text = (
            str(result.feature_items)
            + str(result.error_signals)
            + str(result.context_texts)
        )
        assert "atlas_prod" not in all_text
        assert "prodUserName" not in all_text or "atlas_prod" not in all_text


class TestPseudonymise:
    def test_stable(self):
        """Same input always gives same token."""
        assert pseudonymise("userName", "jsmith") == pseudonymise("userName", "jsmith")

    def test_different_values_differ(self):
        """Different values produce different tokens."""
        assert pseudonymise("userName", "jsmith") != pseudonymise("userName", "adoe")

    def test_different_fields_differ(self):
        """Same value under different field names produces different tokens."""
        assert pseudonymise("userName", "jsmith") != pseudonymise(
            "prodUserName", "jsmith"
        )

    def test_real_value_not_in_token(self):
        """The raw value must not appear in the pseudonym."""
        token = pseudonymise("userName", "jsmith")
        assert "jsmith" not in token

    def test_token_format(self):
        """Token has a readable prefix and a hex suffix."""
        token = pseudonymise("userName", "jsmith")
        parts = token.split("-")
        assert len(parts) == 2
        prefix, digest = parts
        assert len(prefix) > 0
        assert len(digest) == 8
        assert all(c in "0123456789abcdef" for c in digest)

    def test_prodUserID_prefix(self):
        # re.sub(r"(?i)(Name|ID)$", "", "prodUserID") → "prodUser" → lower → "produser"
        token = pseudonymise("prodUserID", "12345")
        assert token.startswith("produser-")

    def test_none_salt_falls_back_to_default(self):
        """Works even when config is unavailable."""
        token = pseudonymise("userName", "x")
        assert token  # just doesn't raise


class TestPseudonymiseDict:
    def test_sensitive_fields_replaced(self):
        data = {"taskID": 42, "userName": "jsmith", "site": "CERN"}
        result = pseudonymise_dict(data)
        assert result["taskID"] == 42
        assert result["site"] == "CERN"
        assert result["userName"] != "jsmith"
        assert result["userName"] == pseudonymise("userName", "jsmith")

    def test_original_not_mutated(self):
        data = {"userName": "jsmith"}
        pseudonymise_dict(data)
        assert data["userName"] == "jsmith"

    def test_none_value_skipped(self):
        data = {"userName": None, "site": "CERN"}
        result = pseudonymise_dict(data)
        assert result["userName"] is None  # None → left as-is

    def test_none_input(self):
        assert pseudonymise_dict(None) is None

    def test_extra_keys(self):
        data = {"site": "CERN", "campaign": "mc23"}
        result = pseudonymise_dict(data, extra_keys={"campaign"})
        assert result["site"] == "CERN"
        assert result["campaign"] != "mc23"

    def test_real_value_not_in_any_value(self):
        data = {"userName": "jsmith", "prodUserName": "jsmith2", "taskID": 1}
        result = pseudonymise_dict(data)
        values_str = str(list(result.values()))
        assert "jsmith" not in values_str


class TestSanitizeForLlm:
    def test_sensitive_replaced_with_redacted(self):
        data = {"taskID": 42, "userName": "jsmith", "site": "CERN"}
        result = sanitize_for_llm(data)
        assert result["userName"] == "<redacted>"
        assert result["taskID"] == 42
        assert result["site"] == "CERN"

    def test_all_default_sensitive_keys_redacted(self):
        data = {k: "value" for k in SENSITIVE_TASK_KEYS}
        result = sanitize_for_llm(data)
        for k in SENSITIVE_TASK_KEYS:
            assert result[k] == "<redacted>", f"{k} should be redacted"

    def test_pseudonymised_dict_also_redacted(self):
        """sanitize_for_llm applied on top of pseudonymise_dict still redacts."""
        data = {"userName": "jsmith", "site": "CERN"}
        pseudo = pseudonymise_dict(data)
        safe = sanitize_for_llm(pseudo)
        assert safe["userName"] == "<redacted>"
        assert safe["site"] == "CERN"

    def test_original_not_mutated(self):
        data = {"userName": "jsmith"}
        sanitize_for_llm(data)
        assert data["userName"] == "jsmith"

    def test_none_input(self):
        assert sanitize_for_llm(None) is None

    def test_extra_keys(self):
        data = {"site": "CERN", "campaign": "mc23"}
        result = sanitize_for_llm(data, extra_keys={"campaign"})
        assert result["site"] == "CERN"
        assert result["campaign"] == "<redacted>"


class TestSensitiveTaskKeys:
    def test_contains_username(self):
        assert "userName" in SENSITIVE_TASK_KEYS

    def test_contains_prod_username(self):
        assert "prodUserName" in SENSITIVE_TASK_KEYS

    def test_contains_prod_user_id(self):
        assert "prodUserID" in SENSITIVE_TASK_KEYS
