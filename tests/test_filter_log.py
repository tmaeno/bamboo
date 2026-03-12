"""Tests for filter_log() and filter_analysis_job_brokerage_log() — log pre-filtering."""

from bamboo.utils.log_filters import (
    SOURCE_FILTER_REGISTRY,
    filter_analysis_job_brokerage_log,
    filter_log,
    filter_log_auto,
    filter_prod_job_brokerage_log,
)

# ---------------------------------------------------------------------------
# Shared brokerage-log fixture
# ---------------------------------------------------------------------------
# A minimal but structurally complete brokerage log with:
#   • data-availability lines
#   • skip lines for VP (50 % cut), memory (75 % cut), and final (100 % cut)
#   • summary funnel table
#   • a "no candidates" outcome
_BROKERAGE_LOG = """\
2026-01-01 00:00:01.000000 : start
2026-01-01 00:00:01.001000 : initial 100 candidates
2026-01-01 00:00:01.002000 : getting the list of sites where scope:dataset/ is available
2026-01-01 00:00:01.003000 :  6 sites : {}
2026-01-01 00:00:01.004000 :  scope:dataset/ is distributed
2026-01-01 00:00:01.005000 : replica_availability disk:True tape:False, is_distributed:True, rses=RSE1,RSE2
2026-01-01 00:00:01.006000 : 40 candidates have input data
2026-01-01 00:00:01.007000 : 40 candidates passed site status check
2026-01-01 00:00:01.008000 : 40 candidates passed zero share check
2026-01-01 00:00:01.009000 :   skip site=SITE_VP1/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.010000 :   skip site=SITE_VP2/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.011000 :   skip site=SITE_VP3/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.012000 :   skip site=SITE_VP4/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.013000 :   skip site=SITE_VP5/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.014000 :   skip site=SITE_VP6/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.015000 :   skip site=SITE_VP7/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.016000 :   skip site=SITE_VP8/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.017000 :   skip site=SITE_VP9/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.018000 :   skip site=SITE_VP10/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.019000 :   skip site=SITE_VP11/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.020000 :   skip site=SITE_VP12/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.021000 :   skip site=SITE_VP13/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.022000 :   skip site=SITE_VP14/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.023000 :   skip site=SITE_VP15/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.024000 :   skip site=SITE_VP16/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.025000 :   skip site=SITE_VP17/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.026000 :   skip site=SITE_VP18/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.027000 :   skip site=SITE_VP19/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.028000 :   skip site=SITE_VP20/SCORE to avoid VP criteria=-avoidvp
2026-01-01 00:00:01.029000 : 20 candidates passed for avoidVP
2026-01-01 00:00:01.030000 : 20 candidates passed for useMP=unuse
2026-01-01 00:00:01.031000 : 20 candidates passed SW/HW check
2026-01-01 00:00:01.032000 :   skip site=SITE_HI1/SCORE due to job RAM shortage. site_minmemory=3001 > job_minramcount=1760 criteria=-highmemory
2026-01-01 00:00:01.033000 :   skip site=SITE_HI2/SCORE due to job RAM shortage. site_minmemory=3001 > job_minramcount=1760 criteria=-highmemory
2026-01-01 00:00:01.034000 :   skip site=SITE_LO1/SCORE due to site RAM shortage. site_maxmemory=1000 < job_minramcount=1760 criteria=-lowmemory
2026-01-01 00:00:01.035000 :   skip site=SITE_LO2/SCORE due to site RAM shortage. site_maxmemory=1000 < job_minramcount=1760 criteria=-lowmemory
2026-01-01 00:00:01.036000 :   skip site=SITE_LO3/SCORE due to site RAM shortage. site_maxmemory=1000 < job_minramcount=1760 criteria=-lowmemory
2026-01-01 00:00:01.037000 : 5 candidates passed memory check = 1760 MBPerCore
2026-01-01 00:00:01.038000 : 5 candidates passed scratch disk check
2026-01-01 00:00:01.039000 : 5 candidates passed SE space check
2026-01-01 00:00:01.040000 : 5 candidates passed walltime check cpuTime*nEventsPerJob=0.4*100
2026-01-01 00:00:01.041000 : 5 candidates passed pilot activity check
2026-01-01 00:00:01.042000 : 5 candidates passed inclusion/exclusion
2026-01-01 00:00:01.043000 : 5 candidates passed hospital check
2026-01-01 00:00:01.044000 : 5 candidates passed for cap with gshare+resource_type check
2026-01-01 00:00:01.045000 : grand nQueue/nRunning ratio : 1.1
2026-01-01 00:00:01.046000 : sites with non-VP data : SITE_A,SITE_B
2026-01-01 00:00:01.047000 : 5 candidates passed overload check
2026-01-01 00:00:01.048000 : 5 candidates for final check
2026-01-01 00:00:01.049000 : gshare: User Analysis , task_class: 1
2026-01-01 00:00:01.050000 : getting rid of problematic site SITE_A/SCORE
2026-01-01 00:00:01.051000 :   skip site=SITE_A/SCORE  consider SITE_A unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.052000 : getting rid of problematic site SITE_B/SCORE
2026-01-01 00:00:01.053000 :   skip site=SITE_B/SCORE  consider SITE_B unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.054000 : getting rid of problematic site SITE_C/SCORE
2026-01-01 00:00:01.055000 :   skip site=SITE_C/SCORE  consider SITE_C unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.056000 : getting rid of problematic site SITE_D/SCORE
2026-01-01 00:00:01.057000 :   skip site=SITE_D/SCORE  consider SITE_D unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.058000 : getting rid of problematic site SITE_E/SCORE
2026-01-01 00:00:01.059000 :   skip site=SITE_E/SCORE  consider SITE_E unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.060000 :
2026-01-01 00:00:01.061000 : no candidates
2026-01-01 00:00:01.062000 :
2026-01-01 00:00:01.063000 : ===== Job brokerage summary =====
2026-01-01 00:00:01.064000 : data locality check: True
2026-01-01 00:00:01.065000 : the number of initial candidates: 100
2026-01-01 00:00:01.066000 :  100 ->  40 candidates,  60% cut : input data check
2026-01-01 00:00:01.067000 :   40 ->  20 candidates,  50% cut : avoid VP queue check
2026-01-01 00:00:01.068000 :   20 ->   5 candidates,  75% cut : memory check
2026-01-01 00:00:01.069000 :    5 ->   0 candidates, 100% cut : final check
2026-01-01 00:00:01.070000 : the number of final candidates: 0
2026-01-01 00:00:01.071000 :
2026-01-01 00:00:01.072000 : no candidates
"""


def _make_log(*lines: str) -> str:
    return "\n".join(lines)


class TestBrokerageFilter:
    """Tests for filter_analysis_job_brokerage_log() — the AtlasBroker-specific pre-filter."""

    # --- detection ----------------------------------------------------------

    def test_returns_none_for_generic_log(self):
        generic = "ERROR something broke\nINFO done\nERROR timeout\n"
        assert filter_analysis_job_brokerage_log(generic) is None

    def test_detects_brokerage_log(self):
        assert filter_analysis_job_brokerage_log(_BROKERAGE_LOG) is not None

    def test_empty_returns_empty_string(self):
        assert filter_analysis_job_brokerage_log("") == ""
        assert filter_analysis_job_brokerage_log("   \n  ") == ""

    # --- summary section ----------------------------------------------------

    def test_summary_always_included(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert "Job brokerage summary" in result
        assert "100 ->  40 candidates" in result or "100 ->  40" in result

    def test_all_summary_cut_lines_present(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert "input data check" in result
        assert "avoid VP queue check" in result
        assert "memory check" in result
        assert "final check" in result

    # --- data-availability lines --------------------------------------------

    def test_data_availability_included(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert "replica_availability" in result
        assert "is distributed" in result

    # --- skip-reason lines for high-impact stages ---------------------------

    def test_top_cut_skip_lines_included(self):
        # memory check (75 % cut) and final check (100 % cut) are in top-3
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        # memory-check skip lines
        assert "SITE_HI1" in result or "SITE_LO1" in result
        # final-check skip lines (bad-site / problematic)
        assert "SITE_A" in result

    def test_skip_lines_capped_per_section(self):
        # VP section (50 % cut) is the 4th highest cut; include it with top_cuts=4.
        # It has 20 skip lines; with cap=10 we expect ≤10 shown + an omitted note.
        result = filter_analysis_job_brokerage_log(
            _BROKERAGE_LOG, top_cuts=4, max_skip_lines_per_section=10
        )
        # The section was included (first VP line must appear)
        assert "SITE_VP1" in result
        # And the cap note must appear (20 - 10 = 10 omitted)
        assert "not shown" in result

    def test_low_impact_stage_not_forced_in(self):
        # "avoid VP queue check" is 50 % cut — present in top_cuts=3.
        # With top_cuts=1 only the highest (final 100 %) should appear; the VP
        # section skip lines may be absent.
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG, top_cuts=1)
        # The summary still mentions all cuts
        assert "avoid VP queue check" in result
        # But the VP-avoidance skip lines themselves need not be present
        # (this is a soft contract — we just verify no crash and summary intact)
        assert "Job brokerage summary" in result

    # --- problematic sites --------------------------------------------------

    def test_problematic_sites_included(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert "getting rid of problematic site" in result

    # --- final outcome ------------------------------------------------------

    def test_no_candidates_included(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert "no candidates" in result

    def test_use_site_included_when_present(self):
        log_with_use = _BROKERAGE_LOG.replace(
            "no candidates",
            "  use site=GOOD_SITE with weight=1.0 criteria=+use",
        )
        result = filter_analysis_job_brokerage_log(log_with_use)
        assert "GOOD_SITE" in result

    # --- compression --------------------------------------------------------

    def test_output_shorter_than_input(self):
        result = filter_analysis_job_brokerage_log(_BROKERAGE_LOG)
        assert len(result.splitlines()) < len(_BROKERAGE_LOG.splitlines())


class TestFilterLogAuto:
    """filter_log_auto() dispatches to the right specialised filter."""

    def test_dispatches_to_brokerage_filter(self):
        result = filter_log_auto(_BROKERAGE_LOG)
        # Brokerage filter output contains the structured section headers
        assert "Job brokerage summary" in result

    def test_falls_back_to_generic_for_plain_log(self):
        plain = "\n".join(
            [f"2024-01-01T00:{i:02d}:00Z DEBUG step {i}" for i in range(10)]
            + ["2024-01-01T00:10:00Z ERROR disk quota exceeded"]
        )
        result = filter_log_auto(plain)
        assert "disk quota exceeded" in result

    def test_empty_returns_empty(self):
        assert filter_log_auto("") == ""

    def test_known_source_name_routes_directly(self):
        # "analysis_job_brokerage_log" is in the registry — no detection heuristics needed.
        result = filter_log_auto(_BROKERAGE_LOG, source_name="analysis_job_brokerage_log")
        assert "Job brokerage summary" in result

    def test_unknown_source_name_falls_through_to_detection(self):
        result = filter_log_auto(_BROKERAGE_LOG, source_name="some_other_log")
        # Detection heuristics still recognise the brokerage structure.
        assert "Job brokerage summary" in result

    def test_registry_contains_expected_keys(self):
        assert "analysis_job_brokerage_log" in SOURCE_FILTER_REGISTRY
        assert "prod_job_brokerage_log" in SOURCE_FILTER_REGISTRY


# ---------------------------------------------------------------------------
# Production brokerage-log fixture (AtlasProdJobBroker style)
# ---------------------------------------------------------------------------
# Key differences vs analysis log:
#   • No data-availability lines (replica_availability / is distributed)
#   • Initial candidate count only in summary header, not in the body
#   • Production-specific criteria: -cpucore, -cache, -io, -cap_rt
_PROD_BROKERAGE_LOG = """\
2026-01-01 00:00:01.000 : start
2026-01-01 00:00:01.001 : 1591 candidates passed site status check
2026-01-01 00:00:01.002 : 1591 candidates passed zero share check
2026-01-01 00:00:01.003 :   skip site=SITE_CPU1/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.004 :   skip site=SITE_CPU2/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.005 :   skip site=SITE_CPU3/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.006 :   skip site=SITE_CPU4/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.007 :   skip site=SITE_CPU5/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.008 :   skip site=SITE_CPU6/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.009 :   skip site=SITE_CPU7/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.010 :   skip site=SITE_CPU8/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.011 :   skip site=SITE_CPU9/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.012 :   skip site=SITE_CPU10/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.013 :   skip site=SITE_CPU11/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.014 :   skip site=SITE_CPU12/ATLASPROD criteria=-cpucore
2026-01-01 00:00:01.015 : 400 candidates passed cpu core check
2026-01-01 00:00:01.016 :   skip site=SITE_IO1/ATLASPROD criteria=-io
2026-01-01 00:00:01.017 :   skip site=SITE_IO2/ATLASPROD criteria=-io
2026-01-01 00:00:01.018 :   skip site=SITE_IO3/ATLASPROD criteria=-io
2026-01-01 00:00:01.019 : 100 candidates passed io check
2026-01-01 00:00:01.020 :   skip site=SITE_CAP1/ATLASPROD criteria=-cap_rt
2026-01-01 00:00:01.021 :   skip site=SITE_CAP2/ATLASPROD criteria=-cap_rt
2026-01-01 00:00:01.022 : 50 candidates for final check
2026-01-01 00:00:01.023 : getting rid of problematic site SITE_BAD1/ATLASPROD
2026-01-01 00:00:01.024 :   skip site=SITE_BAD1/ATLASPROD unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.025 : use site=SITE_GOOD/ATLASPROD with weight=1.5 criteria=+use
2026-01-01 00:00:01.026 :
2026-01-01 00:00:01.027 : ===== Job brokerage summary =====
2026-01-01 00:00:01.028 : the number of initial candidates: 1591
2026-01-01 00:00:01.029 : 1591 ->  400 candidates,  75% cut : cpu core check
2026-01-01 00:00:01.030 :  400 ->  100 candidates,  75% cut : io check
2026-01-01 00:00:01.031 :  100 ->   50 candidates,  50% cut : cap_rt check
2026-01-01 00:00:01.032 :   50 ->   10 candidates,  80% cut : final check
2026-01-01 00:00:01.033 : the number of final candidates: 10
"""


class TestProdBrokerageFilter:
    """Tests for filter_prod_job_brokerage_log() — the AtlasProdJobBroker pre-filter."""

    # --- detection ----------------------------------------------------------

    def test_returns_none_for_generic_log(self):
        generic = "ERROR something broke\nINFO done\nERROR timeout\n"
        assert filter_prod_job_brokerage_log(generic) is None

    def test_detects_prod_brokerage_log(self):
        assert filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG) is not None

    def test_empty_returns_empty_string(self):
        assert filter_prod_job_brokerage_log("") == ""
        assert filter_prod_job_brokerage_log("   \n  ") == ""

    # --- summary section ----------------------------------------------------

    def test_summary_always_included(self):
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "Job brokerage summary" in result
        assert "cpu core check" in result
        assert "io check" in result

    # --- no data-availability section ---------------------------------------

    def test_no_data_availability_section(self):
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "## Data availability" not in result
        assert "replica_availability" not in result

    # --- skip-reason lines --------------------------------------------------

    def test_top_cut_skip_lines_included(self):
        # final check (80% cut) and cpu core / io check (75% each) are top-3
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "SITE_CPU1" in result or "SITE_IO1" in result

    def test_skip_lines_capped_per_section(self):
        # cpu core section has 12 skip lines; cap=5 → omitted note
        result = filter_prod_job_brokerage_log(
            _PROD_BROKERAGE_LOG, top_cuts=4, max_skip_lines_per_section=5
        )
        assert "SITE_CPU1" in result
        assert "not shown" in result

    # --- problematic sites and final outcome --------------------------------

    def test_problematic_sites_included(self):
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "getting rid of problematic site" in result

    def test_use_site_included(self):
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "SITE_GOOD" in result

    # --- compression --------------------------------------------------------

    def test_noise_lines_removed(self):
        # Progress milestone lines (not skip/outcome) should not appear in output.
        result = filter_prod_job_brokerage_log(_PROD_BROKERAGE_LOG)
        assert "candidates passed zero share check" not in result
        assert "candidates passed cpu core check" not in result

    # --- filter_log_auto dispatch -------------------------------------------

    def test_filter_log_auto_routes_via_source_name(self):
        result = filter_log_auto(_PROD_BROKERAGE_LOG, source_name="prod_job_brokerage_log")
        assert "Job brokerage summary" in result
        assert "## Data availability" not in result


class TestSignalExtraction:
    def test_keeps_error_lines(self):
        log = _make_log(
            "2024-01-01 00:00:01 INFO starting",
            "2024-01-01 00:00:02 ERROR brokerage failed",
            "2024-01-01 00:00:03 INFO done",
        )
        result = filter_log(log, context_lines=0)
        assert "brokerage failed" in result
        assert "starting" not in result
        assert "done" not in result

    def test_keeps_context_around_signal(self):
        log = _make_log(
            "line before before",
            "line before",
            "ERROR something bad",
            "line after",
            "line after after",
        )
        result = filter_log(log, context_lines=1)
        assert "line before" in result
        assert "line after" in result
        assert "line before before" not in result
        assert "line after after" not in result

    def test_no_signal_returns_full_log(self):
        log = _make_log("INFO starting", "INFO processing", "INFO done")
        result = filter_log(log, context_lines=0)
        # No signal lines → fall back to full log
        assert "INFO starting" in result
        assert "INFO done" in result

    def test_empty_log_returns_empty(self):
        assert filter_log("") == ""
        assert filter_log("   \n  ") == ""


class TestDeduplication:
    def test_removes_duplicate_lines(self):
        # _NOISE_RE strips 5+-digit numbers, so dataset IDs like 12345/67890/11111
        # are normalised away, collapsing all three lines to the same pattern.
        log = _make_log(
            "ERROR brokerage failed for dataset 12345",
            "ERROR brokerage failed for dataset 67890",
            "ERROR brokerage failed for dataset 11111",
        )
        result = filter_log(log, context_lines=0)
        lines = [l for l in result.splitlines() if l.strip()]
        # All three lines normalise to the same pattern → only one kept
        assert len(lines) == 1

    def test_keeps_distinct_errors(self):
        log = _make_log(
            "ERROR brokerage failed",
            "ERROR timeout waiting for pilot",
        )
        result = filter_log(log, context_lines=0)
        assert "brokerage" in result
        assert "timeout" in result

    def test_timestamps_normalised_for_dedup(self):
        log = _make_log(
            "2024-01-01T10:00:00Z ERROR disk quota exceeded",
            "2024-01-02T11:30:00Z ERROR disk quota exceeded",
        )
        result = filter_log(log, context_lines=0)
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 1


class TestTruncation:
    def test_truncates_to_max_lines(self):
        # 10 signal lines, max_lines=5 → head(2) + omitted + tail(3)
        lines = [f"ERROR error number {i}" for i in range(10)]
        log = "\n".join(lines)
        result = filter_log(
            log, context_lines=0, max_lines=5, head_lines=2, tail_lines=3
        )
        result_lines = result.splitlines()
        assert any("omitted" in l for l in result_lines)
        assert "error number 0" in result
        assert "error number 9" in result

    def test_no_truncation_when_under_limit(self):
        lines = [f"ERROR error {i}" for i in range(5)]
        log = "\n".join(lines)
        result = filter_log(log, context_lines=0, max_lines=100)
        assert "omitted" not in result


class TestReduction:
    def test_large_log_reduces_significantly(self):
        """Simulate a realistic noisy log — should reduce by at least 50%."""
        noise = [
            f"2024-01-01T00:{i:02d}:00Z DEBUG processing item {i}" for i in range(200)
        ]
        signal = [
            "2024-01-01T00:05:00Z ERROR brokerage failed for 3 input datasets",
            "2024-01-01T00:05:01Z ERROR no candidates found at any site",
        ]
        import random

        random.seed(42)
        all_lines = noise + signal
        random.shuffle(all_lines)
        log = "\n".join(all_lines)

        result = filter_log(log, context_lines=2)
        result_lines = [l for l in result.splitlines() if l.strip()]
        assert len(result_lines) < len(all_lines) * 0.5
        assert "brokerage failed" in result
        assert "no candidates" in result
