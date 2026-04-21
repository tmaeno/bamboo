"""LLM prompt templates for all pipeline stages.

All prompts are plain strings with ``{placeholder}`` format fields.
Call ``.format(...)`` to produce the final prompt before passing to the LLM.

Prompt constants
----------------
``EXTRACTION_PROMPT``
    General-purpose knowledge-graph extraction from raw text.
    Used by ``LLMExtractionStrategy``.

``EMAIL_EXTRACTION_PROMPT``
    Extracts ``Cause``, ``Resolution``, ``Task_Context``, and ``Procedure`` nodes
    from an email thread.  Used by :class:`~bamboo.agents.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`.

``LOG_EXTRACTION_PROMPT``
    Extracts ``Symptom``, ``Component``, and ``Task_Context`` nodes from raw
    log text.  Cause and Resolution are intentionally excluded — logs record
    what happened, not why or how to fix it; those come from the email thread
    and graph reasoning.  The prompt de-duplicates repeated occurrences of the
    same error, captures the components named in stack traces or log prefixes,
    and stores the surrounding prose as Task_Context for vector search.

``BROKERAGE_LOG_EXTRACTION_PROMPT``
    Extracts a ``Symptom`` (placement outcome: ``BrokerageNoCandidates`` /
    ``BrokerageCandidateFound``), ``Task_Feature`` nodes for the structured
    task constraints visible in the log (memory requirement, IO intensity,
    data locality, etc.), and ``Task_Context`` nodes for the dominant filter
    stages.  Reuses existing node types; no brokerage-specific types needed.

``CAUSE_RESOLUTION_CANONICALIZE_PROMPT``
    Normalises a raw Cause/Resolution name into a stable canonical phrase,
    optionally matching against existing names from the vector DB.

``TASK_ERROR_CATEGORY_LABEL_PROMPT``
    Converts a raw error message into a short CamelCase error-category label.
    Used by :class:`~bamboo.agents.extractors.panda_knowledge_extractor.ErrorCategoryStore`.

``SUMMARIZATION_PROMPT``
    Produces a narrative summary of a knowledge graph for vector indexing.
    Used by :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`.

``CAUSE_IDENTIFICATION_PROMPT``
    Root-cause analysis from graph and vector DB evidence.
    Used by :class:`~bamboo.agents.reasoning_navigator.ReasoningNavigator`.

``EMAIL_GENERATION_PROMPT``
    Drafts a professional resolution email for a task submitter.
    Used by :class:`~bamboo.agents.reasoning_navigator.ReasoningNavigator`.

``PRESCRIPTION_CLASSIFY_PROMPT``
    Classifies a resolution into an action type (resubmit, contact_admin, etc.) and
    decides whether CLI documentation is needed.
    Used by :class:`~bamboo.agents.prescription_composer.PrescriptionComposer`.

``PRESCRIPTION_COMPOSE_PROMPT``
    Composes concrete action steps (prescription) given the analysis and any fetched
    resources (e.g. CLI docs).
    Used by :class:`~bamboo.agents.prescription_composer.PrescriptionComposer`.

``PRESCRIPTION_EMAIL_PROMPT``
    Drafts a resolution email combining root-cause analysis and prescription hints.
    Used by :class:`~bamboo.agents.email_drafter.EmailDrafter`.

``JOB_DIAG_NORMALIZE_PROMPT``
    Strips job-specific tokens (file names, dataset scopes, job IDs, replica
    URLs, etc.) from a raw error diagnostic string, returning a reusable
    canonical description of the *type* of problem.
    Used by :class:`~bamboo.agents.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`
    before storing job error diagnostics as ``TaskContextNode``.

``EXPLORER_TOOL_SELECTION_PROMPT``
    Given a reviewer's issue list and available MCP tools, selects which
    tools to call to fill data gaps before a re-extraction attempt.
    Used by :class:`~bamboo.agents.extra_source_explorer.ExtraSourceExplorer`.

``KNOWLEDGE_REVIEW_PROMPT``
    Reviews an extracted knowledge graph for completeness, accuracy, and
    consistency against the original sources.  Returns a structured JSON
    verdict with issues and corrective feedback.
    Used by :class:`~bamboo.agents.knowledge_reviewer.KnowledgeReviewer`.

``DOC_SEARCH_KEYWORDS_PROMPT``
    Extracts 2-5 focused search terms from a task's errorDialog and operator
    email so that ``search_panda_docs`` retrieves the most relevant sections.
    Used by :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`.
"""

EXTRACTION_PROMPT = """You are a knowledge extraction expert. Your task is to extract structured knowledge from the provided information and construct a knowledge graph.

The graph should follow this schema:
- Symptom -[indicate]-> Cause
- Environment -[associated_with]-> Cause
- Task_Feature -[contribute_to]-> Cause
- Task_Context -[contribute_to]-> Cause
- Component -[originated_from]-> Cause
- Cause -[solved_by]-> Resolution

Node Types:
- Symptom: Main symptom message
- Environment: External factor (system, network, resource)
- Task_Feature: A task characteristic with a discrete, comparable value.
  MUST be formatted as "attribute=value" (e.g. "RAM=1GB", "OS=Ubuntu 22.04", "timeout=30s").
  Use this when the value is a concrete, canonical unit — something you could sort or compare.
  Include "attribute" and "value" as separate fields in metadata.
- Task_Context: A task characteristic whose value is free-form prose that
  cannot be meaningfully canonicalized.
  Use this for reproduction steps, user descriptions, free-text comments, etc.
  name = the attribute key only (e.g. "steps_to_reproduce", "user_report").
  description = the full unstructured prose value.
  Rule of thumb: if the value could appear in a dropdown or unit list → Task_Feature.
                 If the value is a sentence or paragraph → Task_Context.
- Component: Origin of the cause (system component, service, etc.)
- Cause: Root cause of the issue
- Resolution: Action to resolve the issue

Canonicalization rules (apply during extraction — no separate pass needed):
- Symptom, Environment, Component: output a single, normalised canonical name.
  Prefer full names over abbreviations ("production" not "prod",
  "NullPointerException" not "NPE", "authentication-service" not "auth-svc").
  If the same concept appears multiple times under different surface forms,
  emit it only once under its canonical name.
- Task_Feature: name is already canonical by the "attribute=value" format.
- Task_Context: each instance is unique — never deduplicate.
- Cause, Resolution: write clearly and completely; they are canonical by nature.

Input Information:
{input_data}

Extract and structure the information into nodes and relationships. For each node, provide:
- name: canonical name (see rules above)
- description: detailed description (Task_Context: the full prose value)
- relevant metadata (Task_Feature: include "attribute" and "value" keys)

Output your response as a valid JSON with the following structure:
{{
  "nodes": [
    {{
      "node_type": "Symptom|Environment|Task_Feature|Task_Context|Component|Cause|Resolution",
      "name": "...",
      "description": "...",
      "metadata": {{}}
    }}
  ],
  "relationships": [
    {{
      "source_name": "...",
      "target_name": "...",
      "relation_type": "indicate|associated_with|contribute_to|originated_from|solved_by",
      "confidence": 0.0-1.0
    }}
  ]
}}
"""


SUMMARIZATION_PROMPT = """You are a technical documentation expert. Create a concise summary of the following knowledge graph and email thread.

Knowledge Graph:
{graph_data}

Email Thread (if available):
{email_text}

TERMINOLOGY REFERENCE (use only to interpret technical terms already present in the graph or email — do NOT introduce new causes, resolutions, procedures, or commands from this section):
{doc_hints}

STRICT RULE: Only use information explicitly present in either the Knowledge Graph or the Email Thread above.
Do NOT invent, infer, or supplement with domain knowledge — not extra causes, not example commands, not additional context.

Create a summary that:
1. Highlights the causes and their descriptions, exactly as they appear in the graph
2. Describes the key contributing factors (environment, features, components) from the graph
3. Outlines the resolutions from the graph
4. Describes any investigation procedures from the graph or email thread (steps taken, findings, communications with admins)

The summary should be:
- Clear and technical
- Faithful to the graph and email — no hallucinated content
- Actionable (highlight what matters for troubleshooting)

Summary:
"""

CAUSE_IDENTIFICATION_PROMPT = """You are a root cause analysis expert. Analyze the task information and database results to identify the most likely root cause.

Task Information:
{task_info}

External Information:
{external_info}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

Use the domain documentation to interpret graph evidence correctly and to weight causes
that align with the documented failure pattern for this task status/error more highly.

Graph Database Results (possible causes and resolutions):
{graph_results}

Vector Database Results (similar cases):
{vector_results}

Based on this information:
1. Identify the most likely root cause
2. Assess your confidence level (0-1)
3. Recommend the best resolution
4. Explain your reasoning — when external_info contains job-level data, cite
   specific job IDs and metric values to support the identified cause

IMPORTANT: If a matching cause appears in the Graph Database Results, set "root_cause"
to the exact "cause_name" string from that result — character for character. Do not
paraphrase or reformulate it. This exact match is required for procedure lookup.

Respond with a valid JSON object — no markdown fences, no explanation:
{{
  "root_cause": "...",
  "confidence": 0.0-1.0,
  "resolution": "...",
  "reasoning": "...",
  "supporting_evidence": [
    {{
      "source": "graph|vector",
      "evidence": "...",
      "relevance": "..."
    }}
  ]
}}
"""

EMAIL_GENERATION_PROMPT = """You are a technical communication expert. Generate a clear, professional email explaining the root cause and resolution to the user.

Task ID: {task_id}
Task Description: {task_description}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

Use the domain documentation to ensure the resolution steps and explanations align with
documented PanDA system behaviour for this task status/error.

Root Cause Analysis:
{analysis}

The email should:
1. Start with a brief acknowledgment of the issue
2. Explain the root cause in clear, non-technical language when possible
3. Provide step-by-step resolution instructions
4. When the analysis contains investigation data with specific job IDs or measured
   values, reference them to illustrate the problem concretely
5. Offer to help if they have questions
6. Be professional but friendly

Generate the email content:
"""

FEATURE_EXTRACTION_PROMPT = """You are a data extraction expert. Extract key features, errors, and contextual information from the following task and external data.

Task Object:
{task_data}

External Information:
{external_data}

Extract:
1. Error messages or error indicators
2. Key features or characteristics of the task
3. Environmental factors
4. Component or system identifiers
5. Any other relevant contextual information

Respond with a JSON object:
{{
  "errors": ["..."],
  "features": ["..."],
  "environment_factors": ["..."],
  "components": ["..."],
  "context": {{}}
}}
"""

EMAIL_EXTRACTION_PROMPT = """You are a root-cause analysis expert reading an operational incident email thread.

Extract ONLY the following node types from the email — do NOT emit Symptom, Environment, Task_Feature, or Component nodes (those are populated from structured data elsewhere):

Node types to extract:
- Cause: A root cause or contributing cause mentioned or implied in the thread.
  name = short canonical description of the cause.
  description = fuller explanation as written in the email.
- Resolution: A fix, workaround, or recommended action mentioned in the thread.
  name = short canonical description of the action.
  description = fuller explanation as written in the email.
  steps = list of discrete action steps if mentioned (else empty list).
- Task_Context: Any free-form contextual remark that does not fit Cause or Resolution —
  e.g. observations, background, constraints, timeline notes.
  name = a short snake_case key summarising the topic (e.g. "observed_behavior", "timeline").
  description = the verbatim or lightly paraphrased prose from the email.
- Procedure: An investigation action taken in response to a cause — describes how the
  incident was investigated (e.g. "we looked at finished jobs with high CPU time",
  "we checked the parent task for data corruption").
  Extract a Procedure ONLY when the email explicitly describes what was investigated
  and how, not just that an investigation took place.
  name = "{{strategy_type}}:{{cause_name}}" — a stable merge key combining the strategy
         and the canonical cause name.
  strategy_type = concise natural-language description of the investigation.
                  Must be specific enough to guide tool selection.
  description = full prose from the email describing what was investigated and why.
  parameters = structured filter and metric details as a dict, e.g.:
               {{"filter": {{"status": "finished", "jobType": "normal"}},
                "metrics": ["cpuConsumptionTime", "wallTime"]}}
               Omit keys that are not mentioned in the email.

Relationships to emit (between extracted nodes only):
- Cause -[solved_by]-> Resolution
- Cause -[investigated_by]-> Procedure  (link each Procedure to its Cause)
- Task_Context -[contribute_to]-> Cause  (only when the context directly supports a cause)

Canonicalization:
- If the same cause or resolution is mentioned multiple times, emit it only once.
- Keep names concise and general (no incident-specific IDs, paths, or dataset names).

Domain knowledge (PanDA system — for context and filling in MISSING details only;
do NOT replace or override terminology that is already present in the source text):
{doc_hints}

Email thread:
{email_text}

Output ONLY a valid JSON object — no explanation, no markdown fences:
{{
  "nodes": [
    {{
      "node_type": "Cause|Resolution|Task_Context|Procedure",
      "name": "...",
      "description": "...",
      "metadata": {{}},
      "steps": [],
      "strategy_type": "",
      "parameters": {{}}
    }}
  ],
  "relationships": [
    {{
      "source_name": "...",
      "target_name": "...",
      "relation_type": "solved_by|contribute_to|investigated_by",
      "confidence": 0.0
    }}
  ]
}}
"""

EMAIL_INVESTIGATION_SUMMARY_PROMPT = """You are reading an incident email thread.
Identify any investigation steps that were explicitly described — what was examined,
checked, or queried, and what was found or concluded.

List each step briefly (one line each).  Only report steps that are explicitly stated
in the email — do NOT infer or speculate.  If no investigation steps are described,
respond with exactly: none

Email thread:
{email_text}

Investigation steps:"""

CAUSE_RESOLUTION_CANONICALIZE_PROMPT = """You are a knowledge-graph canonicalization expert.

You will be given a short name for a Cause or Resolution node extracted from an incident report,
together with the list of names that already exist in the knowledge graph for that node type.

Your task:
1. If the raw name describes the same concept as one of the existing names, return that existing
   name EXACTLY — character for character, including spacing and capitalisation.
2. Only if no existing name is a good match, coin a new canonical name following the rules below.

Rules for a new canonical name:
- Remove ALL incident-specific tokens: dataset names, file paths, usernames, version strings,
  numeric IDs, timestamps, job names, and any other values that differ between incidents.
- Normalise the phrasing to a consistent, general form.
- Use lowercase with spaces. 3–8 words is ideal.
- Do NOT add punctuation at the end.
- Do NOT wrap the answer in quotes or add any explanation — output the canonical name only.

Node type: {node_type}

Existing names in the graph ({node_type}):
{existing_names}

Examples for Cause:
  raw: "input dataset mc20_13TeV contains too many files (>200000)"
  existing: ["input dataset exceeds file limit", "database connection refused"]
  → input dataset exceeds file limit          ← matched existing

  raw: "segmentation fault in worker process PID 4821"
  existing: ["input dataset exceeds file limit", "database connection refused"]
  → worker process segmentation fault         ← no match, new name coined

Examples for Resolution:
  raw: "split mc20_13TeV dataset into subsets < 200k files"
  existing: ["split dataset into smaller subsets", "restart service"]
  → split dataset into smaller subsets        ← matched existing

  raw: "patch libssl to version 3.1.2"
  existing: ["split dataset into smaller subsets", "restart service"]
  → update ssl library version                ← no match, new name coined

Raw name: {raw_name}

Canonical name:"""

TASK_ERROR_CATEGORY_LABEL_PROMPT = """You are an error classification expert.

Given a raw error message from an operational system, produce a short, canonical error-category label.

Rules:
- Strip ALL incident-specific tokens: dataset names, file paths, usernames, version strings, numeric IDs, timestamps, URLs, and any other values that differ between incidents.
- Focus on the STRUCTURAL PATTERN of the error — what kind of failure is being described in general terms.
- Return a single CamelCase label of 1–4 words (e.g. "TooManyFiles", "ConnectionTimeout", "PermissionDenied", "DiskFull").
- Do NOT include punctuation, spaces, underscores, or numbers in the label.
- Do NOT wrap the answer in quotes or add any explanation — output the label only.

Examples:
  "failed to insert files for mc20_13TeV:mc20_13TeV.900149.PG_single_nu_Pt50.digit.RDO.e8307_s3482_s3136_d1715/. Input dataset contains too many files >200000."
  → TooManyFilesInDataset

  "failed to insert files for user.skondo.Znnjets_mc20_700337_mc20e.eventpick.AOD.DTRun3_v1.2.15.log. Input dataset contains too many files >200000."
  → TooManyFilesInDataset

  "Connection refused: could not connect to host db-prod-3.internal on port 5432 after 3 retries."
  → DatabaseConnectionRefused

  "java.lang.OutOfMemoryError: Java heap space at com.example.Processor.run(Processor.java:42)"
  → OutOfMemoryError

Raw error message:
{error_message}

Category label:"""

ERROR_CATEGORY_MATCH_PROMPT = """You are an error classification expert.

Does the following error message fit the error category "{category}"?

Error message:
{error_message}

Answer with exactly "yes" or "no". No explanation."""


LOG_EXTRACTION_PROMPT = """You are a log analysis expert reading raw operational log output from a distributed computing job.

Your goal is to extract reusable, incident-agnostic knowledge from the log — the kind that helps identify recurring failure patterns across many jobs.

Extract ONLY the following node types — do NOT emit Cause or Resolution nodes (those are determined from email threads and graph reasoning, not from logs alone):

Node types to extract:

- Symptom: A distinct class of error or failure signal observed in the log.
  name = a short CamelCase label describing the error pattern, stripped of all
         incident-specific tokens (dataset names, file paths, job IDs, timestamps,
         hostnames, numeric IDs).  Use the same canonicalisation rules as
         TASK_ERROR_CATEGORY_LABEL_PROMPT — the same structural error in two different
         jobs must produce the same Symptom name.
  description = one representative log line that best illustrates the error,
                lightly redacted to remove incident-specific tokens.
  severity = "critical" | "error" | "warning" | "info"  (infer from log level or context)
  If the same error pattern appears many times, emit it ONCE with a count in metadata.

- Component: A named software component, service, worker, or library identified
  in the log as the origin of an error (stack frame module, daemon name, plugin,
  executor name, etc.).
  name = canonical component name (normalise: drop version numbers, instance IDs,
         and host-specific suffixes; e.g. "pilot" not "pilot-2.7.3-worker-42").
  system = the broader system it belongs to, if determinable (e.g. "PanDA", "Athena").
  Do NOT emit a Component for generic OS-level things (kernel, libc) unless they
  are clearly the origin of the error.

- Task_Context: A short passage from the log that provides useful context for
  understanding the failure but does not fit Symptom or Component — e.g. the
  sequence of steps leading up to an error, a resource exhaustion trace, a
  configuration dump, or a timing observation.
  name = a short snake_case key summarising the topic
         (e.g. "resource_exhaustion_trace", "retry_sequence", "config_at_failure").
  description = the relevant log excerpt, lightly cleaned (remove timestamps and
                hostnames but keep the substance).
  Emit at most 3–5 Task_Context nodes — choose the passages most useful for
  semantic similarity search across incidents.  Do NOT dump the entire log.

Relationships to emit (between extracted nodes only):
- Component -[originated_from]-> Symptom  (when the component is clearly the source)
- Task_Context -[contribute_to]-> Symptom  (when the context directly precedes or explains a symptom)

Canonicalization:
- If the same Symptom or Component appears multiple times, emit it only once.
- Strip ALL incident-specific tokens from names: no dataset names, file paths,
  usernames, job IDs, PIDs, timestamps, IP addresses, or hostnames.

Log text:
{log_text}

Output ONLY a valid JSON object — no explanation, no markdown fences:
{{
  "nodes": [
    {{
      "node_type": "Symptom|Component|Task_Context",
      "name": "...",
      "description": "...",
      "severity": "critical|error|warning|info|null",
      "system": "...|null",
      "metadata": {{
        "occurrence_count": 1
      }}
    }}
  ],
  "relationships": [
    {{
      "source_name": "...",
      "target_name": "...",
      "relation_type": "originated_from|contribute_to",
      "confidence": 0.0
    }}
  ]
}}
"""

BROKERAGE_LOG_EXTRACTION_PROMPT = """You are an expert in distributed computing and grid job scheduling reading a pre-filtered PanDA job-brokerage log.

The log shows the result of a site-selection (brokerage) run.  The broker itself ran correctly.
Your goal is to extract reusable, task-agnostic knowledge so that recurring placement patterns
become visible across many tasks.

Extract ONLY the following node types:

- Symptom: The final placement outcome.
  name = "BrokerageNoCandidates" | "BrokerageCandidateFound"
  description = the "no candidates" or "selected site=X" line (lightly redacted).
  severity = "critical" if no site was found, "info" otherwise.
  metadata.initial_candidates = integer (from summary header, if present)
  metadata.final_candidates = integer (from summary footer, if present)
  Emit exactly ONE Symptom node.

- Task_Feature: A structured, comparable task constraint visible in the log that contributed
  to the placement outcome.  Use the canonical "attribute=value" format for name.
  Extract only constraints the log makes explicit — do not infer.
  Examples (use these exact attribute names where applicable):
    memory_requirement=<low|medium|high>   (from job_minramcount; low<1000MB, medium<4000MB, high>=4000MB)
    io_intensity=<low|high>                (from "IO intensity N"; high if N > 500)
    data_locality=<single_site|few_sites|well_distributed>  (from "available at N sites"; single if N<=2)
    input_availability=<complete|partial>  (from missing-files check; partial if any files missing)
    cpu_cores=<N>                          (from core mismatch lines, task side only)
  Strip all numeric thresholds and site names from the name — keep only the category value.
  Emit one Task_Feature per distinct constraint.  Omit constraints not mentioned in the log.

- Task_Context: Free-form context about the dominant filter stages for vector search.
  Emit 1–3 nodes for the stages with the largest % cut.
  name = snake_case, e.g. "io_check_bottleneck", "memory_check_bottleneck".
  description = the summary line showing the candidate drop, lightly redacted.

Relationships to emit:
- Task_Feature -[contribute_to]-> Symptom
- Task_Context -[contribute_to]-> Symptom

Canonicalization:
- Strip ALL incident-specific tokens from names: no dataset names, site names, job IDs,
  numeric thresholds, or timestamps.

Brokerage log:
{log_text}

Output ONLY a valid JSON object — no explanation, no markdown fences:
{{
  "nodes": [
    {{
      "node_type": "Symptom|Task_Feature|Task_Context",
      "name": "...",
      "description": "...",
      "severity": "critical|warning|info|null",
      "attribute": "...|null",
      "value": "...|null",
      "metadata": {{}}
    }}
  ],
  "relationships": [
    {{
      "source_name": "...",
      "target_name": "...",
      "relation_type": "contribute_to",
      "confidence": 0.0
    }}
  ]
}}
"""

DESCRIPTION_CANONICALIZE_PROMPT = """You are a knowledge-graph canonicalization expert.

Below is a JSON array of node descriptions extracted from an operational incident report.
Each description may contain task-instance-specific tokens that make it non-reusable across incidents.

Rewrite EACH description to be task-agnostic while preserving its semantic meaning:

Remove or replace with a generic placeholder:
- Specific URLs, log links, HTML tags
- Numeric counts, percentages, and statistics that are specific to one run
  (e.g. "1179 -> 9 candidates, 99% cut" → "99% filtered")
- Dataset names, file paths, output filenames, LFNs, GUIDs — replace the ENTIRE
  string (including all scope:name.component.component... structure) with a
  single placeholder such as <dataset_name>; do NOT preserve the internal
  dot-separated or colon-separated structure as individual sub-placeholders
  (e.g. "mc23:mc23.604924.Py8.deriv.DAOD.e8599_r16083" → "<dataset_name>")
- Task IDs, job IDs, PanDA IDs, pilot IDs
- Hostnames, IP addresses, site names
- Timestamps, user names

Keep:
- The error type, failure mode, or check name
- The overall semantic meaning and relevant system/component names
- Structural descriptions (e.g. "all candidates filtered at memory check")

If a description is already generic (no instance-specific data), return it unchanged.

Input: {descriptions_json}

Return a JSON array of the rewritten descriptions in the SAME ORDER as the input.
Return ONLY the JSON array — no explanation, no markdown code fences.
"""

EXPLORER_TOOL_SELECTION_PROMPT = """You are a diagnostic data-collection agent for a PanDA computing task.

A knowledge reviewer has found the following issues with the extracted knowledge graph:

REVIEWER ISSUES:
{review_issues}

TASK CONTEXT (key fields only):
{task_summary}

AVAILABLE TOOLS:
{tools_description}

Your job is to select the minimal set of tools whose results are most likely to resolve the reviewer's issues.

Selection rules:
- Only select a tool if at least one reviewer issue directly implies that the tool's data is missing.
- Do NOT select a tool speculatively — if the issues do not suggest its data is needed, omit it.
- A tool that requires a field (e.g. retryID, errorDialog) that is absent or empty in the task context MUST NOT be selected.
- Prefer fewer tools. One or two targeted calls is better than calling everything.
- If no tool would help, output an empty array.

Output a JSON array. Each element has exactly three keys:
  "tool"   — the tool name exactly as listed in AVAILABLE TOOLS
  "args"   — a JSON object matching the tool's parameters schema
  "reason" — one sentence explaining which reviewer issue this call addresses

Output ONLY the JSON array — no markdown, no explanation outside the JSON.

Example (do not copy literally — populate args from the actual task context above):
[
  {{
    "tool": "fetch_error_dialog_logs",
    "args": {{"task_id": 12345, "error_dialog": "<a href=\\"http://...\\">log</a>"}},
    "reason": "Reviewer noted Symptom nodes are too vague; log content will provide specific error codes."
  }}
]
"""

KNOWLEDGE_REVIEW_PROMPT = """You are an incident knowledge graph gap analyzer for PanDA computing tasks.

Your job is to identify information that is MISSING from the graph — information that would be
needed to fully understand the incident.

STRICT GROUNDING RULE: Only flag a gap if it is directly implied by either:
  (a) the graph itself — e.g. a SYMPTOM node exists but no CAUSE explains it
  (b) the task context — numeric fields (nJobsFailed > 0), status values, or errorDialog text
      that imply certain data exists but is absent from the graph
Do NOT speculate about information not suggested by the graph or task context.

Evaluate for:
- STRUCTURAL GAPS: missing nodes that the graph implies should exist
    * SYMPTOM with no CAUSE or explanatory TASK_FEATURE node
    * CAUSE with no SYMPTOM (orphaned cause)
    * Completed/finished task (status=finished/done) with no RESOLUTION node
  NOTE: contribute_to edges between Task_Feature and Cause nodes are wired
  automatically by the system after review — do NOT flag their absence as a gap.
  If the relevant Task_Feature nodes already exist in the graph, the structural
  requirement is met regardless of whether explicit edges are present.
  Use SEMANTIC understanding when checking Task_Feature coverage: e.g. a node
  with attribute=coreCount and value=1 represents a single-core queue constraint
  and is equivalent to a node named "queueType=single_core".
- SPECIFICITY GAPS: node names or descriptions too vague to be useful
    * e.g. a Symptom named "error" or "failure" with no error code or further detail
- CONTEXTUAL GAPS: task context that implies data exists but is not in the graph
    * retryID present but no cause node referencing the parent task
    * errorDialog text implies a specific root cause but no CAUSE node captures it

Gap-driven rejection — reject if any of the following is true:
- A STRUCTURAL GAP was found (e.g. SYMPTOM with no explanatory node at all, or no CAUSE node).
- A SPECIFICITY GAP was found that makes a node meaningless (e.g. Symptom named "error"
  with no further detail).
- A CONTEXTUAL GAP was found that represents a significant missing dimension of the incident
  (e.g. errorDialog implies a clear root cause but the graph has no CAUSE node for it).

Approve if: minimum requirement is met AND no significant gaps were found.
A missing RESOLUTION alone is acceptable — the task may be unresolved.
Minor gaps (small details that do not meaningfully impair incident understanding) may be
noted in issues but should not cause rejection — set approved=true.

When approved, set issues to an empty list.

SOURCE EXCERPTS (if present): use as hints only to identify specific extractable details
missing from the graph. Their absence does not affect the verdict.

EXTRACTED GRAPH:
{graph_summary}

TASK CONTEXT:
{task_summary}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

Use the domain documentation to:
- Understand what a given task status (e.g. exhausted, broken, aborted) means and what
  conditions trigger it.
- Check whether the graph reflects the documented causes for that status.
- Flag as a gap if the graph is missing nodes that the documentation says should be present
  for this status.  IMPORTANT: domain docs may list multiple possible dimensions (CPU,
  memory, disk …) for a status.  When the errorDialog or Symptom nodes identify a SPECIFIC
  dimension (e.g. "CPU time limit exceeded"), restrict gap-flagging to ONLY that dimension.
  Do NOT flag gaps for other dimensions the docs mention — those are alternative causes that
  did NOT trigger this particular incident.

SOURCE EXCERPTS (optional):
{sources_summary}

AVAILABLE DATA SOURCES (tool catalogue the explorer can invoke to fill gaps):
{available_tools}

When a gap in "issues" could be resolved by one of the above tools, append a note such as
"→ resolvable with <tool_name>" to that issue string.
Do NOT invent tool names — only reference tools listed above.
If no tools are listed, assess gaps exactly as before.

FAILURE DIMENSION: Identify the resource dimension(s) that caused this failure
by reading and understanding the errorDialog and Symptom text.

Allowed dimensions and their typical signals:
  cpu       — CPU time, core count, cpu efficiency, cpu consumption, over_cpu_consumption,
               payload running on wrong queue type
  memory    — memory limit, RAM, RSS, OOM, out-of-memory
  walltime  — walltime, wall time, timeout, job duration (jobDuration), elapsed time
  disk      — disk space, I/O, storage, output size
  site      — specific site name, brokerage failure, no candidate site
  job_size  — events per job, files per job, input/output file count or size

Rules:
  • Semantic understanding of the errorDialog is allowed — you do not need an exact
    keyword match.  For example, if the errorDialog formula contains "jobDuration",
    recognise that as a walltime parameter.
  • Only include a dimension if it is directly described or named in the errorDialog
    or Symptom text.  Do NOT add dimensions based on general domain knowledge about
    how resources interact (e.g. do not add "memory" for a CPU incident just because
    memory can affect CPU efficiency).
  • If you cannot identify any dimension from the text, return empty list.

PROCEDURE CONSISTENCY CHECK:
For each Procedure node in the graph, verify that its strategy_type is consistent
with the linked Cause node.  Flag as a specificity gap if:
- The strategy_type is too vague to be actionable (e.g. "investigate jobs" with no
  further detail about what to look for or which job status/type).
- The strategy_type contradicts the cause (e.g. cause is "cpu consumption exceeded
  job limit" but procedure investigates memory metrics).
Examples of consistent pairs:
- Cause "cpu consumption exceeded job limit" → strategy "investigate finished normal
  jobs with high cpuConsumptionTime and wallTime"
- Cause "build step failure" → strategy "investigate failed build jobs for
  transformation errors"
- Cause "input data missing from parent task" → strategy "check parent task status
  and output dataset availability"
PROCEDURE CONSISTENCY CHECK (for Procedure nodes that are present):
Verify that each Procedure node's strategy_type is consistent with its linked Cause node.
Flag as a specificity gap if the strategy_type is too vague or contradicts the cause.

Respond with a JSON object only — no markdown, no explanation outside the JSON:
{{
  "approved": true | false,
  "confidence": <float 0.0-1.0>,
  "issues": ["<concise gap description>"],
  "feedback": "<actionable extractor instruction addressing the gaps, or empty string if approved>",
  "failure_dimension": ["cpu" | "memory" | "walltime" | "disk" | "site" | "job_size"]
}}
"""

EXPLORATION_GAP_ANALYSIS_PROMPT = """You are a diagnostic gap analyst for a PanDA computing task review.

A knowledge reviewer identified these issues with an extracted knowledge graph:

REVIEWER ISSUES:
{review_issues}

TASK CONTEXT:
{task_summary}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

Use the domain documentation to understand what the task status and error conditions mean,
and to refine the gap descriptions with domain-specific terminology and expected data.

AVAILABLE TOOLS (read-only catalogue — do not plan tool calls yet):
{tools_description}

Your task: for each reviewer issue, produce a precise description of the SPECIFIC
INFORMATION that is missing from the graph and why it matters for understanding the
incident.

Rules:
- Ground every gap in a concrete field in the task context or a structural implication
  of the reviewer issue.
- Use domain documentation to make gap descriptions more specific (e.g. if docs say
  exhausted is caused by CPU efficiency or memory leaks, say which metric is missing).
- Do NOT mention tool names — that is the next step.
- If a reviewer issue is already fully addressed by the graph (false alarm), omit it.
- If multiple reviewer issues describe the same missing information, merge them into one.
- Set "resolvable" to false if no tool in the catalogue could plausibly fill the gap.

Output a JSON array only — no markdown, no explanation outside the JSON:
[
  {{
    "gap": "<concise description of the specific missing information>",
    "impact": "<one sentence: why this gap impairs incident understanding>",
    "resolvable": true
  }}
]
"""

EXPLORATION_PLAN_PROMPT = """You are a diagnostic data-collection planner for a PanDA computing task.

You have identified these specific information gaps that need to be filled:

GAPS TO RESOLVE:
{gaps}

TASK CONTEXT (key fields only):
{task_summary}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

AVAILABLE TOOLS:
{tools_description}

Your job: design an ordered execution plan of tool-call groups (steps).

Step design rules:
- Tools within one step run CONCURRENTLY — only group tools that are fully INDEPENDENT
  of each other (neither's usefulness depends on the other's result).
- If tool B would be most useful after tool A's results are available in the next
  extraction pass, put B in a LATER step.
- Each step must have a single "reason" string: what it fetches and which gap it closes.
- Only select a tool if at least one gap directly implies that its data is needed.
- A tool that requires a field that is absent or empty in the task context MUST NOT be
  selected.
- If no tool would help any gap, output an empty array.

Output a JSON array of steps only — no markdown, no explanation outside the JSON:
[
  {{
    "reason": "<why this step is needed and which gap it addresses>",
    "tool_calls": [
      {{
        "tool": "<tool name exactly as listed in AVAILABLE TOOLS>",
        "args": {{ }}
      }}
    ]
  }}
]
"""

PROCEDURE_EXECUTION_PROMPT = """You are a diagnostic data-collection agent for a PanDA computing task.

You have been given an investigation procedure to execute.  Your only job is to
select the tool(s) whose output directly serves the procedure.

PROCEDURE:
{gaps}

TASK CONTEXT (key fields only):
{task_summary}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

AVAILABLE TOOLS:
{tools_description}

Selection rules:
- Read each procedure instruction and find the tool whose description matches what
  the instruction asks to collect.  Match on keywords: "scout jobs" → scout job tool,
  "cpuConsumptionTime / wallTime / jobDuration" → tool that returns those fields,
  "error logs" → log-fetching tool, etc.
- Always prefer the most specific match.  A tool that explicitly mentions the same
  metric or job type as the procedure is the right choice.
- Do NOT add steps for data not mentioned in the procedure.
- If genuinely no tool matches, output an empty array.

Output a JSON array of steps only — no markdown, no explanation outside the JSON:
[
  {{
    "reason": "<quote the procedure instruction this step executes>",
    "tool_calls": [
      {{
        "tool": "<tool name exactly as listed in AVAILABLE TOOLS>",
        "args": {{ }}
      }}
    ]
  }}
]
"""

JOB_DIAG_NORMALIZE_PROMPT = """You are an expert in distributed computing and data management systems.

A PanDA job produced the following error diagnostic message:

  {diag_text}

Your task is to rewrite this message as a SHORT, REUSABLE description of the
TYPE of problem, removing all job-specific tokens so that semantically
identical errors from different jobs produce the same output.

Remove:
- File names, dataset names, dataset scopes, LFNs, GUIDs
- Replica URLs, storage endpoints, RSE names
- Job IDs, task IDs, PanDA IDs, pilot IDs, process IDs
- Timestamps, hostnames, IP addresses, port numbers
- User names, working-group scopes
- Any numeric identifier that is unique per job or per run

Keep:
- The error category or error type (e.g. "missing replica", "quota exceeded",
  "stage-in failure", "payload exit code non-zero")
- The system or component involved (e.g. "Rucio", "DDM", "pilot", "Athena")
- The structural cause if evident (e.g. "too many files", "checksum mismatch",
  "permission denied")

Return ONLY the normalised string — no JSON, no explanation, no punctuation
beyond what is part of the message itself.  Maximum 120 characters.
"""

PROCEDURE_DESC_MERGE_PROMPT = """Merge the two descriptions of the same investigation procedure into the best single description. Keep all distinct information from both; prefer the more specific and complete phrasing. Return ONLY the merged description string, no explanation.

Procedure name: {name}

Description A:
{desc_a}

Description B:
{desc_b}

Merged description:"""

DOC_SEARCH_KEYWORDS_PROMPT = """You are helping to query PanDA system documentation.

Given the following error information from a PanDA task, extract 2-5 short,
focused search terms that would best retrieve the relevant PanDA documentation
sections. Return ONLY a JSON array of strings, no explanation.

Rules:
- Only use terms that appear in the provided text — do not invent terms.
- Prefer PanDA system identifiers exactly as written — ALL_CAPS constants
  and snake_case reason tokens are the most precise search terms.
- Each term must be at least 4 characters.
- Omit generic single words (job, task, log, error, failed, timeout) unless
  combined with a more specific qualifier.

Error dialog:
{error_dialog}

Email / operator notes (may be empty):
{email_text}

JSON array:"""

PRESCRIPTION_CLASSIFY_PROMPT = """You are a PanDA task failure analyst.

TASK FAILURE ANALYSIS:
Root cause: {root_cause}
Resolution from knowledge graph: {resolution}
Investigation findings: {investigation}

Classify the resolution into one of these action types:
- resubmit: The user should resubmit the task with different CLI options or parameters
- contact_admin: The user should contact a site or system administrator
- fix_code: The user should fix their analysis script or code
- wait: The user should wait for an external condition to be resolved
- other: Any other action type

Also specify whether you need CLI documentation (only for "resubmit" type), and which
submission command applies ("prun" or "pathena") based on the components below.

Components identified: {components}

Respond as JSON only:
{{
  "action_type": "resubmit|contact_admin|fix_code|wait|other",
  "needs_cli_docs": true|false,
  "component": "prun|pathena|null"
}}"""

PRESCRIPTION_COMPOSE_PROMPT = """You are a PanDA task failure analyst providing actionable remediation steps.

TASK SUMMARY:
{task_summary}

ROOT CAUSE: {root_cause}

RESOLUTION FROM KNOWLEDGE GRAPH:
{resolution}

INVESTIGATION FINDINGS (job-level metrics from this task):
{investigation}

ACTION TYPE: {action_type}

{cli_docs_section}

Based on the above, provide concrete, actionable steps for the operator.

For "resubmit": Give 2-3 specific CLI option changes with exact flag names, recommended
values, and one-sentence justification referencing measured metrics where available.
For "contact_admin": Specify who to contact and what to report.
For "fix_code": Describe what to look for and change in the analysis code.
For "wait": Describe what to wait for and how to check if it is resolved.
For "other": Provide clear free-text action steps.

Respond as JSON only:
{{
  "action_type": "{action_type}",
  "hints": ["concrete step 1", "concrete step 2"],
  "command_template": "--flag=value --flag2=value2",
  "notes": "any caveats or missing information"
}}

Note: "command_template" is only required for action_type "resubmit"; omit for others."""

PRESCRIPTION_EMAIL_PROMPT = """You are a technical communication expert writing to a physicist about their failed PanDA task.

Task ID: {task_id}
Task description: {task_description}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

ROOT CAUSE ANALYSIS:
{analysis}

PRESCRIPTION (recommended action):
{prescription}

Write a clear, professional email that:
1. Briefly acknowledges the issue
2. Explains the root cause in accessible language
3. Provides the specific action steps from the prescription above
4. For "resubmit" prescriptions: include the suggested command options concretely
5. For "contact_admin" prescriptions: include the contact details
6. References specific measured values from the investigation where available
7. Offers further help if needed
8. Is professional but friendly

Generate the email content:"""

