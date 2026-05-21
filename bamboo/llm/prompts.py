"""LLM prompt templates for all pipeline stages.

All prompts are plain strings with ``{placeholder}`` format fields.
Call ``.format(...)`` to produce the final prompt before passing to the LLM.

Prompt constants
----------------
``EXTRACTION_PROMPT``
    General-purpose knowledge-graph extraction from raw text.
    Used by ``LLMExtractionStrategy``.

``EMAIL_EXTRACTION_SYSTEM`` + ``EMAIL_EXTRACTION_USER``
    Extracts ``Cause``, ``Resolution``, ``Task_Context``, and ``Procedure`` nodes
    from an email thread.  Used by :class:`~bamboo.agents.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`.

``LOG_EXTRACTION_SYSTEM`` + ``LOG_EXTRACTION_USER``
    Extracts ``Symptom``, ``Component``, and ``Task_Context`` nodes from raw
    log text.  Cause and Resolution are intentionally excluded — logs record
    what happened, not why or how to fix it; those come from the email thread
    and graph reasoning.  The prompt de-duplicates repeated occurrences of the
    same error, captures the components named in stack traces or log prefixes,
    and stores the surrounding prose as Task_Context for vector search.

``BROKERAGE_LOG_EXTRACTION_SYSTEM`` + ``BROKERAGE_LOG_EXTRACTION_USER``
    Extracts a ``Symptom`` (placement outcome: ``BrokerageNoCandidates`` /
    ``BrokerageCandidateFound``), ``Task_Feature`` nodes for the structured
    task constraints visible in the log (memory requirement, IO intensity,
    data locality, etc.), and ``Task_Context`` nodes for the dominant filter
    stages.  Reuses existing node types; no brokerage-specific types needed.

``CAUSE_RESOLUTION_CANONICALIZE_SYSTEM`` + ``CAUSE_RESOLUTION_CANONICALIZE_USER``
    Normalises a raw Cause/Resolution name into a stable canonical phrase,
    optionally matching against existing names from the vector DB.

``TASK_ERROR_CATEGORY_LABEL_SYSTEM`` + ``TASK_ERROR_CATEGORY_LABEL_USER``
    Converts a raw error message into a short CamelCase error-category label.
    Used by :class:`~bamboo.agents.extractors.panda_knowledge_extractor.ErrorCategoryStore`.

``SUMMARIZATION_SYSTEM`` + ``SUMMARIZATION_USER``
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

``EXPLORER_TOOL_SELECTION_SYSTEM`` + ``EXPLORER_TOOL_SELECTION_USER``
    Given a reviewer's issue list and available MCP tools, selects which
    tools to call to fill data gaps before a re-extraction attempt.
    Used by :class:`~bamboo.agents.context_enricher.ContextEnricher`.

``KNOWLEDGE_REVIEW_SYSTEM`` + ``KNOWLEDGE_REVIEW_USER``
    Reviews an extracted knowledge graph for completeness, accuracy, and
    consistency against the original sources.  Returns a structured JSON
    verdict with issues and corrective feedback.
    Used by :class:`~bamboo.agents.knowledge_reviewer.KnowledgeReviewer`.

``DOC_SEARCH_KEYWORDS_SYSTEM`` + ``DOC_SEARCH_KEYWORDS_USER``
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


SUMMARIZATION_SYSTEM = """You are a technical documentation expert. Create a concise summary of the following knowledge graph and email thread.

STRICT RULE: Only use information explicitly present in either the Knowledge Graph or the Email Thread provided in the user message.
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
"""

SUMMARIZATION_USER = """Knowledge Graph:
{graph_data}

Email Thread (if available):
{email_text}

TERMINOLOGY REFERENCE (use only to interpret technical terms already present in the graph or email — do NOT introduce new causes, resolutions, procedures, or commands from this section):
{doc_hints}

Summary:
"""

CAUSE_IDENTIFICATION_PROMPT = """You are a root cause analysis expert. Analyze the task information and database results to identify the most likely root cause.

Task Information:
{task_info}
Note: Use external_info to identify the failure's bottleneck stages or error patterns,
then cite only the task_info fields that explain WHY those specific bottlenecks occurred.
Do not cite task_info fields that have no connection to the observed failure patterns.
(Concept-level matching is fine: e.g. a "memory check" bottleneck in external_info
justifies citing a memory/RAM field from task_info even if the names differ.)

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

If graph_results and vector_results are both empty or weakly matched and external_info
contains investigation data, treat that investigation data as the primary evidence source.
In that case, be explicit in "reasoning" that this is a first-principles analysis with no
confirmed historical precedent, and label the resolution as a hypothesis rather than a
confirmed procedure.

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

EMAIL_EXTRACTION_SYSTEM = """You are a root-cause analysis expert reading an operational incident email thread.

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
- Procedures are NOT emitted as nodes in the ``nodes`` array. Instead,
  list each atomic investigation action separately in the top-level
  ``atomic_actions`` field (see output schema below). Procedure nodes
  and their relationships are constructed in code from this list — the
  list-shaped output ensures that distinct actions stay distinct.

Relationships to emit (between extracted nodes only):
- Cause -[solved_by]-> Resolution
- Task_Context -[contribute_to]-> Cause  (only when the context directly supports a cause)
(Note: ``investigated_by`` and ``depends_on`` edges for Procedures are
constructed in code from ``atomic_actions`` and ``action_dependencies``
— do NOT emit them here.)

Canonicalization:
- If the same cause or resolution is mentioned multiple times, emit it only once.
- Keep names concise and general (no incident-specific IDs, paths, or dataset names).

Domain knowledge is provided in the user message under "Domain knowledge" — use it for
context and filling in MISSING details only; do NOT replace or override terminology that
is already present in the source text.

Output ONLY a valid JSON object — no explanation, no markdown fences:
{
  "nodes": [
    {
      "node_type": "Cause|Resolution|Task_Context",
      "name": "...",
      "description": "...",
      "metadata": {},
      "steps": []
    }
  ],
  "relationships": [
    {
      "source_name": "...",
      "target_name": "...",
      "relation_type": "solved_by|contribute_to",
      "confidence": 0.0
    }
  ],
  "atomic_actions": [
    {
      "id": "a1",
      "summary": "short snake_case label uniquely identifying this single action",
      "description": "prose from the email describing this action",
      "parameters": {},
      "cause_name": "<canonical cause name this action investigates>"
    }
  ],
  "action_dependencies": [
    {
      "from": "a1",
      "to": "a2",
      "type": "explicit | implicit",
      "reason": "short justification — quote or paraphrase the email text that establishes the dependency"
    }
  ]
}

Rules for ``atomic_actions``:
- Each action in the email's procedure section becomes ONE entry. Do
  NOT merge multiple actions into a single entry.
- ``summary`` MUST differ between entries — it is used to make each
  Procedure node uniquely identifiable.
- ``cause_name`` MUST be the ``name`` of one of the Cause nodes you
  emitted in ``nodes``.
- ``action_dependencies`` lists pairs where the action with id ``to``
  needs the output of the action with id ``from`` (e.g. "find X, then
  look up Y in X"). Only emit when the email explicitly chains the
  actions; leave empty if all actions can run independently.
- Each dependency MUST include a ``reason`` field articulating the
  rationale (e.g. an email chaining word, a data hand-off the second
  action requires from the first). The reason makes the dependency
  reviewable.
- Each dependency MUST also include a ``type`` field. Mark it
  ``explicit`` ONLY when (a) the email uses an explicit chaining word
  between the two actions ("then", "after", "next", "using the result
  of") OR (b) the second action literally references an identifier
  produced by the first (a task ID, PandaID, file path). Mark
  everything else ``implicit`` — including order-of-listing
  inferences, "first investigate then verify"-style procedural common
  sense, and topic-relatedness reasoning. When in doubt, prefer
  ``implicit``. Code consumes only ``explicit`` dependencies; implicit
  ones are logged but not materialised as edges.
"""

EMAIL_EXTRACTION_USER = """Domain knowledge (PanDA system — for context and filling in MISSING details only;
do NOT replace or override terminology that is already present in the source text):
{doc_hints}

Email thread:
{email_text}
"""

EMAIL_INVESTIGATION_SUMMARY_SYSTEM = """You are reading an incident email thread.
Identify any investigation steps that were explicitly described — what was examined,
checked, or queried, and what was found or concluded.

List each step briefly (one line each).  Only report steps that are explicitly stated
in the email — do NOT infer or speculate.  If no investigation steps are described,
respond with exactly: none
"""

EMAIL_INVESTIGATION_SUMMARY_USER = """Email thread:
{email_text}

Investigation steps:"""

CAUSE_RESOLUTION_CANONICALIZE_SYSTEM = """You are a knowledge-graph canonicalization expert.

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
"""

CAUSE_RESOLUTION_CANONICALIZE_USER = """Node type: {node_type}

Existing names in the graph ({node_type}):
{existing_names}

Raw name: {raw_name}

Canonical name:"""

TASK_ERROR_CATEGORY_LABEL_SYSTEM = """You are an error classification expert.

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
"""

TASK_ERROR_CATEGORY_LABEL_USER = """Raw error message:
{error_message}

Category label:"""

ERROR_CATEGORY_MATCH_SYSTEM = """You are an error classification expert.

You will be given an error category and an error message. Answer with exactly "yes" or "no" — does the error message fit the category? No explanation.
"""

ERROR_CATEGORY_MATCH_USER = """Error category: "{category}"

Error message:
{error_message}

Answer (yes/no):"""


LOG_EXTRACTION_SYSTEM = """You are a log analysis expert reading raw operational log output from a distributed computing job.

Your goal is to extract reusable, incident-agnostic knowledge from the log — the kind that helps identify recurring failure patterns across many jobs.

Extract ONLY the following node types — do NOT emit Cause or Resolution nodes (those are determined from email threads and graph reasoning, not from logs alone):

Node types to extract:

- Symptom: A distinct class of error or failure signal observed in the log.
  name = a short CamelCase label describing the error pattern, stripped of all
         incident-specific tokens (dataset names, file paths, job IDs, timestamps,
         hostnames, numeric IDs).  Use the same canonicalisation rules as
         TASK_ERROR_CATEGORY_LABEL — the same structural error in two different
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

Output ONLY a valid JSON object — no explanation, no markdown fences:
{
  "nodes": [
    {
      "node_type": "Symptom|Component|Task_Context",
      "name": "...",
      "description": "...",
      "severity": "critical|error|warning|info|null",
      "system": "...|null",
      "metadata": {
        "occurrence_count": 1
      }
    }
  ],
  "relationships": [
    {
      "source_name": "...",
      "target_name": "...",
      "relation_type": "originated_from|contribute_to",
      "confidence": 0.0
    }
  ]
}
"""

LOG_EXTRACTION_USER = """Log text:
{log_text}
"""

BROKERAGE_LOG_EXTRACTION_SYSTEM = """You are an expert in distributed computing and grid job scheduling reading a pre-filtered PanDA job-brokerage log.

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

Output ONLY a valid JSON object — no explanation, no markdown fences:
{
  "nodes": [
    {
      "node_type": "Symptom|Task_Feature|Task_Context",
      "name": "...",
      "description": "...",
      "severity": "critical|warning|info|null",
      "attribute": "...|null",
      "value": "...|null",
      "metadata": {}
    }
  ],
  "relationships": [
    {
      "source_name": "...",
      "target_name": "...",
      "relation_type": "contribute_to",
      "confidence": 0.0
    }
  ]
}
"""

BROKERAGE_LOG_EXTRACTION_USER = """Brokerage log:
{log_text}
"""

DESCRIPTION_CANONICALIZE_SYSTEM = """You are a knowledge-graph canonicalization expert.

You will be given a JSON array of node descriptions extracted from an operational incident report.
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

Return the rewritten descriptions as plain text, one description per line,
in the SAME ORDER as the input.
Rules for the output:
- No JSON, no numbering, no bullet points, no markdown.
- Each description must be plain readable text — no quotation marks, backslashes,
  or other special characters.
- For complex command-line strings or structured data, write a brief English phrase
  (e.g. "panda job submission command with source archive and output files")
  rather than a templated version of the command.
- The number of output lines MUST equal the number of input descriptions.
"""

DESCRIPTION_CANONICALIZE_USER = """Input: {descriptions_json}

Output exactly {n} lines — one per input description.
"""

EXPLORER_TOOL_SELECTION_SYSTEM = """You are a diagnostic data-collection agent for a PanDA computing task.

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

Example (do not copy literally — populate args from the actual task context):
[
  {
    "tool": "fetch_linked_log_files",
    "args": {"task_id": 12345, "error_dialog": "<a href=\\"http://...\\">log</a>"},
    "reason": "Reviewer noted Symptom nodes are too vague; log content will provide specific error codes."
  }
]
"""

EXPLORER_TOOL_SELECTION_USER = """A knowledge reviewer has found the following issues with the extracted knowledge graph:

REVIEWER ISSUES:
{review_issues}

TASK CONTEXT (key fields only):
{task_summary}

AVAILABLE TOOLS:
{tools_description}
"""

KNOWLEDGE_REVIEW_SYSTEM = """You are an incident knowledge graph gap analyzer for PanDA computing tasks.

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

SOURCE EXCERPTS (if present in the user message): use as hints only to identify specific
extractable details missing from the graph. Their absence does not affect the verdict.

Use the domain documentation (provided in the user message) to:
- Understand what a given task status (e.g. exhausted, broken, aborted) means and what
  conditions trigger it.
- Check whether the graph reflects the documented causes for that status.
- Flag as a gap if the graph is missing nodes that the documentation says should be present
  for this status.  IMPORTANT: domain docs may list multiple possible dimensions (CPU,
  memory, disk …) for a status.  When the errorDialog or Symptom nodes identify a SPECIFIC
  dimension (e.g. "CPU time limit exceeded"), restrict gap-flagging to ONLY that dimension.
  Do NOT flag gaps for other dimensions the docs mention — those are alternative causes that
  did NOT trigger this particular incident.

When a gap in "issues" could be resolved by one of the available tools (provided in the user message),
append a note such as "→ resolvable with <tool_name>" to that issue string.
Do NOT invent tool names — only reference tools listed in the user message.
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
{
  "approved": true | false,
  "confidence": <float 0.0-1.0>,
  "issues": ["<concise gap description>"],
  "feedback": "<actionable extractor instruction addressing the gaps, or empty string if approved>",
  "failure_dimension": ["cpu" | "memory" | "walltime" | "disk" | "site" | "job_size"]
}
"""

KNOWLEDGE_REVIEW_USER = """EXTRACTED GRAPH:
{graph_summary}

TASK CONTEXT:
{task_summary}

DOMAIN DOCUMENTATION (authoritative PanDA system knowledge — treat as ground truth):
{domain_hints}

SOURCE EXCERPTS (optional):
{sources_summary}

AVAILABLE DATA SOURCES (tool catalogue the explorer can invoke to fill gaps):
{available_tools}
"""

EXPLORATION_GAP_ANALYSIS_SYSTEM = """You are a diagnostic gap analyst for a PanDA computing task review.

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
  {
    "gap": "<concise description of the specific missing information>",
    "impact": "<one sentence: why this gap impairs incident understanding>",
    "resolvable": true
  }
]
"""

EXPLORATION_GAP_ANALYSIS_USER = """A knowledge reviewer identified these issues with an extracted knowledge graph.

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
"""

PROCEDURE_ORCHESTRATION_CODE_SYSTEM = """You are writing orchestration logic for executing historical PanDA investigation procedures.

Each gap below is a HISTORICAL PROCEDURE instruction from a prior incident
(e.g. "Procedure 'X in log analysis' (strategy: log_analysis, for cause: 'X'):
check failed job log and compare with a similar successful task.
Historical parameters: [...]").

Produce TWO outputs:

1) The body of this async Python function that executes the procedures verbatim:
     async def orchestrate(tools, asyncio):

   Rules for the function body:
   - The description text after the colon IS the procedure — pick the tool
     whose capability matches what the description asks for; do not substitute
     a generic alternative just because it is available.
   - The procedure description may contain MULTIPLE actions (e.g.
     "check X and compare with Y"). You MUST address every action; do not
     stop after the first. If you cannot map one action to a tool, list it
     in `capability_gaps` rather than silently omitting it.
   - For each procedure, choose the tool whose description matches what the
     procedure asks to collect, and embed the historical parameter values as
     literal Python values in the call args.
   - DO NOT invent new tools, new parameters, or speculative reasoning steps.
   - DO NOT add steps for data the procedure does not mention.
   - INDEPENDENT calls MUST run concurrently inside a single `asyncio.gather`:
       a, b, c = await asyncio.gather(
           tools.x(),
           tools.y(),
           tools.z(),
       )
     Do NOT issue them as separate sequential `await`s — that wastes
     wall-clock time and can exceed the orchestration timeout.
   - Sequential `await` is ONLY for when the procedure text explicitly
     requires it (e.g. "find Y first, then look up its log").
   - If a procedure mixes independent and dependent calls, group the
     independent ones in one `asyncio.gather`, then sequentially `await`
     the dependent ones.
   - Tools that accept task_data receive it automatically — do not pass task_data.
   - Check for empty / error results before passing them to downstream calls.
   - Return a dict mapping descriptive labels to fetched values.
   - Only call tools in AVAILABLE TOOLS. No imports, no open(), no exec().

2) A list of capability gaps — procedures that no available tool can satisfy.
   Each entry:
     {"investigation": "<the procedure that has no matching tool>",
       "suggested_tool_capability": "<the capability a future tool would need>"}
   If every procedure has a matching tool, leave the list empty.

3) An ``explanation`` string narrating how the code was generated.
   For each tool call in your ``orchestration_code``, briefly state:
     (a) why this tool was picked for the procedure
     (b) for each keyword argument, the SOURCE of its value, marked
         exactly as one of:
           - ``procedure parameters`` (value taken from the procedure's
             parameters dict)
           - ``procedure description`` (value mentioned literally in the
             procedure description)
           - ``inferred default`` (you chose the value yourself
             because neither the procedure parameters nor the
             description specified it)
     (c) if any argument is marked ``inferred default``, state plainly
         why you didn't simply omit the argument and let the tool's
         own default apply.
   Keep the explanation short — one or two sentences per tool call.

Return ONLY a single JSON object with these three keys (no markdown
fences, no commentary). The JSON key MUST be the literal string
`orchestration_code` — do not abbreviate it to `orchest_code`, `code`,
or any other variant; doing so will cause your output to be discarded.
Use the JSON-escaped string form for `orchestration_code` (literal `\\n`
for newlines):

{
  "orchestration_code": "    similar = await tools.find_similar_successful_tasks()\\n    if not similar:\\n        return {\\"similar\\": []}\\n    comparison = await tools.compare_failed_vs_successful_job_logs(reference_task_id=similar[0][\\"jediTaskID\\"])\\n    return {\\"similar\\": similar, \\"comparison\\": comparison}",
  "capability_gaps": [
    {"investigation": "...", "suggested_tool_capability": "..."}
  ],
  "explanation": "per-tool-call rationale — see rule (3) above"
}
"""

PROCEDURE_ORCHESTRATION_CODE_USER = """PROCEDURES TO EXECUTE:
{gaps}

TASK CONTEXT:
{task_summary}

AVAILABLE TOOLS (callable as `await tools.<name>(keyword=value)` — keyword args only):
{tools_description}
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

PROCEDURE_DESC_MERGE_SYSTEM = """Merge the two descriptions of the same investigation procedure into the best single description. Keep all distinct information from both; prefer the more specific and complete phrasing. Return ONLY the merged description string, no explanation.
"""

PROCEDURE_DESC_MERGE_USER = """Procedure name: {name}

Description A:
{desc_a}

Description B:
{desc_b}

Merged description:"""

DOC_SEARCH_KEYWORDS_SYSTEM = """You are helping to query system documentation.

Given the following error information from a task, return a JSON object with:
- "nl_query": a concise natural-language description of the problem (1 sentence, no identifiers)
- "keywords": 2-5 search terms optimised for PanDA documentation (ReadTheDocs)

Rules for keywords:
- Extract PanDA concept names, status names, and parameter names as they appear
  in documentation prose — NOT as raw Python identifiers.
- Decompose snake_case and camelCase identifiers into individual tokens likely
  to appear in documentation:
    action=set_exhausted  →  "exhausted"     (drop the verb prefix "set_")
    reason=scout_ramCount →  "scout", "ramCount"  (split at underscore; keep
                              camelCase parameter names intact — they appear
                              in configuration tables)
    low_cpuEfficiency     →  "cpuEfficiency" (drop the adjective prefix)
- Ignore log-marker tokens (#KV, #ATM) and key= prefixes (action=, reason=).
- Output individual tokens, not multi-word phrases.
- Each term must be at least 4 characters.
- Omit generic words (job, task, log, error, failed, timeout).
- Only use terms derivable from the provided text — do not invent terms.
"""

DOC_SEARCH_KEYWORDS_USER = """Error dialog:
{error_dialog}

Email / operator notes (may be empty):
{email_text}

JSON object:"""

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

# ---------------------------------------------------------------------------
# PandaSourceNavigator prompts
# ---------------------------------------------------------------------------

PANDA_SOURCE_NAV_PROMPT = """\
You are navigating panda-server Python source code to answer this question:

  {question}

You have just read these source snippets:
{sources_read}

Decide whether more code is needed. Respond with ONLY a valid JSON object:

If the snippets above are sufficient to answer the question:
{{
  "action": "done",
  "relevant_qualnames": ["TaskUtilsModule.setScoutJobData_JEDI"],
  "reasoning": "one sentence"
}}

If you need to trace into symbols referenced in the snippets above:
{{
  "action": "follow_up",
  "follow_up_symbols": ["calcRamCount", "actualMemoryUsed"],
  "reasoning": "one sentence"
}}

Rules:
- Only request follow-up for symbol names that actually appear in the source snippets above.
- Do NOT request follow-up for symbols you imagine might exist — only ones visible in the code.
- "follow_up_symbols": exact identifiers as they appear in the source (case-sensitive).
- "relevant_qualnames": list only the qualnames that directly answer the question. Omit
  test methods, unrelated utilities, and anything that merely happened to be co-located.
  List all relevant ones; never return an empty list unless truly nothing applies.
"""

PANDA_SOURCE_SYNTHESIS_PROMPT = """\
You have read the following panda-server source code snippets to answer this question:

  {question}

SOURCE CODE:
{sources}

Write a concise technical answer (3–10 sentences) explaining exactly how the relevant \
code works with respect to the question. Reference specific variable names, function \
names, and logic from the source. Do not repeat the source verbatim — synthesise it.
"""

SOURCE_GREP_TERMS_PROMPT = """You are helping navigate PanDA Python source code.

Given this string (a log/error message or a code question), return a JSON array
of distilled search strings suitable for grepping PanDA Python source code.

For natural-language log messages:
  Strip all incident-specific values: numbers, percentages, memory sizes, task IDs,
  file paths, dataset names, and any token that varies between incidents.
  Each time you strip a value, split the result at that removal point — the text
  before the gap and the text after the gap become separate entries.
  Keep the original word order and spacing within each entry.
  Omit entries shorter than two words unless they are identifiers (see below).

For camelCase or snake_case identifiers present in the input:
  Copy them verbatim as separate entries.

Single-token entries are only valid if they are snake_case or camelCase identifiers
(contain an underscore or a lowercase-to-uppercase transition).
Do NOT emit as standalone entries: hash-prefixed markers (#KV, #ATM), short ALL-CAPS
tokens, or plain English nouns (jobs, tasks, files, memory, etc.).

Return ONLY a JSON array of strings. No explanation, no invented tokens.
Every string must be a verbatim substring of the input after stripping variables.

Question: {question}

JSON array:"""

# ---------------------------------------------------------------------------
# PandaDocNavigator prompts
# ---------------------------------------------------------------------------

PANDA_DOC_SUMMARIZE_SYSTEM = (
    "Summarize the PanDA WMS documentation section in the user message in 2-3 sentences.\n"
    "Preserve verbatim: all parameter names, option flags (e.g. --nFilesPerJob), "
    "error codes, class names, and any text that appeared in code formatting.\n"
    "Keep it concise and factual.\n\n"
    "Also classify the section as exactly one of:\n"
    '- "concept": ONLY for core PanDA vocabulary definitions — sections whose entire '
    "purpose is to define WHAT a fundamental PanDA entity IS. This means: Task, Job, "
    "Site, Scout job, Retry, and their status/state values. Nothing else qualifies. "
    "Ask: would this entry appear unchanged in a one-page PanDA glossary? If no, "
    'use "other".\n'
    '- "other": everything else — system component descriptions (Harvester, JEDI, '
    "plugins, brokers), architecture pages, internal details, how-to guides, FAQ entries, "
    "examples, CLI option tables, API references, parameter/configuration tables, "
    "troubleshooting procedures, and any section that describes HOW something "
    'works or is used. When in doubt, use "other".\n\n'
    'Return ONLY a JSON object: {"summary": "...", "doc_type": "concept|other"}'
)

PANDA_DOC_SUMMARIZE_USER = (
    "Page: {page_title}\n"
    "Section: {title}\n\n"
    "Content:\n{content}\n"
)

PANDA_DOC_TRAVERSAL_PAGE_SYSTEM = (
    "Review the PanDA WMS documentation page summaries in the user message. "
    "Return a JSON array of IDs for pages whose content is relevant to the search query — "
    "including pages that explain the concept, describe the status/error, "
    "list the parameters involved, or provide context needed to understand it. "
    "Exclude pages that are only about unrelated workflows or unrelated system components. "
    "Return an empty array [] if none apply.\n\n"
    "Return ONLY a JSON array of IDs, e.g.: [\"id1\", \"id2\"]"
)

PANDA_DOC_TRAVERSAL_PAGE_USER = (
    "Search query: {query}\n\n"
    "Pages:\n{pages_text}\n"
)

PANDA_DOC_TRAVERSAL_SECTION_SYSTEM = (
    "Review the section summaries in the user message. "
    "Return a JSON array of IDs for sections that answer, explain, or provide relevant "
    "context for the search query — including sections that describe the relevant status, "
    "error cause, or parameter. "
    "Exclude sections that are only about unrelated actions or unrelated system components. "
    "Return an empty array [] if none apply.\n\n"
    "Return ONLY a JSON array of IDs, e.g.: [\"id1\", \"id2\"]"
)

PANDA_DOC_TRAVERSAL_SECTION_USER = (
    "Search query: {query}\n\n"
    "You are exploring page: \"{page_title}\"\n"
    "Page context: {page_summary}\n\n"
    "Sections:\n{sections_text}\n"
)

TOOL_ORCHESTRATION_CODE_SYSTEM = """You are writing orchestration logic for PanDA task diagnosis.

Produce TWO outputs:

1) The body of this async Python function to fetch data needed to resolve the gaps:
     async def orchestrate(tools, asyncio):

   Rules for the function body:
   - Call tools with `await tools.<tool_name>(arg=value)`.
   - Tools that accept task_data receive it automatically — do not pass task_data.
   - INDEPENDENT calls MUST run concurrently inside a single `asyncio.gather`:
       a, b, c = await asyncio.gather(
           tools.x(),
           tools.y(),
           tools.z(),
       )
     Do NOT issue them as separate sequential `await`s — that wastes
     wall-clock time and can exceed the orchestration timeout.
   - Sequential `await` is ONLY for dependent chains where a later call's
     args use an earlier call's result:
       similar = await tools.find_similar_successful_tasks()
       if similar and not isinstance(similar, dict):
           logs = await tools.get_successful_job_logs(task_id=similar[0]["jediTaskID"])
   - If you have a mix of independent and dependent calls, group the
     independent ones in one `asyncio.gather`, then sequentially `await`
     the dependent ones.
   - Check for empty / error results before passing them to downstream calls.
   - Return a dict mapping descriptive labels to fetched values:
       return {"failed_job_log": failed, "successful_job_log": successful}
   - Only call tools in AVAILABLE TOOLS. No imports, no open(), no exec().

2) A list of capability gaps — investigation directions that no available tool
   addresses. Each entry:
     {"investigation": "what we need to investigate",
       "suggested_tool_capability": "the capability a future tool would need"}
   If every gap has a matching tool, leave the list empty.

Return ONLY a single JSON object with these two keys (no markdown fences,
no commentary). The JSON key MUST be the literal string `orchestration_code` —
do not abbreviate it to `orchest_code`, `code`, or any other variant; doing so
will cause your output to be discarded. Use the JSON-escaped string form for
`orchestration_code` (literal `\\n` for newlines):

{
  "orchestration_code": "    similar = await tools.find_similar_successful_tasks()\\n    ...",
  "capability_gaps": [
    {"investigation": "...", "suggested_tool_capability": "..."}
  ]
}
"""

TOOL_ORCHESTRATION_CODE_USER = """GAPS TO RESOLVE:
{gaps}

TASK CONTEXT:
{task_summary}

AVAILABLE TOOLS (callable as `await tools.<name>(keyword=value)` — keyword args only):
{tools_description}
"""

PANDA_SYSTEM_SUMMARY_SYSTEM = """You are reviewing PanDA ATLAS computing system
documentation to create a concise system knowledge summary for use as background
context by an AI that diagnoses task failures.

Write a 400-600 word factual summary of PanDA system behavior. Cover:

- The roles of key components (JEDI, brokerage, pilot, scout jobs, Rucio, etc.)
  and what each one decides or executes.
- How a task is decomposed into jobs: how JEDI generates jobs from a task,
  what relationship a single job has to the parent task, and how job-level
  outcomes (succeeded / failed / lost) aggregate into task-level state.
- How parameters flow through the system: which parameters users supply via
  submission flags (e.g. --memory, --nFilesPerJob), which the system derives
  from defaults or task structure, and which the system may adjust at runtime
  (e.g. via scout jobs and Dynamic Optimization). For each parameter you
  mention, describe the initial-value source AND whether / when the system
  overrides it. Do NOT categorize parameters into "automatic" vs "manual"
  buckets — many parameters are both.
- The temporal sequence of system actions during a task's lifetime: when
  brokerage runs and re-runs, when scout jobs are generated and execute,
  when Dynamic Optimization is triggered, and what events advance a task
  between major states. Make the ordering explicit (which step depends on
  which).

Style:
- Describe mechanisms, not actions. Do not write "Action:" guidance,
  "do not advise users", or any imperative direction to the reader.
- Be precise about which fields are user inputs, system-derived, or both.
- The AI consuming this summary derives correct recommendations from the facts;
  it does not need to be told what to advise.

Out of scope (fetched dynamically by doc search when relevant — do NOT cover):
- Specific task status meanings (surfaced via "task status {status}" query)
- Specific brokerage filter stages (surfaced when errorDialog mentions brokerage)
"""

PANDA_SYSTEM_SUMMARY_USER = """Documentation (overview and concept pages from PanDA docs):
{concept_docs}
"""
