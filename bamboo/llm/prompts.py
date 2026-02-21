"""LLM prompt templates for all pipeline stages.

All prompts are plain strings with ``{placeholder}`` format fields.
Call ``.format(...)`` to produce the final prompt before passing to the LLM.

Prompt constants
----------------
``EXTRACTION_PROMPT``
    General-purpose knowledge-graph extraction from raw text.
    Used by ``LLMExtractionStrategy``.

``EMAIL_EXTRACTION_PROMPT``
    Extracts ``Cause``, ``Resolution``, and ``Task_Context`` nodes from an
    email thread.  Used by :class:`~bamboo.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`.

``CAUSE_RESOLUTION_CANONICALIZE_PROMPT``
    Normalises a raw Cause/Resolution name into a stable canonical phrase,
    optionally matching against existing names from the vector DB.

``ERROR_CATEGORY_LABEL_PROMPT``
    Converts a raw error message into a short CamelCase error-category label.
    Used by :class:`~bamboo.extractors.panda_knowledge_extractor.ErrorCategoryStore`.

``SUMMARIZATION_PROMPT``
    Produces a narrative summary of a knowledge graph for vector indexing.
    Used by :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`.

``CAUSE_IDENTIFICATION_PROMPT``
    Root-cause analysis from graph and vector DB evidence.
    Used by :class:`~bamboo.agents.reasoning_navigator.ReasoningAgent`.

``EMAIL_GENERATION_PROMPT``
    Drafts a professional resolution email for a task submitter.
    Used by :class:`~bamboo.agents.reasoning_navigator.ReasoningAgent`.

``FEATURE_EXTRACTION_PROMPT``
    Extracts structured features from raw task and external data.
    (Legacy; superseded by ``PandaKnowledgeExtractor`` for Panda tasks.)
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


SUMMARIZATION_PROMPT = """You are a technical documentation expert. Create a comprehensive yet concise entry of the following knowledge graph.

Knowledge Graph:
{graph_data}

Create a entry that:
1. Highlights the main errors and their causes
2. Describes the key contributing factors (environment, features, components)
3. Outlines the recommended resolutions
4. Captures important patterns and insights

The entry should be:
- Clear and technical
- Searchable (include key terms)
- Contextual (provide enough background)
- Actionable (highlight what matters for troubleshooting)

Summary:
"""

CAUSE_IDENTIFICATION_PROMPT = """You are a root cause analysis expert. Analyze the task information and database results to identify the most likely root cause.

Task Information:
{task_info}

External Information:
{external_info}

Graph Database Results (possible causes and resolutions):
{graph_results}

Vector Database Results (similar cases):
{vector_results}

Based on this information:
1. Identify the most likely root cause
2. Assess your confidence level (0-1)
3. Recommend the best resolution
4. Explain your reasoning

Respond with a JSON object:
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

Root Cause Analysis:
{analysis}

The email should:
1. Start with a brief acknowledgment of the issue
2. Explain the root cause in clear, non-technical language when possible
3. Provide step-by-step resolution instructions
4. Offer to help if they have questions
5. Be professional but friendly

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

Relationships to emit (between extracted nodes only):
- Cause -[solved_by]-> Resolution
- Task_Context -[contribute_to]-> Cause  (only when the context directly supports a cause)

Canonicalization:
- If the same cause or resolution is mentioned multiple times, emit it only once.
- Keep names concise and general (no incident-specific IDs, paths, or dataset names).

Email thread:
{email_text}

Output ONLY a valid JSON object — no explanation, no markdown fences:
{{
  "nodes": [
    {{
      "node_type": "Cause|Resolution|Task_Context",
      "name": "...",
      "description": "...",
      "metadata": {{}},
      "steps": []
    }}
  ],
  "relationships": [
    {{
      "source_name": "...",
      "target_name": "...",
      "relation_type": "solved_by|contribute_to",
      "confidence": 0.0
    }}
  ]
}}
"""

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

ERROR_CATEGORY_LABEL_PROMPT = """You are an error classification expert.

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


