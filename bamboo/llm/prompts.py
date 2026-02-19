"""LLM prompts for various tasks."""

EXTRACTION_PROMPT = """You are a knowledge extraction expert. Your task is to extract structured knowledge from the provided information and construct a knowledge graph.

The graph should follow this schema:
- Error -[indicate]-> Cause
- Environment -[associated_with]-> Cause
- Task_Feature -[contribute_to]-> Cause
- Task_Context -[contribute_to]-> Cause
- Component -[originated_from]-> Cause
- Cause -[solved_by]-> Resolution

Node Types:
- Error: Main error message
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
- Error, Environment, Component: output a single, normalised canonical name.
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
      "node_type": "Error|Environment|Task_Feature|Task_Context|Component|Cause|Resolution",
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
