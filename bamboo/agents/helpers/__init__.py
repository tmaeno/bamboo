"""Non-agent support modules for the agents package.

These are the agent subsystem's plumbing — distinct from the agent classes in the
parent :mod:`bamboo.agents` package and from the generic, app-wide helpers in
:mod:`bamboo.utils` (which carry no dependency on the agent layer):

* :mod:`~bamboo.agents.helpers.deps` — dependency-injection factory (``build_deps``,
  ``resolve_task_data``).
* :mod:`~bamboo.agents.helpers.orchestration` — the LLM-authored code sandbox
  (``ToolProxy``, ``run_orchestration_code``).
* :mod:`~bamboo.agents.helpers.tool_selection` — shared tool-list renderer
  (``render_tools``) and budget-gated retrieval selection for orchestration prompts.
* :mod:`~bamboo.agents.helpers.internal_tools` /
  :mod:`~bamboo.agents.helpers.procedure_tools` — tool registries.
* :mod:`~bamboo.agents.helpers.context_prefetch` /
  :mod:`~bamboo.agents.helpers.task_data_bootstrap` — context/graph priming.
"""
