# Agentic Research Assistant Architecture

## Directory Layout

- agentic_research_assistant/: Core application package
  - agents/: Legacy agents
  - crew/: CrewAI agents, tasks, and orchestrator
  - autogen_integration/: AutoGen collaboration layer
  - tools/: Tools (LangChain tools, web/pdf readers)
  - config/: Centralized configuration
  - prompts/: Prompt templates
  - data/: Runtime data (ignored)
  - reports/: Outputs (ignored)

## Execution Flows

- Legacy: main.py (custom agents)
- CrewAI: crew/research_crew.py
- AutoGen: autogen_integration/autogen_agents.py
- Hybrid: main.py orchestrates all with flags

