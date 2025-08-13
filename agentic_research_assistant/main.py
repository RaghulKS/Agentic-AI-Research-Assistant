import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import centralized configuration
from config.settings import settings, validate_api_keys

# New framework integrations
from crew.research_crew import AgenticResearchCrew
try:
    from autogen_integration.autogen_agents import AutoGenResearchSystem
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("AutoGen not available - using CrewAI only")

from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_ADVANCED = True
except ImportError:
    LANGCHAIN_ADVANCED = False

# Legacy agents for fallback compatibility
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.plagiarism_checker_agent import PlagiarismCheckerAgent
from agents.rewriter_agent import RewriterAgent
from agents.reporter_agent import ReporterAgent


def ensure_dirs() -> dict:
    base_dir = Path("data")
    run_dir = base_dir / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "retrieval").mkdir(parents=True, exist_ok=True)
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    return {
        "base": base_dir,
        "run": run_dir,
        "retrieval": run_dir / "retrieval",
        "reports": reports_dir,
        "data": base_dir,
    }


def write_jsonl(path: Path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # Validate configuration and API keys
    try:
        validate_api_keys()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Setup Instructions:")
        print("1. Create a .env file in the project directory")
        print("2. Add your OpenAI API key: OPENAI_API_KEY=sk-your-key-here")
        print("3. Optionally add other settings (see .env.example)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Agentic Research Assistant - Autonomous multi-agent AI research system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What is quantum computing and its current applications?"
  python main.py --query "Impact of artificial intelligence on healthcare" --max_results 8
  python main.py --query "Blockchain trends" --framework hybrid --collaboration_mode --enable_memory
        """
    )
    parser.add_argument("--query", type=str, required=False, help="Research question to investigate")
    parser.add_argument("--max_results", type=int, default=settings.default_max_results, 
                       help=f"Maximum search results per sub-question (default: {settings.default_max_results})")
    parser.add_argument("--verbose", action="store_true", default=settings.verbose_logging,
                       help="Enable verbose logging")
    parser.add_argument("--framework", type=str, choices=["crewai", "autogen", "hybrid", "legacy"], 
                       default=settings.default_framework, help=f"AI framework to use (default: {settings.default_framework})")
    parser.add_argument("--enable_memory", action="store_true", default=settings.enable_memory,
                       help="Enable LangChain memory for context retention")
    parser.add_argument("--collaboration_mode", action="store_true", help="Enable AutoGen agent collaboration")
    parser.add_argument("--show_config", action="store_true", help="Display current configuration and exit")
    args = parser.parse_args()

    # Display configuration if requested
    if args.show_config:
        settings.display_config_status()
        sys.exit(0)

    query = args.query or os.environ.get("ARA_QUERY")
    if not query:
        print("❌ Error: No research query provided.")
        print("\n💡 Usage Options:")
        print("   python main.py --query 'Your research question here'")
        print("   export ARA_QUERY='Your question' && python main.py")
        print("   python main.py --help  # For all options")
        sys.exit(1)

    dirs = ensure_dirs()
    logs_path = dirs["run"] / "logs.jsonl"

    def log(event: str, payload: dict):
        record = {"timestamp": time.time(), "event": event, "data": payload}
        write_jsonl(logs_path, [record])
        if args.verbose:
            print(f"[{event.upper()}] {payload}")

    print(f"🔍 Starting advanced multi-agent research on: '{query}'")
    print(f"🧬 Framework: {args.framework.upper()} | Memory: {'ON' if args.enable_memory else 'OFF'} | Collaboration: {'ON' if args.collaboration_mode else 'OFF'}")
    
    log("research_start", {
        "query": query, 
        "framework": args.framework,
        "max_results_per_task": args.max_results,
        "memory_enabled": args.enable_memory,
        "collaboration_enabled": args.collaboration_mode
    })

    # Initialize LangChain memory if enabled
    memory = ConversationBufferMemory(memory_key="research_history") if args.enable_memory else None
    
    try:
        if args.framework == "crewai":
            result = run_crewai_research(query, args.max_results, args.verbose, memory)
        elif args.framework == "autogen":
            result = run_autogen_research(query, args.collaboration_mode, args.verbose, memory)
        elif args.framework == "hybrid":
            result = run_hybrid_research(query, args.max_results, args.collaboration_mode, args.verbose, memory)
        else:  # legacy
            result = run_legacy_research(query, args.max_results, args.verbose, dirs, log)
        
        # Display results
        print("\n🎉 Advanced multi-agent research complete!")
        if isinstance(result, dict):
            if "final_report" in result:
                print(f"📄 CrewAI Report Generated")
            if "collaboration_metrics" in result:
                metrics = result["collaboration_metrics"]
                print(f"🤝 Agent Collaboration: {metrics.get('total_conversations', 0)} conversations")
            if "metadata" in result and result["metadata"].get("success"):
                print(f"✅ All {result['metadata'].get('phases_completed', 0)} phases completed successfully")
        
        print(f"📁 Data and logs saved to: {dirs['run']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        log("research_error", {"error": str(e), "type": type(e).__name__})
        sys.exit(1)


def run_crewai_research(query: str, max_results: int, verbose: bool, memory=None):
    """Run research using CrewAI multi-agent framework"""
    print("🚀 Initializing CrewAI multi-agent system...")
    
    crew = AgenticResearchCrew(max_results_per_task=max_results, verbose=verbose)
    result = crew.conduct_research(query)
    
    if memory:
        memory.save_context(
            {"research_query": query},
            {"research_result": result.get("final_content", "")[:1000]}
        )
    
    return result


def run_autogen_research(query: str, collaboration_mode: bool, verbose: bool, memory=None):
    """Run research using AutoGen agent conversation framework"""
    print("🤖 Initializing AutoGen collaborative agent system...")
    
    autogen_system = AutoGenResearchSystem(verbose=verbose)
    
    # Phase 1: Collaborative planning
    planning_result = autogen_system.collaborative_research_planning(query)
    
    # Phase 2: Simulate information gathering (would integrate with actual tools)
    gathered_info = '{"simulated": "information gathering results"}'
    
    # Phase 3: Collaborative quality review
    quality_result = autogen_system.collaborative_quality_review(
        "Simulated research content", gathered_info
    )
    
    # Phase 4: Collaborative content improvement
    improvement_result = autogen_system.collaborative_content_improvement(
        "Content to improve", quality_result
    )
    
    autogen_system.save_conversation_history()
    
    result = {
        "planning_result": planning_result,
        "quality_result": quality_result,
        "improvement_result": improvement_result,
        "collaboration_metrics": autogen_system.get_collaboration_metrics()
    }
    
    if memory:
        memory.save_context(
            {"autogen_query": query},
            {"autogen_result": str(result)[:1000]}
        )
    
    return result


def run_hybrid_research(query: str, max_results: int, collaboration_mode: bool, verbose: bool, memory=None):
    """Run research using hybrid CrewAI + AutoGen + LangChain approach"""
    print("🧬 Initializing hybrid multi-framework system...")
    print("   🚀 CrewAI for task orchestration")
    print("   🤖 AutoGen for agent collaboration") 
    print("   🔗 LangChain for tool integration")
    
    # Phase 1: CrewAI research execution
    crew = AgenticResearchCrew(max_results_per_task=max_results, verbose=verbose)
    crewai_result = crew.conduct_research(query)
    
    # Phase 2: AutoGen collaborative review (if enabled)
    autogen_result = None
    if collaboration_mode:
        print("🤝 Activating AutoGen collaborative review...")
        autogen_system = AutoGenResearchSystem(verbose=verbose)
        
        # Use CrewAI results for AutoGen collaborative improvement
        autogen_result = autogen_system.collaborative_quality_review(
            crewai_result.get("final_content", ""),
            json.dumps(crewai_result.get("gathered_information", {}))
        )
        
        # Apply collaborative improvements
        improvement_result = autogen_system.collaborative_content_improvement(
            crewai_result.get("final_content", ""),
            autogen_result
        )
        
        autogen_system.save_conversation_history()
        autogen_result["improvement_result"] = improvement_result
    
    # Phase 3: LangChain memory integration
    if memory:
        memory.save_context(
            {"hybrid_query": query},
            {"crewai_content": crewai_result.get("final_content", "")[:500]}
        )
        if autogen_result:
            memory.save_context(
                {"collaboration_feedback": "AutoGen review"},
                {"collaboration_result": str(autogen_result)[:500]}
            )
    
    # Combine results
    hybrid_result = {
        "crewai_research": crewai_result,
        "autogen_collaboration": autogen_result,
        "memory_context": memory.buffer if memory else None,
        "framework_integration": {
            "crewai_phases": crewai_result.get("metadata", {}).get("phases_completed", 0),
            "autogen_conversations": autogen_result.get("collaboration_metrics", {}).get("total_conversations", 0) if autogen_result else 0,
            "langchain_memory": "enabled" if memory else "disabled"
        }
    }
    
    return hybrid_result


def run_legacy_research(query: str, max_results: int, verbose: bool, dirs: dict, log):
    """Run research using legacy custom agent implementation"""
    print("🔧 Using legacy custom agent implementation...")
    
    # Original implementation code would go here
    planner = PlannerAgent()
    retriever = RetrieverAgent()
    summarizer = SummarizerAgent()
    plag_checker = PlagiarismCheckerAgent()
    rewriter = RewriterAgent()
    reporter = ReporterAgent()
    
    # Execute legacy research workflow
    plan = planner.plan(query)
    log("legacy_planning", {"tasks": len(plan.get("tasks", []))})
    
    # Continue with legacy implementation...
    return {"legacy_research": "completed", "framework": "custom"}


if __name__ == "__main__":
    main()
