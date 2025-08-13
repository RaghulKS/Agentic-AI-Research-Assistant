import json
import time
from pathlib import Path
from typing import Dict, Any

from crewai import Crew, Process
from crew.tasks import (
    create_planning_task,
    create_information_gathering_task,
    create_synthesis_task,
    create_quality_check_task,
    create_editing_task,
    create_report_generation_task
)
from crew.agents import (
    create_research_planner,
    create_information_gatherer,
    create_content_synthesizer,
    create_quality_controller,
    create_content_editor,
    create_report_specialist
)


class AgenticResearchCrew:
    """
    CrewAI-based research system that orchestrates multiple AI agents
    to conduct comprehensive research with quality control
    """
    
    def __init__(self, max_results_per_task: int = 5, verbose: bool = False):
        self.max_results_per_task = max_results_per_task
        self.verbose = verbose
        self.execution_log = []
        
    def _log_event(self, event: str, data: Dict[str, Any]):
        """Log execution events for transparency"""
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            "data": data
        }
        self.execution_log.append(log_entry)
        
        if self.verbose:
            print(f"[{event.upper()}] {data}")
    
    def conduct_research(self, research_query: str) -> Dict[str, Any]:
        """
        Conduct comprehensive research using the multi-agent CrewAI system
        
        Args:
            research_query: The research question to investigate
            
        Returns:
            Dict containing research results, reports, and metadata
        """
        self._log_event("research_start", {"query": research_query})
        
        try:
            # Initialize all agents
            agents = {
                "planner": create_research_planner(),
                "gatherer": create_information_gatherer(), 
                "synthesizer": create_content_synthesizer(),
                "quality_controller": create_quality_controller(),
                "editor": create_content_editor(),
                "reporter": create_report_specialist()
            }
            
            self._log_event("agents_initialized", {"agent_count": len(agents)})
            
            # Phase 1: Research Planning
            print("🧠 Phase 1: Research Planning...")
            planning_task = create_planning_task(research_query)
            planning_crew = Crew(
                agents=[agents["planner"]],
                tasks=[planning_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            research_plan = planning_crew.kickoff()
            self._log_event("planning_complete", {"plan_length": len(str(research_plan))})
            
            # Phase 2: Information Gathering
            print("🌐 Phase 2: Information Gathering...")
            gathering_task = create_information_gathering_task(str(research_plan), self.max_results_per_task)
            gathering_crew = Crew(
                agents=[agents["gatherer"]],
                tasks=[gathering_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            gathered_info = gathering_crew.kickoff()
            self._log_event("gathering_complete", {"info_length": len(str(gathered_info))})
            
            # Phase 3: Content Synthesis
            print("✍️ Phase 3: Content Synthesis...")
            synthesis_task = create_synthesis_task(research_query, str(gathered_info))
            synthesis_crew = Crew(
                agents=[agents["synthesizer"]],
                tasks=[synthesis_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            synthesized_content = synthesis_crew.kickoff()
            self._log_event("synthesis_complete", {"content_length": len(str(synthesized_content))})
            
            # Phase 4: Quality Control
            print("🔍 Phase 4: Quality Assessment...")
            quality_task = create_quality_check_task(str(synthesized_content), str(gathered_info))
            quality_crew = Crew(
                agents=[agents["quality_controller"]],
                tasks=[quality_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            quality_report = quality_crew.kickoff()
            self._log_event("quality_check_complete", {"report_length": len(str(quality_report))})
            
            # Phase 5: Content Editing (if needed)
            print("📝 Phase 5: Content Enhancement...")
            editing_task = create_editing_task(str(synthesized_content), str(quality_report))
            editing_crew = Crew(
                agents=[agents["editor"]],
                tasks=[editing_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            final_content = editing_crew.kickoff()
            self._log_event("editing_complete", {"final_length": len(str(final_content))})
            
            # Phase 6: Report Generation
            print("📊 Phase 6: Report Generation...")
            metadata = json.dumps({
                "total_phases": 6,
                "research_query": research_query,
                "processing_time": time.time(),
                "agent_count": len(agents)
            })
            
            report_task = create_report_generation_task(research_query, str(final_content), metadata)
            report_crew = Crew(
                agents=[agents["reporter"]],
                tasks=[report_task],
                process=Process.sequential,
                verbose=self.verbose
            )
            
            final_report = report_crew.kickoff()
            self._log_event("report_generation_complete", {"report_data": str(final_report)})
            
            # Save execution log
            self._save_execution_log()
            
            return {
                "research_query": research_query,
                "research_plan": str(research_plan),
                "gathered_information": str(gathered_info),
                "synthesized_content": str(synthesized_content),
                "quality_report": str(quality_report),
                "final_content": str(final_content),
                "final_report": str(final_report),
                "execution_log": self.execution_log,
                "metadata": {
                    "phases_completed": 6,
                    "agents_used": list(agents.keys()),
                    "processing_time": time.time(),
                    "success": True
                }
            }
            
        except Exception as e:
            self._log_event("research_error", {"error": str(e), "type": type(e).__name__})
            raise
    
    def _save_execution_log(self):
        """Save the execution log for transparency and debugging"""
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            
        log_file = data_dir / f"crewai_execution_{int(time.time())}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.execution_log, f, ensure_ascii=False, indent=2)
            
        if self.verbose:
            print(f"📁 Execution log saved to: {log_file}")
    
    def get_agent_interactions(self) -> Dict[str, Any]:
        """Get summary of agent interactions and decision points"""
        return {
            "total_events": len(self.execution_log),
            "events_by_type": {},
            "processing_timeline": [
                {"event": entry["event"], "timestamp": entry["timestamp"]} 
                for entry in self.execution_log
            ]
        }
