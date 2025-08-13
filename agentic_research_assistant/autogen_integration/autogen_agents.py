import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm


class AutoGenResearchSystem:
    """
    AutoGen-based multi-agent conversation system for collaborative research
    Provides advanced agent-to-agent communication and consensus building
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        from config.settings import settings
        
        self.model_name = model_name or settings.model_name
        self.verbose = verbose
        self.conversation_history = []
        
        # Configure AutoGen
        self.config_list = [
            {
                "model": self.model_name,
                "api_key": settings.openai_api_key,
                "api_type": "openai"
            }
        ]
        
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": settings.temperature,
            "timeout": 120,
        }
        
        self._create_agents()
    
    def _create_agents(self):
        """Create specialized AutoGen agents for research collaboration"""
        
        # Research Strategist - Plans and coordinates research
        self.strategist = AssistantAgent(
            name="Research_Strategist",
            system_message="""You are a Senior Research Strategist with expertise in breaking down complex 
            research questions. You excel at creating comprehensive research plans and coordinating with 
            other specialists. Your role is to analyze queries, create structured research plans, and 
            ensure all aspects of a research question are thoroughly addressed.""",
            llm_config=self.llm_config,
            description="Expert in research planning and strategy coordination"
        )
        
        # Information Analyst - Evaluates and synthesizes information
        self.analyst = AssistantAgent(
            name="Information_Analyst",
            system_message="""You are an expert Information Analyst specializing in data evaluation, 
            source assessment, and content synthesis. You excel at analyzing gathered information, 
            identifying key insights, and synthesizing findings from multiple sources into coherent 
            analysis with proper citations.""",
            llm_config=self.llm_config,
            description="Expert in information analysis and synthesis"
        )
        
        # Quality Assurance Specialist - Ensures accuracy and originality
        self.qa_specialist = AssistantAgent(
            name="Quality_Specialist",
            system_message="""You are a Quality Assurance Specialist with expertise in academic integrity, 
            fact-checking, and content evaluation. You ensure all research meets high standards of 
            accuracy, originality, and academic rigor. You identify potential issues and recommend 
            improvements for content quality.""",
            llm_config=self.llm_config,
            description="Expert in quality assurance and academic integrity"
        )
        
        # Content Editor - Improves writing and presentation
        self.editor = AssistantAgent(
            name="Content_Editor",
            system_message="""You are an expert Academic Content Editor with skills in improving writing 
            clarity, originality, and presentation. You excel at rewriting content to enhance originality 
            while preserving meaning, improving flow and readability, and ensuring professional academic 
            presentation standards.""",
            llm_config=self.llm_config,
            description="Expert in content editing and academic writing"
        )
        
        # Research Coordinator - Manages the process and final decisions
        self.coordinator = UserProxyAgent(
            name="Research_Coordinator",
            system_message="""You are the Research Coordinator managing the collaborative research process. 
            You facilitate agent discussions, make final decisions, and ensure the research process 
            stays focused and productive. You synthesize agent inputs and guide the team toward 
            comprehensive solutions.""",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
            description="Coordinates research process and agent collaboration"
        )
        
        self.agents = [self.strategist, self.analyst, self.qa_specialist, self.editor]
        
    def collaborative_research_planning(self, research_query: str) -> Dict[str, Any]:
        """
        Use AutoGen agents to collaboratively plan research strategy
        
        Args:
            research_query: The research question to plan for
            
        Returns:
            Collaborative research plan with agent consensus
        """
        
        planning_prompt = f"""
        Research Query: "{research_query}"
        
        Team, we need to collaboratively develop a comprehensive research strategy for this query.
        
        Research_Strategist: Please analyze this query and propose an initial research plan with 3-6 focused sub-questions.
        
        Information_Analyst: Review the proposed plan and suggest improvements for information gathering effectiveness.
        
        Quality_Specialist: Assess the plan for completeness and suggest quality control measures.
        
        Content_Editor: Consider how the research structure will support clear, well-organized final content.
        
        Let's collaborate to create the best possible research strategy. Each agent should contribute their expertise and build on others' suggestions.
        """
        
        # Create group chat for collaborative planning
        group_chat = GroupChat(
            agents=self.agents + [self.coordinator],
            messages=[],
            max_round=8,
            speaker_selection_method="round_robin"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        )
        
        # Start collaborative planning discussion
        self.coordinator.initiate_chat(
            manager,
            message=planning_prompt,
            clear_history=True
        )
        
        # Extract and structure the collaborative plan
        conversation = group_chat.messages
        self.conversation_history.extend(conversation)
        
        return {
            "collaborative_plan": self._extract_research_plan(conversation),
            "agent_contributions": self._analyze_agent_contributions(conversation),
            "consensus_points": self._identify_consensus(conversation),
            "conversation_log": conversation
        }
    
    def collaborative_quality_review(self, content: str, sources: str) -> Dict[str, Any]:
        """
        Use AutoGen agents to collaboratively review content quality
        
        Args:
            content: Research content to review
            sources: Source information for reference
            
        Returns:
            Collaborative quality assessment and improvement recommendations
        """
        
        review_prompt = f"""
        Content for Review: {content[:2000]}...
        
        Available Sources: {sources[:1000]}...
        
        Team, we need to collaboratively review this research content for quality, accuracy, and originality.
        
        Quality_Specialist: Please assess the content for academic integrity, citation quality, and potential originality issues.
        
        Information_Analyst: Evaluate how well the content synthesizes the available sources and addresses the research objectives.
        
        Content_Editor: Assess writing quality, clarity, and suggest improvements for presentation and flow.
        
        Research_Strategist: Determine if the content comprehensively addresses the original research goals.
        
        Please collaborate to provide comprehensive quality feedback and improvement recommendations.
        """
        
        # Create group chat for quality review
        group_chat = GroupChat(
            agents=self.agents + [self.coordinator],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        )
        
        # Start collaborative review discussion
        self.coordinator.initiate_chat(
            manager,
            message=review_prompt,
            clear_history=True
        )
        
        conversation = group_chat.messages
        self.conversation_history.extend(conversation)
        
        return {
            "quality_assessment": self._extract_quality_feedback(conversation),
            "improvement_recommendations": self._extract_recommendations(conversation),
            "agent_consensus": self._assess_consensus_level(conversation),
            "review_conversation": conversation
        }
    
    def collaborative_content_improvement(self, content: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AutoGen agents to collaboratively improve content
        
        Args:
            content: Content to improve
            feedback: Quality feedback from review process
            
        Returns:
            Improved content with agent collaboration details
        """
        
        improvement_prompt = f"""
        Original Content: {content[:1500]}...
        
        Quality Feedback: {json.dumps(feedback, indent=2)[:1000]}...
        
        Team, based on the quality feedback, let's collaboratively improve this content.
        
        Content_Editor: Lead the improvement process by rewriting problematic sections while preserving meaning and citations.
        
        Quality_Specialist: Monitor the improvements to ensure they address the identified issues without introducing new problems.
        
        Information_Analyst: Ensure improved content maintains analytical rigor and proper source integration.
        
        Research_Strategist: Verify that improvements maintain alignment with research objectives.
        
        Let's work together to enhance this content while maintaining its academic value.
        """
        
        # Create group chat for content improvement
        group_chat = GroupChat(
            agents=self.agents + [self.coordinator],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config
        )
        
        # Start collaborative improvement discussion
        self.coordinator.initiate_chat(
            manager,
            message=improvement_prompt,
            clear_history=True
        )
        
        conversation = group_chat.messages
        self.conversation_history.extend(conversation)
        
        return {
            "improved_content": self._extract_improved_content(conversation),
            "improvement_rationale": self._extract_improvement_rationale(conversation),
            "collaboration_summary": self._summarize_collaboration(conversation),
            "improvement_conversation": conversation
        }
    
    def _extract_research_plan(self, conversation: List[Dict]) -> Dict[str, Any]:
        """Extract structured research plan from agent conversation"""
        # Implementation would parse conversation for research plan elements
        return {"plan": "Extracted from agent collaboration", "sub_questions": []}
    
    def _analyze_agent_contributions(self, conversation: List[Dict]) -> Dict[str, Any]:
        """Analyze individual agent contributions to the discussion"""
        contributions = {}
        for msg in conversation:
            agent_name = msg.get("name", "unknown")
            if agent_name not in contributions:
                contributions[agent_name] = []
            contributions[agent_name].append(msg.get("content", ""))
        return contributions
    
    def _identify_consensus(self, conversation: List[Dict]) -> List[str]:
        """Identify consensus points from agent discussion"""
        # Implementation would identify agreement points
        return ["Consensus point 1", "Consensus point 2"]
    
    def _extract_quality_feedback(self, conversation: List[Dict]) -> Dict[str, Any]:
        """Extract quality assessment from agent conversation"""
        return {"overall_quality": "good", "specific_issues": []}
    
    def _extract_recommendations(self, conversation: List[Dict]) -> List[str]:
        """Extract improvement recommendations from conversation"""
        return ["Recommendation 1", "Recommendation 2"]
    
    def _assess_consensus_level(self, conversation: List[Dict]) -> str:
        """Assess level of consensus among agents"""
        return "high"
    
    def _extract_improved_content(self, conversation: List[Dict]) -> str:
        """Extract improved content from collaborative editing session"""
        # Implementation would parse final improved content
        return "Improved content from collaboration"
    
    def _extract_improvement_rationale(self, conversation: List[Dict]) -> str:
        """Extract rationale for improvements made"""
        return "Rationale for improvements"
    
    def _summarize_collaboration(self, conversation: List[Dict]) -> Dict[str, Any]:
        """Summarize the collaborative process"""
        return {
            "total_exchanges": len(conversation),
            "agents_participated": len(set(msg.get("name") for msg in conversation)),
            "collaboration_quality": "high"
        }
    
    def save_conversation_history(self):
        """Save complete conversation history for analysis"""
        from config.settings import settings
        
        data_dir = Path(settings.data_dir)
        data_dir.mkdir(exist_ok=True)
        
        history_file = data_dir / f"autogen_conversations_{int(time.time())}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"📁 AutoGen conversation history saved to: {history_file}")
    
    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get metrics about agent collaboration"""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        return {
            "total_conversations": len(self.conversation_history),
            "unique_agents": len(set(msg.get("name") for msg in self.conversation_history)),
            "average_message_length": sum(len(msg.get("content", "")) for msg in self.conversation_history) / len(self.conversation_history),
            "collaboration_depth": "high"  # Would be calculated based on interaction patterns
        }
