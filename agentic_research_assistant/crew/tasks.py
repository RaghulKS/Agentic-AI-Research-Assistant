from crewai import Task
from crew.agents import (
    create_research_planner,
    create_information_gatherer,
    create_content_synthesizer,
    create_quality_controller,
    create_content_editor,
    create_report_specialist
)


def create_planning_task(research_query: str):
    """Task for breaking down the research query into sub-questions"""
    return Task(
        description=f"""
        Analyze the research query: "{research_query}"
        
        Break this complex query into 3-6 focused, searchable sub-questions that will 
        comprehensively address the user's research needs. Each sub-question should:
        
        1. Address a distinct aspect of the main query
        2. Be specific enough to yield targeted search results
        3. Build toward comprehensive understanding
        4. Be formulated as clear, searchable questions
        
        Consider these dimensions:
        - Core definitions and concepts
        - Current state and recent developments  
        - Key stakeholders and entities
        - Mechanisms, processes, or methodologies
        - Impacts, implications, and outcomes
        - Future trends and predictions
        
        Output your analysis as a structured research plan with numbered sub-questions
        and brief search strategies for each.
        """,
        agent=create_research_planner(),
        expected_output="A structured research plan with 3-6 specific sub-questions and search strategies"
    )


def create_information_gathering_task(research_plan: str, max_results: int = 5):
    """Task for gathering information based on the research plan"""
    return Task(
        description=f"""
        Based on the research plan: {research_plan}
        
        For each sub-question in the plan, conduct comprehensive web searches to gather 
        relevant, high-quality information. For each search:
        
        1. Formulate effective search queries
        2. Gather up to {max_results} relevant sources per sub-question
        3. Extract and evaluate content quality
        4. Focus on authoritative, recent, and comprehensive sources
        5. Ensure content is substantial and relevant
        
        Compile all gathered information in a structured format, organizing findings 
        by sub-question with source attribution.
        """,
        agent=create_information_gatherer(),
        expected_output="Comprehensive information gathered for each research sub-question with source details"
    )


def create_synthesis_task(research_query: str, gathered_info: str):
    """Task for synthesizing information into comprehensive analysis"""
    return Task(
        description=f"""
        Original research query: "{research_query}"
        Gathered information: {gathered_info}
        
        Synthesize the gathered information into a comprehensive analysis that directly 
        addresses the original research query. Your synthesis should:
        
        1. Address each aspect of the original query comprehensively
        2. Integrate findings from multiple sources coherently
        3. Include proper in-text citations [1], [2], etc.
        4. Organize information logically with clear sections
        5. Highlight key insights, patterns, and conclusions
        6. Maintain academic rigor and objectivity
        7. Ensure all claims are properly supported and cited
        
        Structure your response with clear headings and comprehensive coverage 
        of all research dimensions.
        """,
        agent=create_content_synthesizer(),
        expected_output="Comprehensive synthesized analysis with proper citations addressing the original query"
    )


def create_quality_check_task(synthesized_content: str, source_data: str):
    """Task for checking content quality and originality"""
    return Task(
        description=f"""
        Content to review: {synthesized_content}
        Source materials: {source_data}
        
        Conduct a thorough quality and originality assessment of the synthesized content:
        
        1. Analyze content for potential plagiarism using semantic similarity
        2. Check for appropriate citation practices
        3. Identify any segments that may be too similar to source material
        4. Assess overall originality score
        5. Flag specific segments that need improvement
        6. Provide recommendations for enhancing originality
        
        Generate a detailed quality report with originality metrics and specific 
        recommendations for improvement.
        """,
        agent=create_quality_controller(),
        expected_output="Quality assessment report with originality score and specific improvement recommendations"
    )


def create_editing_task(content: str, quality_report: str):
    """Task for improving content originality through expert editing"""
    return Task(
        description=f"""
        Content to edit: {content}
        Quality assessment: {quality_report}
        
        Based on the quality assessment, improve the content's originality while 
        preserving accuracy and meaning:
        
        1. Rewrite flagged segments to improve originality
        2. Restructure sentences and paragraphs for better flow
        3. Use varied vocabulary and sentence patterns
        4. Maintain all factual accuracy and citations
        5. Preserve the analytical depth and insights
        6. Ensure the content remains comprehensive and coherent
        
        If no editing is needed, return the original content with confirmation 
        of its high quality.
        """,
        agent=create_content_editor(),
        expected_output="Improved content with enhanced originality while maintaining accuracy and citations"
    )


def create_report_generation_task(research_query: str, final_content: str, metadata: str):
    """Task for generating the final professional report"""
    return Task(
        description=f"""
        Research Query: "{research_query}"
        Final Content: {final_content}
        Research Metadata: {metadata}
        
        Create a comprehensive, professional research report that effectively 
        communicates the research findings:
        
        1. Structure the content with clear, logical organization
        2. Include an executive summary highlighting key findings
        3. Add proper headings and sections for readability
        4. Include research methodology and source information
        5. Add quality metrics and originality assessment
        6. Ensure professional formatting and presentation
        7. Generate both Markdown and PDF versions
        
        The report should serve both academic and professional audiences with 
        clear communication of insights and findings.
        """,
        agent=create_report_specialist(),
        expected_output="Professional research report generated in Markdown and PDF formats with comprehensive findings"
    )
