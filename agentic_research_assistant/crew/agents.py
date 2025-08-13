from crewai import Agent
from langchain_openai import ChatOpenAI
from tools.langchain_tools import WebSearchTool, PlagiarismCheckTool, ReportGenerationTool
from config.settings import settings


def create_llm():
    """Create the language model for agents using centralized configuration"""
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key
    )


def create_research_planner():
    """Research Planning Agent - Analyzes queries and creates research strategies"""
    return Agent(
        role="Senior Research Strategist",
        goal="Break down complex research questions into focused, actionable sub-tasks that will comprehensively address the user's query",
        backstory="""You are a world-class research strategist with expertise in academic research, 
        information science, and knowledge synthesis. You excel at decomposing complex questions 
        into logical, searchable components that build toward comprehensive understanding.""",
        verbose=True,
        allow_delegation=True,
        llm=create_llm(),
        max_iter=3,
        memory=True
    )


def create_information_gatherer():
    """Information Gathering Agent - Searches and retrieves relevant data"""
    return Agent(
        role="Expert Information Researcher",
        goal="Gather comprehensive, high-quality information from web sources to answer specific research questions",
        backstory="""You are a skilled information researcher with expertise in web search optimization, 
        source evaluation, and content extraction. You know how to find authoritative, relevant sources 
        and extract the most valuable information for research purposes.""",
        verbose=True,
        allow_delegation=False,
        tools=[WebSearchTool()],
        llm=create_llm(),
        max_iter=5,
        memory=True
    )


def create_content_synthesizer():
    """Content Synthesis Agent - Analyzes and synthesizes information with citations"""
    return Agent(
        role="Expert Research Analyst & Writer",
        goal="Synthesize gathered information into comprehensive, well-cited analysis that directly addresses research questions",
        backstory="""You are a distinguished academic researcher and writer with expertise in 
        information synthesis, citation practices, and analytical writing. You excel at 
        combining multiple sources into coherent, insightful analysis while maintaining 
        rigorous citation standards.""",
        verbose=True,
        allow_delegation=False,
        llm=create_llm(),
        max_iter=3,
        memory=True
    )


def create_quality_controller():
    """Quality Control Agent - Checks originality and ensures academic integrity"""
    return Agent(
        role="Academic Integrity Specialist",
        goal="Ensure all content meets high standards of originality and academic integrity while maintaining factual accuracy",
        backstory="""You are an academic integrity expert with deep knowledge of plagiarism detection, 
        originality assessment, and ethical research practices. You ensure all content is 
        appropriately original while preserving accuracy and proper attribution.""",
        verbose=True,
        allow_delegation=False,
        tools=[PlagiarismCheckTool()],
        llm=create_llm(),
        max_iter=3,
        memory=True
    )


def create_content_editor():
    """Content Editor Agent - Rewrites content to improve originality"""
    return Agent(
        role="Expert Academic Editor",
        goal="Improve content originality through skillful rewriting while preserving meaning, accuracy, and citations",
        backstory="""You are a master academic editor with expertise in paraphrasing, 
        restructuring, and rewriting content to enhance originality without losing meaning. 
        You maintain the highest standards of academic writing while ensuring content 
        authenticity.""",
        verbose=True,
        allow_delegation=False,
        llm=create_llm(),
        max_iter=3,
        memory=True
    )


def create_report_specialist():
    """Report Generation Agent - Creates professional research reports"""
    return Agent(
        role="Professional Report Writer",
        goal="Create comprehensive, professional research reports that effectively communicate findings and insights",
        backstory="""You are an expert report writer with extensive experience in academic 
        and professional research communication. You excel at organizing complex information 
        into clear, engaging reports that serve both academic and business audiences.""",
        verbose=True,
        allow_delegation=False,
        tools=[ReportGenerationTool()],
        llm=create_llm(),
        max_iter=3,
        memory=True
    )
