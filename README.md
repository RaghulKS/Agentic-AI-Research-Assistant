Agentic Research Assistant

A fully autonomous, multi-agent AI research system powered by **CrewAI**, **LangChain**, **AutoGen**, and **GPT-4o**. The system decomposes complex queries into sub-questions, retrieves information from web sources and PDFs, synthesizes results with citations, checks for plagiarism, and rewrites flagged content while preserving accuracy.

Framework Integration

This project showcases integration of cutting-edge AI frameworks:

- **CrewAI**: Multi-agent orchestration and task management
- **LangChain**: Tool integration, memory management, and chains  
- **AutoGen**: Agent-to-agent collaboration and consensus building
- **GPT-4o**: Advanced reasoning, planning, and content generation
- **scikit-learn**: Semantic similarity analysis for plagiarism detection
- **spaCy**: Advanced NLP processing
- **DuckDuckGo API**: Real-time web search capabilities
- **ReportLab**: Professional PDF generation

Features

- **Intelligent Planning**: AI agents automatically decompose complex queries into focused research strategies
- **Web & PDF Research**: Advanced web scraping with PDF content extraction using PyMuPDF
- ** Synthesis**: Multi-agent content synthesis with proper academic citations
- **Originality Checking**: Semantic similarity analysis to detect potential plagiarism
- **Smart Rewriting**: AI-powered content rewriting to improve originality while preserving meaning
- ** Professional Reports**: Clean Markdown and PDF reports with executive summaries
- ** Agent Collaboration**: AutoGen-powered agent discussions for quality improvement
- **Memory Integration**: LangChain memory for context retention across research sessions
- ** Full Autonomy**: End-to-end execution with comprehensive logging and transparency

Advanced Architecture

Multi-Framework Integration
- **CrewAI Framework**: Orchestrates specialized research agents in sequential workflows
- **LangChain Tools**: Provides standardized tool interfaces and memory management
- **AutoGen Collaboration**: Enables agent-to-agent discussions for quality control
- **Hybrid Execution**: Combines all frameworks for maximum capability

Specialized AI Agents
- **Research Strategist**: Plans research approach and decomposes queries (CrewAI)
- **Information Gatherer**: Conducts web search and content extraction (LangChain Tools)
- **Content Synthesizer**: Analyzes and synthesizes information with citations (CrewAI)
- **Quality Controller**: Checks originality and academic integrity (LangChain Tools)
- **Content Editor**: Rewrites content for improved originality (CrewAI)
- **Report Specialist**: Generates professional research reports (CrewAI)

Quick Start

1. Install Dependencies
```bash
python -m venv venv
./venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

2. Configure API Keys
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run Research
```bash
python main.py --query "What is quantum computing and its current applications?"
```

Advanced Usage Examples

```bash
# CrewAI Framework (Multi-agent orchestration)
python main.py --query "Impact of artificial intelligence on healthcare" --framework crewai

# AutoGen Framework (Agent collaboration)
python main.py --query "Climate change solutions" --framework autogen --collaboration_mode

# Hybrid Framework (CrewAI + AutoGen + LangChain)
python main.py --query "Blockchain technology trends" --framework hybrid --collaboration_mode --enable_memory

# Legacy Framework (Original implementation)
python main.py --query "Quantum computing applications" --framework legacy

# Advanced Research with Maximum Features
python main.py --query "Future of renewable energy" --framework hybrid --max_results 8 --verbose --enable_memory --collaboration_mode
```

Output

- *Markdown Report*: `reports/report.md` - Professional research report
- *PDF Report*: `reports/report.pdf` - PDF version (if pandoc available)
- *Raw Data*: `data/run_YYYYMMDD_HHMMSS/` - Source materials and logs
- *Retrieval Data*: Individual JSON files for each research task

Requirements

- Python 3.8+
- OpenAI API key (GPT-4o recommended)
- Internet connection for web research
- Optional: Copyleaks API for enhanced plagiarism detection

Configuration

All API keys and settings are managed through a **single centralized configuration file**. The system supports both `.env` files and environment variables.

Quick Setup

1. **Create `.env` file** in the `agentic_research_assistant` directory:
```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional - Advanced Configuration
MODEL_NAME=gpt-4o
TEMPERATURE=0.3
DEFAULT_FRAMEWORK=hybrid
VERBOSE_LOGGING=false
```

2. **View Current Configuration**:
```bash
python main.py --show_config
```

Complete Configuration Options

Core API Keys
- `OPENAI_API_KEY` *(Required)* - Your OpenAI API key
- `COPYLEAKS_API_KEY` *(Optional)* - Enhanced plagiarism checking
- `COPYLEAKS_API_EMAIL` *(Optional)* - Copyleaks account email

Model Configuration
- `MODEL_NAME` *(Default: gpt-4o)* - AI model to use
- `TEMPERATURE` *(Default: 0.3)* - Model creativity (0.0-2.0)
- `MAX_TOKENS` *(Default: 4000)* - Maximum response length

Research Settings
- `DEFAULT_MAX_RESULTS` *(Default: 5)* - Sources per research task
- `TIMEOUT_SECONDS` *(Default: 20)* - Web request timeout
- `MAX_CONTENT_LENGTH` *(Default: 8000)* - Maximum content per source
- `ORIGINALITY_THRESHOLD` *(Default: 0.8)* - Plagiarism detection threshold

Framework Behavior
- `DEFAULT_FRAMEWORK` *(Default: hybrid)* - AI framework (crewai/autogen/hybrid/legacy)
- `VERBOSE_LOGGING` *(Default: false)* - Detailed logging
- `ENABLE_MEMORY` *(Default: false)* - LangChain memory retention

Output Directories
- `REPORTS_DIR` *(Default: reports)* - Report output location
- `DATA_DIR` *(Default: data)* - Research data storage
- `LOGS_DIR` *(Default: logs)* - System logs location

Security Features

- **Centralized Management**: All secrets in one place
- **Environment Variable Support**: Works with system env vars
- **Validation**: Automatic API key format validation  
- **Error Handling**: Clear messages for missing configuration
- **No Hardcoded Secrets**: All secrets externalized

Technical Implementation

Framework Integration
- **CrewAI**: Sequential task execution with role-based agents
- **LangChain**: Tool standardization and memory management  
- **AutoGen**: Multi-agent conversations and collaboration
- **GPT-4o**: Advanced language model for all AI operations

Core Technologies
- **Web Search**: DuckDuckGo Search API with intelligent content extraction
- **PDF Processing**: PyMuPDF for document parsing and text extraction
- **Plagiarism Detection**: TF-IDF vectorization with cosine similarity analysis
- **Report Generation**: Markdown formatting with optional PDF conversion via pypandoc/ReportLab
- **Memory Management**: LangChain ConversationBufferMemory for context retention
- **Agent Orchestration**: CrewAI Process.sequential for structured workflows

Advanced Features
- **Semantic Search**: Intelligent query decomposition and search strategy optimization
- **Content Synthesis**: Multi-source information integration with proper citations
- **Quality Control**: Automated originality checking and content improvement
- **Collaborative Intelligence**: Agent-to-agent discussions for enhanced decision making
- **Comprehensive Logging**: Full transparency of AI decision-making processes
- **Modular Architecture**: Easily extensible and maintainable codebase
