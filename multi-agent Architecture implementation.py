import os
from typing import List, Dict, Any
from datetime import datetime
import requests
import json
import argparse
import markdown
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import FileCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class IndustryResearchAgent:
    """Agent responsible for researching industry and company information."""
    
    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Web Search",
                description="Search the web for information about companies and industries",
                func=self.search_tool.run
            )
        ]
        
        research_prompt = """You are an Industry Research Agent specialized in gathering comprehensive information about companies and industries.

Task: Research the specified company or industry thoroughly.

Context Given: {context}

Detailed Instructions:
1. Use the tools available to research the company/industry thoroughly
2. Focus on identifying:
   - Industry classification and segment details
   - Key products/services and business model
   - Strategic focus areas and priorities
   - Current technological infrastructure and digital maturity
   - Major challenges and pain points in operations
   - Competitive landscape and market position
   - Recent initiatives or transformations

For each finding, cite the source of information if possible.

Format the output as a structured report with clear sections and bullet points. Include a brief executive summary at the beginning.

{format_instructions}

{tools}

Please begin your research on: {query}
"""
        
        self.prompt = PromptTemplate(
            template=research_prompt,
            input_variables=["context", "query", "tools", "format_instructions"]
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def research(self, company_or_industry: str, context: str = "") -> Dict[str, Any]:
        """Conduct research on the specified company or industry."""
        result = self.agent_executor.invoke({
            "query": company_or_industry,
            "context": context,
            "format_instructions": "Provide a detailed analysis with sections on industry overview, business model, tech infrastructure, and strategic priorities.",
            "tools": self.tools
        })
        
        return {
            "research_results": result["output"],
            "company_or_industry": company_or_industry
        }


class UseCaseGenerationAgent:
    """Agent responsible for generating AI/ML/GenAI use cases based on industry research."""
    
    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Industry AI Trends Search",
                description="Search for AI and ML trends in specific industries",
                func=self.search_tool.run
            )
        ]
        
        usecase_prompt = """You are a Use Case Generation Agent specialized in identifying valuable AI and GenAI applications for businesses.

Task: Generate relevant, high-impact AI/ML/GenAI use cases for the company/industry based on the research provided.

Industry Research: {research}

Additional Context: {context}

Detailed Instructions:
1. Analyze the industry research to identify key pain points and opportunities
2. Research current AI/ML adoption trends in this specific industry
3. Generate concrete use cases in these categories:
   - Operations optimization and efficiency
   - Customer experience enhancement
   - Decision support and business intelligence
   - Predictive maintenance and analytics
   - Process automation and workflow optimization
   - Document intelligence and knowledge management
   - Other industry-specific applications

For each use case:
- Provide a clear title and concise description
- Explain the specific business problem it solves
- Outline expected benefits and potential ROI areas
- Rate implementation complexity (Low/Medium/High)
- Note any prerequisites or challenges
- Reference similar implementations in the industry where possible

Focus on practical, feasible solutions rather than speculative applications.

{format_instructions}

{tools}

Please generate AI/ML/GenAI use cases for: {company_or_industry}
"""
        
        self.prompt = PromptTemplate(
            template=usecase_prompt,
            input_variables=["research", "context", "company_or_industry", "tools", "format_instructions"]
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def generate_use_cases(self, research_results: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Generate AI/ML/GenAI use cases based on research results."""
        result = self.agent_executor.invoke({
            "research": research_results["research_results"],
            "context": context,
            "company_or_industry": research_results["company_or_industry"],
            "format_instructions": "Present use cases in a structured format with clear categorization and prioritization.",
            "tools": self.tools
        })
        
        return {
            "use_cases": result["output"],
            "company_or_industry": research_results["company_or_industry"]
        }


class ResourceCollectionAgent:
    """Agent responsible for collecting resources and datasets for implementing AI use cases."""
    
    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Dataset and Resource Search",
                description="Search for datasets, tutorials, and implementation resources for AI use cases",
                func=self.search_tool.run
            )
        ]
        
        resource_prompt = """You are a Resource Collection Agent specialized in finding relevant datasets and implementation resources for AI/ML/GenAI projects.

Task: Collect and organize datasets, code repositories, tutorials, and other resources for implementing the proposed AI use cases.

Use Cases: {use_cases}

Additional Context: {context}

Detailed Instructions:
1. For each proposed use case, search for:
   - Relevant datasets from Kaggle, HuggingFace, GitHub, etc.
   - Pre-trained models or APIs that could be leveraged
   - Implementation tutorials or guides
   - Academic papers or case studies on similar applications
   - Open-source tools that could accelerate development

2. For each resource:
   - Provide the full, clickable URL
   - Include a brief description of the resource
   - Explain how it relates to the specific use case
   - Note any limitations or considerations

3. Additionally, suggest GenAI-specific solutions like:
   - Document search and retrieval systems
   - Automated report generation
   - AI-powered chat systems for internal or customer-facing use
   - Knowledge extraction from unstructured data

Ensure all links are valid and directly accessible.

{format_instructions}

{tools}

Please collect resources for implementing AI/ML/GenAI use cases for: {company_or_industry}
"""
        
        self.prompt = PromptTemplate(
            template=resource_prompt,
            input_variables=["use_cases", "context", "company_or_industry", "tools", "format_instructions"]
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True
        )
    
    def collect_resources(self, use_case_results: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Collect resources for implementing the proposed AI use cases."""
        result = self.agent_executor.invoke({
            "use_cases": use_case_results["use_cases"],
            "context": context,
            "company_or_industry": use_case_results["company_or_industry"],
            "format_instructions": "Organize resources by use case category with clear links and descriptions.",
            "tools": self.tools
        })
        
        return {
            "resources": result["output"],
            "company_or_industry": use_case_results["company_or_industry"],
            "use_cases": use_case_results["use_cases"]
        }


class FinalProposalGenerator:
    """Component responsible for generating the final proposal with prioritized use cases and resources."""
    
    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    
    def generate_proposal(self, research_results: Dict[str, Any], 
                         use_case_results: Dict[str, Any], 
                         resource_results: Dict[str, Any]) -> str:
        """Generate the final proposal with prioritized use cases and implementation resources."""
        
        proposal_prompt = f"""Generate a comprehensive final proposal for AI/GenAI implementation opportunities for {research_results['company_or_industry']}.

The proposal should include:

1. Executive Summary:
   - Brief overview of {research_results['company_or_industry']}
   - Key opportunities identified for AI/GenAI implementation
   - Expected benefits and strategic alignment

2. Industry and Company Analysis:
{research_results['research_results']}

3. Prioritized AI/GenAI Use Cases:
{use_case_results['use_cases']}

4. Implementation Resources:
{resource_results['resources']}

5. Implementation Roadmap:
   - Recommended sequence of use case implementation
   - Key dependencies and prerequisites
   - Estimated timeline and resource requirements

6. Expected Outcomes:
   - Business impact metrics
   - ROI considerations
   - Competitive advantages

Format this as a professional proposal with clear section headers, concise bullet points, and visual separation between sections. Ensure all resource links are properly formatted as clickable links.
"""
        
        response = self.llm.invoke(proposal_prompt)
        return response.content


class AIUseCaseGenerationSystem:
    """Main system coordinating the multi-agent workflow for AI use case generation."""
    
    def __init__(self):
        self.industry_research_agent = IndustryResearchAgent()
        self.use_case_generation_agent = UseCaseGenerationAgent()
        self.resource_collection_agent = ResourceCollectionAgent()
        self.final_proposal_generator = FinalProposalGenerator()
    
    def run(self, company_or_industry: str, context: str = "") -> Dict[str, Any]:
        """Run the complete workflow to generate AI use cases and implementation resources."""
        
        print(f"\n{'='*80}\nStarting research for: {company_or_industry}\n{'='*80}\n")
        research_results = self.industry_research_agent.research(company_or_industry, context)
        
        print(f"\n{'='*80}\nGenerating use cases based on research\n{'='*80}\n")
        use_case_results = self.use_case_generation_agent.generate_use_cases(research_results, context)
        
        print(f"\n{'='*80}\nCollecting implementation resources\n{'='*80}\n")
        resource_results = self.resource_collection_agent.collect_resources(use_case_results, context)
        
        print(f"\n{'='*80}\nGenerating final proposal\n{'='*80}\n")
        final_proposal = self.final_proposal_generator.generate_proposal(
            research_results,
            use_case_results,
            resource_results
        )
        
        # Save results to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_slug = company_or_industry.lower().replace(" ", "_")
        
        # Save markdown version
        with open(f"{company_slug}_{timestamp}_proposal.md", "w") as f:
            f.write(final_proposal)
        
        # Save HTML version
        html_content = markdown.markdown(final_proposal)
        with open(f"{company_slug}_{timestamp}_proposal.html", "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Use Case Proposal for {company_or_industry}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                    h3 {{ color: #2980b9; }}
                    a {{ color: #3498db; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    .resource {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
                    .use-case {{ border-left: 4px solid #3498db; padding-left: 15px; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <h1>AI/GenAI Implementation Proposal for {company_or_industry}</h1>
                {html_content}
                <footer>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </footer>
            </body>
            </html>
            """)
        
        return {
            "research_results": research_results,
            "use_case_results": use_case_results,
            "resource_results": resource_results,
            "final_proposal": final_proposal,
            "files": {
                "markdown": f"{company_slug}_{timestamp}_proposal.md",
                "html": f"{company_slug}_{timestamp}_proposal.html"
            }
        }

# Command line interface
def main():
    parser = argparse.ArgumentParser(description='Generate AI/ML/GenAI use cases for a company or industry')
    parser.add_argument('company_or_industry', type=str, help='Name of the company or industry to analyze')
    parser.add_argument('--context', type=str, default='', help='Additional context or requirements')
    
    args = parser.parse_args()
    
    system = AIUseCaseGenerationSystem()
    results = system.run(args.company_or_industry, args.context)
    
    print(f"\n{'='*80}")
    print(f"Process complete! Files saved:")
    print(f"- Markdown: {results['files']['markdown']}")
    print(f"- HTML: {results['files']['html']}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
