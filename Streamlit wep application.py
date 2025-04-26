import streamlit as st
import os
import tempfile
from datetime import datetime
import time
import base64
from typing import List, Dict, Any

# Import the multi-agent system
from multi_agent_architecture import (
    IndustryResearchAgent,
    UseCaseGenerationAgent,
    ResourceCollectionAgent,
    FinalProposalGenerator
)

# Configure page
st.set_page_config(
    page_title="AI Use Case Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for download buttons
def get_download_link(content, filename, text):
    """Generate a download link for a file."""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def run_agents_with_progress(company_or_industry, context):
    """Run the agent system with progress indicators."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize agents
    status_text.text("Initializing agents...")
    industry_research_agent = IndustryResearchAgent()
    use_case_generation_agent = UseCaseGenerationAgent()
    resource_collection_agent = ResourceCollectionAgent()
    final_proposal_generator = FinalProposalGenerator()
    progress_bar.progress(10)
    
    # Step 1: Industry Research
    status_text.text("Researching industry and company information...")
    research_results = industry_research_agent.research(company_or_industry, context)
    progress_bar.progress(40)
    
    # Step 2: Use Case Generation
    status_text.text("Generating AI/ML/GenAI use cases...")
    use_case_results = use_case_generation_agent.generate_use_cases(research_results, context)
    progress_bar.progress(70)
    
    # Step 3: Resource Collection
    status_text.text("Collecting implementation resources...")
    resource_results = resource_collection_agent.collect_resources(use_case_results, context)
    progress_bar.progress(90)
    
    # Step 4: Final Proposal 
    status_text.text("Creating final proposal...")
    final_proposal = final_proposal_generator.generate_proposal(
        research_results,
        use_case_results,
        resource_results
    )
    progress_bar.progress(100)
    status_text.text("Process complete!")
    
    return {
        "research_results": research_results,
        "use_case_results": use_case_results, 
        "resource_results": resource_results,
        "final_proposal": final_proposal
    }

# App title and description
st.title("🤖 AI Use Case Generation System")
st.markdown("""
This system helps you identify and explore AI/ML/GenAI use cases for your company or industry.
Simply enter your company or industry name, and our multi-agent system will:
- Research your industry and business model
- Generate relevant AI use cases tailored to your needs
- Collect implementation resources and datasets
- Create a comprehensive proposal with actionable insights
""")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

company_or_industry = st.sidebar.text_input(
    "Company or Industry Name",
    help="Enter the name of your company or industry (e.g., 'Tesla' or 'Healthcare')"
)

context = st.sidebar.text_area(
    "Additional Context (Optional)",
    help="Provide any specific goals, challenges, or requirements for the AI implementation"
)

run_button = st.sidebar.button("Generate AI Use Cases")

# Add architecture diagram
st.sidebar.header("System Architecture")
st.sidebar.markdown("""
```
User Input → Industry Research Agent → Use Case Generation Agent → Resource Collection Agent → Final Proposal
```
""")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Industry Research", "AI Use Cases", "Implementation Resources", "Final Proposal"])

# Run the system when button is clicked
if run_button and company_or_industry:
    with st.spinner(f"Analyzing {company_or_industry}... This may take several minutes."):
        try:
            # Store in session state
            results = run_agents_with_progress(company_or_industry, context)
            st.session_state.results = results
            
            # Generate file names with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_slug = company_or_industry.lower().replace(" ", "_")
            markdown_filename = f"{company_slug}_{timestamp}_proposal.md"
            
            st.session_state.markdown_filename = markdown_filename
            st.session_state.has_results = True
            
            st.success("Analysis complete! See the tabs for detailed results.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.has_results = False
elif run_button:
    st.warning("Please enter a company or industry name.")

# Display results if available
if 'has_results' in st.session_state and st.session_state.has_results:
    results = st.session_state.results
    
    # Tab 1: Industry Research
    with tab1:
        st.header(f"Industry Research: {results['research_results']['company_or_industry']}")
        st.markdown(results['research_results']['research_results'])
    
    # Tab 2: AI Use Cases
    with tab2:
        st.header("Recommended AI/GenAI Use Cases")
        st.markdown(results['use_case_results']['use_cases'])
    
    # Tab 3: Implementation Resources
    with tab3:
        st.header("Implementation Resources")
        st.markdown(results['resource_results']['resources'])
    
    # Tab 4: Final Proposal
    with tab4:
        st.header("Final AI Implementation Proposal")
        st.markdown(results['final_proposal'])
        
        # Download buttons
        st.download_button(
            label="Download Proposal (Markdown)",
            data=results['final_proposal'],
            file_name=st.session_state.markdown_filename,
            mime="text/markdown"
        )
else:
    # Default content for each tab
    with tab1:
        st.info("Enter a company or industry name and click 'Generate AI Use Cases' to begin.")
    with tab2:
        st.info("AI use cases will appear here after analysis.")
    with tab3:
        st.info("Implementation resources will appear here after analysis.")
    with tab4:
        st.info("The final proposal will appear here after analysis.")

# Footer
st.markdown("---")
st.markdown("Developed by InstaResz Business Services Pvt. Ltd | AI Use Case Generation System")
