import os
from crewai import Agent, Crew, Process, Task
from langchain_openai import AzureChatOpenAI
import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import json
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]="NA"

# Initialize LLM
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-4o",
)
def load_pdfs():
    ret = PyPDFDirectoryLoader("/Users/umang/Desktop/Resumes:CV")
    return ret.load()

#TAKING INPUTS
job_description=input("Enter the job description:\n")
job_role=input("Enter the job Role:\n")
key_skills=input("Input Key Skills:\n")
#Agents
resume_analyzer=Agent(
            role='Resume Processor',
            goal='Analyze the resumes for relevant information',
            verbose=True,
            memory=True,
            backstory="You are an expert in analyzing resumes, identifying key skills and experiences.",
        
            allow_delegation=True,
            llm=llm
)

senior_engineer=Agent(
    role='Senior Engineer',
            goal='Look for relevant skills required for the job',
            verbose=True,
            memory=True,
            backstory="You are a diligent senior engineer with fifteen years of experience. You demand a solid profile with the necessary skills.",
            
            allow_delegation=False,
            llm=llm
)

human_resource=Agent(
    role='Human Resource Manager',
            goal='Look at the resumes, analyze them, and check the company-culture fit',
            verbose=True,
            memory=True,
            backstory="You are an HR manager tasked to evaluate candidates based on their ethos and check if they would thrive in the company's culture.",
           
            allow_delegation=False,
            llm=llm
)
#Tasks
resume_analyzer_task=Task(
    description=dedent(f"""\
                Analyze the given resumes, check the {job_role} and {job_description} and look for skills required for the job.
                Resumes: {load_pdfs()}
            """),
            expected_output=dedent("""\
                A comprehensive report detailing the candidate's profile, skills, and years of experience, along with specific points relevant to highlight the key aspects of the candidate's skillset. Based on this, rank the resumes.
            """),
            agent=resume_analyzer
)
senior_engineer_task=Task(
    description=dedent(f"""\
                Look for the relevant skills needed for this {job_description}, analyze if the necessary programming languages are present, and rank them once more based on your own analysis.
            """),
            expected_output=dedent("""\
                A final ranking of the candidates suitable for the job.
            """),
            agent=senior_engineer
)
email_extraction_task=Task(
    description=dedent("""\
                Your job is to extract the emails from the top two ranked resumes.
            """),
            expected_output=dedent("""\
                The emails of the selected candidates from the given resumes only if you feel they are qualified enough else just reject them, if none qualify no issues, return no one.
            """),
            agent=human_resource
)
#CREW
crew = Crew(
    agents=[resume_analyzer, senior_engineer, human_resource],
    tasks=[
        resume_analyzer_task,
        senior_engineer_task,
        email_extraction_task
    ],
    process=Process.sequential,
    full_output=True,
    share_crew=False,
)
#KICKOFF
result = crew.kickoff()
print("Resume analysis complete")
print(result)
