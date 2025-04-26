from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

search_tool = SerperDevTool()


researcher = Agent(
    role="Senior researcher",
    goal="Gather accurate information on Machine Learning Basics from realiable sources.",
    backstory="You are an expert reasearcher skilled at finding credible information online",
    tools=[search_tool],
    verbose=True,
)


summarizer = Agent(
    role="Study Guide Writer",
    goal="Summarize research into a concise study guide with key concepts.",
    backstory="You are a skilled educator who creates clear, concise study materials.",
    verbose=True,
)

quiz_creator = Agent(
    role="Quiz Developer",
    goal="Create engaging quiz questions based on the study guide.",
    backstory="You are an expert in designing educational quizzes to reinforce learning.",
    verbose=True,
)

research_task = Task(
    description="Search the web for reliable information on Machine Learning Basics and compile a detailed report.",
    expected_output="A detailed report with key information on Machine Learning Basics.",
    agent=researcher,
)

summarization_task = Task(
    description="Use the research report to create a concise study guide with a 200-word summary and 5 key concepts.",
    expected_output="A study guide with a 200-word summary and 5 key concepts listed.",
    agent=summarizer,
    context=[research_task],
)

quiz_task = Task(
    description="Based on the study guide, generate 5 multiple-choice quiz questions with 4 options each and indicate the correct answer.",
    expected_output="Five multiple-choice questions with answers.",
    agent=quiz_creator,
    context=[summarization_task],
)

crew = Crew(
    agents=[researcher, summarizer, quiz_creator],
    tasks=[research_task, summarization_task, quiz_task],
    process="sequential",
    verbose=True,
)

crew.kickoff()
