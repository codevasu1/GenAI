"""Create and run a human-in-the-loop email drafting workflow with LangGraph."""

# Load environment variables before initializing any integrations that may use them.
from dotenv import load_dotenv
load_dotenv()


# Import LangGraph primitives, the chat model, and types used by the workflow.
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal


# Define the data carried between nodes during each workflow run.
class EmailState(BaseModel):
    query: str = ""
    draft: str = ""
    human_feedback: str = ""
    final_response: str = ""


# Initialize the model used to generate and revise email drafts.
llm = ChatGroq(model="openai/gpt-oss-20b")



# Workflow nodes: generate a draft, request human review, and finalize the result.

def draftEmail(state: EmailState) -> EmailState:
    """Generate a new email draft or revise it using the latest human feedback."""
    draft = llm.invoke(f"""
    Draft email based on this user query: {state.query},
    Previous draft: \n{state.draft or "create a new draft"}\n
    Note: if you recive human feedback, incorporate it into the draft,
    human feedback: {state.human_feedback or "none"}
""").content
    
    
    state.draft = draft
    return state 


def humanFeedback(state: EmailState) -> EmailState:
    """Pause the graph and capture approval or revision instructions."""
    # The graph resumes here when the caller supplies a Command(resume=...).
    feedback = interrupt({
        "draft_email": state.draft,
        "question": "Do you approve this draft? or suggest changes."
    })
    feedback = (feedback or "").strip().lower()

    # Approval clears feedback; any other response is treated as a revision request.
    if feedback in ("approved", "yes", "okay"):
        state.human_feedback = ""
        print(f"[humanFeedback]: {state.human_feedback}")
    else:
        state.human_feedback = feedback
        print(f"[humanFeedback]: {state.human_feedback}")

    return state



def finalizedEmail(state: EmailState) -> EmailState:
    """Record whether the human approved the draft and report the outcome."""
    if state.human_feedback:
        state.final_response = state.human_feedback
        print(f"[finalizedEmail]: Mail not finalized due to disapproval.")
    else:
        state.final_response = "Email sent successfully ..."
        print(f"[finalizedEmail]: Mail finalized. Email sent successfully ...")

    return state


# Route revision requests back for another draft; route approvals to finalization.
def conditional_routing(state: EmailState) -> Literal["draftEmail", "finalizedEmail"]:
    if state.human_feedback:
        return "draftEmail"
    return "finalizedEmail"



# Assemble the workflow graph by registering its nodes and connecting its transitions.
graph = StateGraph(EmailState)

# Register each processing step as a graph node.
graph.add_node("draftEmail", draftEmail)
graph.add_node("humanFeedback", humanFeedback)
graph.add_node("finalizedEmail", finalizedEmail)

# Start with drafting, collect feedback, then route or finish based on approval.
graph.add_edge(START, "draftEmail")
graph.add_edge("draftEmail", "humanFeedback")
graph.add_conditional_edges("humanFeedback", conditional_routing)
graph.add_edge("finalizedEmail", END)

# Enable checkpointing so the graph can pause for feedback and resume the same run.
graph = graph.compile(checkpointer=InMemorySaver())

# Keep a stable thread identifier so interrupted workflow state can be resumed.
config = {"configurable": {"thread_id":"my_random_thread_id_R435y6tg5y6te34rgtevgsdfhrgherht"}}


# Start an email request and display the first draft returned by the graph.

query = input("user: ")
res = graph.invoke({"query": query}, config=config) # Run once to start the graph
print("Assistant: \n", res['draft'], "\n")

# Resume after each human response; approvals finish the loop, while revision
# requests resume the paused node and cycle through drafting again.
while True:
    feedback = input("user: ")

    if feedback.lower() in ["yes", "approved", "okay"]:
        res = graph.invoke(Command(resume="yes"),config=config)
        print("Assistant: final draft\n", res['draft'], "\n")

        break
    else:
        res = graph.invoke(Command(resume=feedback),config=config)

    print("Assistant: new draft\n", res['draft'], "\n")


""" 
what this script does is it allows a user to iteratively draft an email with human-in-the-loop feedback.

# first draft

res = graph.invoke({"query":"I want to send email for medical leave request "
"from date 11 oct to 13 oct 2026. create simple email."},config=config)
print("Assistant: \n", res['draft'], "\n")

# human intervention
    # now your loop gets interrupted by human feedback
    # if you print res object, you'll see something like this:'__interrupt__': [Interrupt(value={'draft_email': '**Subject:** ...
    # if you interrupt the loop and then resume with "yes", the process will continue from where it left off.
    # if you interrupt the loop and then resume with "no", the process will not continue. And res will go back to draftEmail node.
# new draft
res = graph.invoke(Command(resume="I want to change dates make it 14-16, in the end just mention only my name"),config=config)
print("Assistant: \n", res['draft'], "\n")


# final human approval    
    # until last human feedback is approved, email drafting will continue
res = graph.invoke(Command(resume="yes"),config=config)
print("Assistant: final draft\n", res['draft'], "\n")

"""
# Example interactive session demonstrating a revision followed by approval.
"""
user: create email regarding medical leave from date 14 nov to 20 nov, 2026
Assistant: 
 **Subject:** Medical Leave Request - 14 Nov 2026 to 20 Nov 2026  

Dear [Manager's Name],

I hope you are doing well. I am writing to formally request medical leave from **14 November 2026** through **20 November 2026**.  

During this period, I will be unable to attend to work duties due to a scheduled medical treatment and recovery. I have arranged for [Colleague's Name]to cover urgent tasks and will ensure all pending items are documented before my leave begins.  

Please let me know if you require any additional documentation or if there are specific procedures I should follow. I will keep you updated on my progress and will be reachable by email for any critical matters.

Thank you for your understanding and support.

Best regards,

[Your Full Name]  
[Your Position]  
[Your Department]  
[Phone Number]  
[Email Address] 

user: change dates from 12 nov to 18 nov, and mention only my name at the end
[humanFeedback]: change dates from 12 nov to 18 nov, and mention only my name at the end
Assistant: new draft
 **Subject:** Medical Leave Request - 12 Nov 2026 to 18 Nov 2026  

Dear [Manager's Name],

I hope you are doing well. I am writing to formally request medical leave from **12 November 2026** through **18 November 2026**.

During this period, I will be unable to attend to work duties due to a scheduled medical treatment and recovery. I have arranged for [Colleague's Name]to cover urgent tasks and will ensure all pending items are documented before my leave begins.

Please let me know if you require any additional documentation or if there are specific procedures I should follow. I will keep you updated on my progress and will be reachable by email for any critical matters.

Thank you for your understanding and support.

Best regards,

[Your Full Name] 

user: yes
[humanFeedback]: 
[finalizedEmail]: Mail finalized. Email sent successfully ...
Assistant: final draft
 **Subject:** Medical Leave Request - 12 Nov 2026 to 18 Nov 2026  

Dear [Manager's Name],

I hope you are doing well. I am writing to formally request medical leave from **12 November 2026** through **18 November 2026**.

During this period, I will be unable to attend to work duties due to a scheduled medical treatment and recovery. I have arranged for [Colleague's Name]to cover urgent tasks and will ensure all pending items are documented before my leave begins.

Please let me know if you require any additional documentation or if there are specific procedures I should follow. I will keep you updated on my progress and will be reachable by email for any critical matters.

Thank you for your understanding and support.

Best regards,

[Your Full Name]

"""