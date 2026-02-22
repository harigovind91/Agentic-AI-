import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process

# Load environment variables
load_dotenv()

app = FastAPI(title="HAI Compliance Screening System")

# Data Model
class Order(BaseModel):
    order_id: str
    customer_name: str
    country: str
    amount: float
    product_type: str

# --- HAI Agents Configuration ---

# 1. Compliance Analyst Agent
compliance_analyst = Agent(
    role='Senior Compliance Analyst',
    goal='Identify legal and regulatory violations in real-time.',
    backstory='Expert in international trade laws and sanction lists. You ensure 100% legal integrity.',
    verbose=True,
    allow_delegation=False,
    llm=os.getenv("MODEL_NAME")
)

# 2. Risk & Ethics Auditor (Sanskar Module)
ethics_auditor = Agent(
    role='Risk & Ethics Auditor',
    goal='Assess the ethical impact and generate audit-ready reports.',
    backstory='You prioritize transparency and selfless business practices. You flag high-risk transactions.',
    verbose=True,
    llm=os.getenv("MODEL_NAME")
)

@app.post("/screen-order")
async def screen_order(order: Order):
    # Task 1: Legal Screening
    screening_task = Task(
        description=f"Analyze order {order.order_id} for {order.customer_name} from {order.country}. Verify against compliance rules.",
        agent=compliance_analyst,
        expected_output="Detailed compliance status (CLEARED/FLAGGED) with reasons."
    )

    # Task 2: Audit Report Generation
    audit_task = Task(
        description=f"Create a final audit report for order {order.order_id}. Evaluate the 'Sattvic' business integrity and risk level.",
        agent=ethics_auditor,
        expected_output="A structured report ready for Wipro internal review."
    )

    crew = Crew(
        agents=[compliance_analyst, ethics_auditor],
        tasks=[screening_task, audit_task],
        process=Process.sequential
    )

    try:
        result = crew.kickoff()
        return {"status": "Analysis Complete", "report": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
