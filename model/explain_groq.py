import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=300
)

def explain_prediction(email_text, label):
    """Use Groq LLaMA to explain why an email is classified as Spam or Ham."""
    prompt = f"""
You are an AI assistant explaining spam detection results.
The email text is below:

"{email_text}"

The model classified it as **{label}**.
Briefly explain why an email like this could be considered {label}, using logical reasoning and examples.
"""
    response = client.invoke([HumanMessage(content=prompt)])
    return response.content
