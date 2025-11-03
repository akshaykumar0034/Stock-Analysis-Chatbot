import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Initialize the Gemini-2.5-Flash model
# We use a low temperature (0.0) for routing and analysis to be more deterministic
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                           google_api_key=GOOGLE_API_KEY,
                           temperature=0.0)