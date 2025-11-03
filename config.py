import os
import finnhub  
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# --- Gemini Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                           google_api_key=GOOGLE_API_KEY,
                           temperature=0.0)

# --- Finnhub Config ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in .env file. Please add it.")

# Initialize the Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)