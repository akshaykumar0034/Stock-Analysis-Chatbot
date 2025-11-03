import json
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

# Import our components
from config import llm
from utils.stock_data import get_current_price, get_historical_data, get_stock_info, get_stock_news
from utils.technical_analysis import calculate_ta_indicators
from utils.sentiment import analyze_sentiment
from utils.plotting import plot_stock_trend

# --- 1. Define Agent State ---
class AgentState(TypedDict):
    query: str
    stock_symbol: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    plot: Optional[Any] = None
    response: str = ""
    
    # Flags to control the flow
    request_data: bool = False
    request_analysis: bool = False
    request_sentiment: bool = False
    request_plot: bool = False
    is_greeting: bool = False

# --- 2. Define Router Logic ---
class QueryRouter(BaseModel):
    """Decides the next steps and extracts the stock symbol."""
    stock_symbol: Optional[str] = Field(description="The stock ticker symbol, e.g., 'AAPL', 'TSLA', 'GOOG'.")
    request_data: bool = Field(description="True if the user wants fundamental data (P/E, Market Cap, EPS).")
    request_analysis: bool = Field(description="True if the user wants technical analysis (RSI, MACD, Bollinger).")
    request_sentiment: bool = Field(description="True if the user wants news or market sentiment.")
    request_plot: bool = Field(description="True if the user wants a chart, plot, or trend visualization.")
    is_greeting: bool = Field(description="True if the user is just saying hi, thanks, or goodbye.")

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert financial assistant. Your job is to parse the user's query and route it.
    - Extract the stock symbol (ticker).
    - Classify the request based on the tools.
    - If a query is general (e.g., "Analyze TSLA"), set request_data, request_analysis, and request_sentiment to True.
    - If a query asks for a plot (e.g., "Plot AAPL"), set *only* request_plot to True.
    - If a query asks for *only* technical analysis (e.g., "What's the RSI for MSFT?"), set *only* request_analysis to True.
    - If a query asks for *only* fundamentals (e.g., "What's the P/E for GOOG?"), set *only* request_data to True.
    - If no stock symbol is mentioned, ask the user for one."""),
    ("human", "{query}")
])

structured_llm = llm.with_structured_output(QueryRouter)
router_chain = router_prompt | structured_llm

# --- 3. Define Graph Nodes ---

def route_query_node(state: AgentState):
    """Parses the query and sets routing flags in the state."""
    query = state['query']
    router_output = router_chain.invoke({"query": query})
    
    return {
        "stock_symbol": router_output.stock_symbol,
        "request_data": router_output.request_data,
        "request_analysis": router_output.request_analysis,
        "request_sentiment": router_output.request_sentiment,
        "request_plot": router_output.request_plot,
        "is_greeting": router_output.is_greeting,
    }

def get_data_node(state: AgentState):
    """Fetches fundamental data."""
    # *** FIX: Check the flag before running ***
    if not state.get("request_data"):
        return {} # Do nothing
    
    symbol = state['stock_symbol']
    try:
        price = get_current_price(symbol)
        info = get_stock_info(symbol)
        data = {"current_price": price, **info}
        return {"data": data}
    except Exception as e:
        return {"response": f"Error fetching data for {symbol}: {e}. Is the ticker correct?"}

def get_analysis_node(state: AgentState):
    """Performs technical analysis."""
    # *** FIX: Check the flag before running ***
    if not state.get("request_analysis"):
        return {} # Do nothing

    symbol = state['stock_symbol']
    try:
        hist_data = get_historical_data(symbol)
        analysis = calculate_ta_indicators(hist_data)
        return {"analysis": analysis}
    except Exception as e:
        return {"response": f"Error performing analysis for {symbol}: {e}"}

def get_sentiment_node(state: AgentState):
    """Analyzes news sentiment."""
    # *** FIX: Check the flag before running ***
    if not state.get("request_sentiment"):
        return {} # Do nothing

    symbol = state['stock_symbol']
    try:
        news = get_stock_news(symbol)
        sentiment = analyze_sentiment(news)
        return {"sentiment": sentiment}
    except Exception as e:
        error_sentiment = {
            "overall_sentiment": "Error",
            "details": f"Failed to run sentiment analysis: {e}"
        }
        return {"sentiment": error_sentiment}


def create_plot_node(state: AgentState):
    """Creates a stock trend plot."""
    # *** FIX: Check the flag before running ***
    if not state.get("request_plot"):
        return {} # Do nothing

    symbol = state['stock_symbol']
    try:
        hist_data = get_historical_data(symbol, period="1y") # Use 1 year for a good plot
        plot_fig = plot_stock_trend(hist_data, symbol)
        return {"plot": plot_fig}
    except Exception as e:
        return {"response": f"Error creating plot for {symbol}: {e}"}

def generate_response_node(state: AgentState):
    """Generates the final LLM response based on all gathered data."""
    
    if state.get("is_greeting"):
        return {"response": "Hello! How can I help you with your stock analysis today?"}
    
    if not state.get("stock_symbol"):
        return {"response": "I'm sorry, I couldn't identify a stock symbol. Could you please specify which stock you're interested in?"}

    # *** FIX: Build a smarter context for the LLM ***
    context = {
        "query": state['query'],
        "symbol": state['stock_symbol'],
        # Only add data to context if it was actually gathered
        "data": state.get('data') if state.get('data') else "Not requested.",
        "analysis": state.get('analysis') if state.get('analysis') else "Not requested.",
        "sentiment": state.get('sentiment') if state.get('sentiment') else "Not requested.",
    }
    
    # Check if *any* data was gathered.
    plot_only = (
        not state.get('data') and 
        not state.get('analysis') and 
        not state.get('sentiment') and
        state.get('plot')
    )

    prompt_template = """You are a helpful AI stock analysis assistant.
Your task is to synthesize all the information provided and answer the user's query.
- Be clear, concise, and professional.
- Use Markdown to format your response (e.g., lists, bolding).
- Only present the data that was requested and gathered.
- If data is "Not requested", do not mention it.
- If the user *only* asked for a plot, just confirm the plot is ready.
- If you have technical analysis, explain what the numbers mean (e.g., "RSI is 65, which is nearing overbought territory").
- If there was an error in a previous step, inform the user politely.

Here is the data you gathered:
{context_json}

My original query was: {query}
"""

    # *** FIX: Handle plot-only requests separately ***
    if plot_only:
        # If the *only* request was a plot, give a simple response.
        final_response_content = f"Here is the requested plot for {state.get('stock_symbol')}."
    else:
        # Otherwise, use the LLM to synthesize the data
        prompt = ChatPromptTemplate.from_template(prompt_template)
        final_response_chain = prompt | llm
        context_json = json.dumps(context, default=str, indent=2)
        response = final_response_chain.invoke({"query": state['query'], "context_json": context_json})
        final_response_content = response.content

    # Add the plot confirmation line *only* if a plot was generated
    if state.get("plot"):
        final_response_content += "\n\nI have also generated the interactive plot you requested."
        
    return {"response": final_response_content}

# --- 4. Define Conditional Edges ---

def check_for_symbol(state: AgentState):
    """Checks if a stock symbol was provided or if it's a greeting."""
    if state.get("is_greeting"):
        return "generate_response"
    if state.get("stock_symbol"):
        # *** FIX: Go to a new routing node ***
        return "decide_first_tool" 
    else:
        return "generate_response" # Will generate "Please provide a symbol"

# *** FIX: New routing function ***
def decide_first_tool(state: AgentState):
    """Decides which tool node to run first."""
    if state.get("request_data"):
        return "get_data"
    if state.get("request_analysis"):
        return "get_analysis"
    if state.get("request_sentiment"):
        return "get_sentiment"
    if state.get("request_plot"):
        return "create_plot"
    # Fallback if no tool is flagged
    return "generate_response" 

def after_data(state: AgentState):
    """After fetching data, decide the next step."""
    if state.get("request_analysis"):
        return "get_analysis"
    if state.get("request_sentiment"):
        return "get_sentiment"
    if state.get("request_plot"):
        return "create_plot"
    return "generate_response"

def after_analysis(state: AgentState):
    """After analysis, decide the next step."""
    if state.get("request_sentiment"):
        return "get_sentiment"
    if state.get("request_plot"):
        return "create_plot"
    return "generate_response"

def after_sentiment(state: AgentState):
    """After sentiment, decide the next step."""
    if state.get("request_plot"):
        return "create_plot"
    return "generate_response"

# --- 5. Build the Graph ---
workflow = StateGraph(AgentState)

# Add all the nodes
workflow.add_node("route_query", route_query_node)
workflow.add_node("get_data", get_data_node)
workflow.add_node("get_analysis", get_analysis_node)
workflow.add_node("get_sentiment", get_sentiment_node)
workflow.add_node("create_plot", create_plot_node)
workflow.add_node("generate_response", generate_response_node)

# *** FIX: Add the new dummy node ***
workflow.add_node("decide_first_tool", lambda state: {}) 

# 1. Start at the router
workflow.set_entry_point("route_query")

# 2. Check if we have a symbol
workflow.add_conditional_edges(
    "route_query",
    check_for_symbol,
    {
        "generate_response": "generate_response",
        # *** FIX: Point to the new decider node ***
        "decide_first_tool": "decide_first_tool", 
    }
)

# 3. *** FIX: Add the new routing logic from the decider node ***
workflow.add_conditional_edges(
    "decide_first_tool",
    decide_first_tool,
    {
        "get_data": "get_data",
        "get_analysis": "get_analysis",
        "get_sentiment": "get_sentiment",
        "create_plot": "create_plot",
        "generate_response": "generate_response"
    }
)

# 4. Define the tool-calling pipeline (sequential)
workflow.add_conditional_edges(
    "get_data",
    after_data,
    {
        "get_analysis": "get_analysis",
        "get_sentiment": "get_sentiment",
        "create_plot": "create_plot",
        "generate_response": "generate_response"
    }
)
workflow.add_conditional_edges(
    "get_analysis",
    after_analysis,
    {
        "get_sentiment": "get_sentiment",
        "create_plot": "create_plot",
        "generate_response": "generate_response"
    }
)
workflow.add_conditional_edges(
    "get_sentiment",
    after_sentiment,
    {
        "create_plot": "create_plot",
        "generate_response": "generate_response"
    }
)

# 5. After the plot node, always generate a response
workflow.add_edge("create_plot", "generate_response")

# 6. The final node is the end
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()