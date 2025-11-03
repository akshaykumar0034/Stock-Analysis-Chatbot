import json
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

# Import our components
from config import llm
# *** FIX: We now need get_historical_data for our new tool ***
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
    
    # *** FIX: Add new state for our new tool ***
    historical_calc: Optional[Dict[str, Any]] = None

    response: str = ""
    
    # Flags to control the flow
    request_data: bool = False
    request_analysis: bool = False
    request_sentiment: bool = False
    request_plot: bool = False
    
    # *** FIX: Add new flag for our new tool ***
    request_historical_calc: bool = False 
    is_greeting: bool = False

# --- 2. Define Router Logic ---
class QueryRouter(BaseModel):
    """Decides the next steps and extracts the stock symbol."""
    stock_symbol: Optional[str] = Field(description="The stock ticker symbol, e.g., 'AAPL', 'TATAMOTORS.NS', 'GOOG'.")
    request_data: bool = Field(description="True if the user wants fundamental data (P/E, Market Cap, EPS).")
    request_analysis: bool = Field(description="True if the user wants technical analysis (RSI, MACD, Bollinger).")
    request_sentiment: bool = Field(description="True if the user wants news or market sentiment.")
    request_plot: bool = Field(description="True if the user wants a chart, plot, or trend visualization.")
    
    # *** FIX: Add new field for our new tool ***
    request_historical_calc: bool = Field(description="True if the user asks for profit/loss over a time period, e.g., 'last 7 days' or 'performance this week'.")
    
    is_greeting: bool = Field(description="True if the user is just saying hi, thanks, or goodbye.")

# *** FIX: This is the new, smarter prompt ***
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert financial assistant. Your job is to parse the user's query and route it.
    You MUST follow these rules for tickers:
    1.  Extract the company name (e.g., "Tata Motors", "BPCL", "Apple").
    2.  Determine the stock ticker.
    3.  **For Indian stocks, you MUST append the '.NS' suffix.** (e.g., Tata Motors -> 'TATAMOTORS.NS', BPCL -> 'BPCL.NS', Reliance -> 'RELIANCE.NS').
    4.  For US stocks, use the plain ticker (e.g., Apple -> 'AAPL', Tesla -> 'TSLA').
    5.  The user might also provide a direct ticker like 'TTM'. 'TTM' is the US-listed ADR for Tata Motors. If the user asks for 'Tata Motors', prefer the Indian '.NS' ticker ('TATAMOTORS.NS'). If they ask for 'TTM' specifically, use 'TTM'.

    You MUST follow these rules for tools:
    - If a query is general (e.g., "Analyze TSLA"), set request_data, request_analysis, and request_sentiment to True.
    - If a query asks for a plot (e.g., "Plot AAPL"), set *only* request_plot to True.
    - If a query asks for *only* technical analysis (e.g., "What's the RSI for MSFT?"), set *only* request_analysis to True.
    - If a query asks for *only* fundamentals (e.g., "What's the P/E for GOOG?"), set *only* request_data to True.
    
    - **If a user asks if a market is 'bull or bear' (e.g., "Reliance market is today bull or bear?"), you MUST set request_analysis=True AND request_sentiment=True.**
    - **If a user asks for 'profit or loss' or 'performance' over a period (e.g., "last 7 days market of reliance"), you MUST set request_historical_calc=True.**
    
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
        "request_historical_calc": router_output.request_historical_calc, # *** FIX: Pass new flag
        "is_greeting": router_output.is_greeting,
    }

def get_data_node(state: AgentState):
    """Fetches fundamental data."""
    if not state.get("request_data"):
        return {} 
    
    symbol = state['stock_symbol']
    try:
        price = get_current_price(symbol)
        info = get_stock_info(symbol)
        data = {"current_price": price, **info}
        return {"data": data}
    except Exception as e:
        return {"data": f"Error fetching data for {symbol}: {e}. Is the ticker correct?"}

def get_analysis_node(state: AgentState):
    """Performs technical analysis."""
    if not state.get("request_analysis"):
        return {}

    symbol = state['stock_symbol']
    try:
        # --- FIX: Changed from default (6mo) to "1y" ---
        hist_data = get_historical_data(symbol, period="1y") 
        
        analysis = calculate_ta_indicators(hist_data)
        return {"analysis": analysis}
    except Exception as e:
        return {"analysis": f"Error performing analysis for {symbol}: {e}"}
def get_sentiment_node(state: AgentState):
    """Analyzes news sentiment."""
    if not state.get("request_sentiment"):
        return {}

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
    if not state.get("request_plot"):
        return {}

    symbol = state['stock_symbol']
    try:
        hist_data = get_historical_data(symbol, period="1y") 
        plot_fig = plot_stock_trend(hist_data, symbol)
        return {"plot": plot_fig}
    except Exception as e:
        return {"plot": f"Error creating plot for {symbol}: {e}"}

# *** FIX: This is the new node (our new "tool") ***
def calculate_profit_loss_node(state: AgentState):
    """Calculates profit/loss over a 7-day period."""
    if not state.get("request_historical_calc"):
        return {}
        
    symbol = state['stock_symbol']
    try:
        # Fetch 8 days of data to get 7 full trading periods
        hist_data = get_historical_data(symbol, period="8d")
        if hist_data.empty or len(hist_data) < 2:
            return {"historical_calc": "Not enough historical data to calculate."}
        
        # Get the first and last closing prices
        start_price = hist_data['Close'].iloc[0]
        end_price = hist_data['Close'].iloc[-1]
        
        # Calculate percentage change
        percent_change = ((end_price - start_price) / start_price) * 100
        
        result = {
            "period": "7-day",
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "profit_loss_percent": round(percent_change, 2)
        }
        return {"historical_calc": result}
        
    except Exception as e:
        return {"historical_calc": f"Error calculating profit/loss for {symbol}: {e}"}


def generate_response_node(state: AgentState):
    """Generates the final LLM response based on all gathered data."""
    
    if state.get("is_greeting"):
        return {"response": "Hello! How can I help you with your stock analysis today?"}
    
    if not state.get("stock_symbol"):
        return {"response": "I'm sorry, I couldn't identify a stock symbol. Could you please specify which stock you're interested in?"}

    context = {
        "query": state['query'],
        "symbol": state['stock_symbol'],
        "data": state.get('data') if state.get('data') else "Not requested.",
        "analysis": state.get('analysis') if state.get('analysis') else "Not requested.",
        "sentiment": state.get('sentiment') if state.get('sentiment') else "Not requested.",
        "historical_calc": state.get('historical_calc') if state.get('historical_calc') else "Not requested.", # *** FIX: Add new data to context ***
    }
    
    plot_only = (
        not state.get('data') and 
        not state.get('analysis') and 
        not state.get('sentiment') and
        not state.get('historical_calc') and
        state.get('plot')
    )

    prompt_template = """You are a helpful AI stock analysis assistant.
Your task is to synthesize all the information provided and answer the user's query.
- Be clear, concise, and professional.
- Use Markdown to format your response (e.g., lists, bolding).
- Only present the data that was requested and gathered.
- If data is "Not requested", do not mention it.
- **If you have 'analysis' (RSI/MACD) and 'sentiment', combine them to answer "bull or bear" questions.**
- **If you have 'historical_calc', use it to answer the "profit or loss" question.**
- If the user *only* asked for a plot, just confirm the plot is ready.
- If you have technical analysis, explain what the numbers mean (e.g., "RSI is 65, which is nearing overbought territory").
- If there was an error in a previous step (e.g., in 'data' or 'analysis'), inform the user politely that that specific piece of data couldn't be fetched.

Here is the data you gathered:
{context_json}

My original query was: {query}
"""

    if plot_only:
        final_response_content = f"Here is the requested plot for {state.get('stock_symbol')}."
    else:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        final_response_chain = prompt | llm
        context_json = json.dumps(context, default=str, indent=2)
        response = final_response_chain.invoke({"query": state['query'], "context_json": context_json})
        final_response_content = response.content

    if state.get("plot"):
        final_response_content += "\n\nI have also generated the interactive plot you requested."
        
    return {"response": final_response_content}

# --- 4. Define Conditional Edges ---

def check_for_symbol(state: AgentState):
    """Checks if a stock symbol was provided or if it's a greeting."""
    if state.get("is_greeting"):
        return "generate_response"
    if state.get("stock_symbol"):
        return "decide_first_tool" 
    else:
        return "generate_response" 

def decide_first_tool(state: AgentState):
    """Decides which tool node to run first."""
    if state.get("request_data"):
        return "get_data"
    # *** FIX: Add new tool to logic ***
    if state.get("request_historical_calc"):
        return "calculate_profit_loss"
    if state.get("request_analysis"):
        return "get_analysis"
    if state.get("request_sentiment"):
        return "get_sentiment"
    if state.get("request_plot"):
        return "create_plot"
        
    # Fallback if user just types a ticker "TSLA"
    if not state['request_data'] and not state['request_analysis'] and not state['request_sentiment'] and not state['request_plot'] and not state['request_historical_calc']:
        state['request_data'] = True
        state['request_analysis'] = True
        state['request_sentiment'] = True
        return "get_data"
    return "generate_response" 

def after_data(state: AgentState):
    """After fetching data, decide the next step."""
    if state.get("request_historical_calc"):
        return "calculate_profit_loss"
    if state.get("request_analysis"):
        return "get_analysis"
    if state.get("request_sentiment"):
        return "get_sentiment"
    if state.get("request_plot"):
        return "create_plot"
    return "generate_response"

# *** FIX: Add new "after" function for our new node ***
def after_historical_calc(state: AgentState):
    """After profit/loss calc, decide the next step."""
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
workflow.add_node("calculate_profit_loss", calculate_profit_loss_node) # *** FIX: Add new node
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("decide_first_tool", lambda state: {}) 

# 1. Start at the router
workflow.set_entry_point("route_query")

# 2. Check if we have a symbol
workflow.add_conditional_edges(
    "route_query",
    check_for_symbol,
    {
        "generate_response": "generate_response",
        "decide_first_tool": "decide_first_tool", 
    }
)

# 3. Add the new routing logic from the decider node
workflow.add_conditional_edges(
    "decide_first_tool",
    decide_first_tool,
    {
        "get_data": "get_data",
        "calculate_profit_loss": "calculate_profit_loss", # *** FIX: Add new route
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
        "calculate_profit_loss": "calculate_profit_loss", # *** FIX: Add new route
        "get_analysis": "get_analysis",
        "get_sentiment": "get_sentiment",
        "create_plot": "create_plot",
        "generate_response": "generate_response"
    }
)

# *** FIX: Add conditional logic for new node ***
workflow.add_conditional_edges(
    "calculate_profit_loss",
    after_historical_calc,
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