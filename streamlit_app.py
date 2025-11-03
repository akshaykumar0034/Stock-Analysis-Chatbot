import streamlit as st
from agent.graph import app  # Import the compiled LangGraph app
from utils.database import init_db
from sqlalchemy import text  # <--- IMPORT text HERE

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Stock Analysis Chatbot",
    page_icon="📈",
    layout="wide"
)

# --- 2. Initialize Database ---
try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize database: {e}")

# --- 3. Connect to the Database ---
conn = st.connection("stock_db", type="sql", url="sqlite:///stock_app.db")

# --- 4. Page Title ---
st.title("📈 AI Stock Analysis Chatbot")
st.caption(f"Powered by Gemini-2.5-Flash, LangGraph, and Python 3.12")

# --- 5. Portfolio Management (with DB) ---
with st.sidebar:
    st.header("My Portfolio")
    
    # Load portfolio from DB into session_state on first run
    if 'portfolio' not in st.session_state:
        try:
            # NOTE: conn.query() does NOT need text()
            portfolio_df = conn.query("SELECT symbol, shares FROM portfolio")
            st.session_state.portfolio = dict(zip(portfolio_df['symbol'], portfolio_df['shares']))
        except Exception as e:
            st.error(f"Failed to load portfolio: {e}")
            st.session_state.portfolio = {}

    # Input for new stock
    symbol = st.text_input("Stock Symbol (e.g., AAPL)", max_chars=5).upper()
    shares = st.number_input("Number of Shares", min_value=0, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add/Update"):
            if symbol and shares > 0:
                # --- DB WRITE ---
                with conn.session as s:
                    # --- FIX: Wrap in text() ---
                    s.execute(
                        text("INSERT OR REPLACE INTO portfolio (symbol, shares) VALUES (:symbol, :shares)"),
                        {"symbol": symbol, "shares": shares}
                    )
                    s.commit()
                st.session_state.portfolio[symbol] = shares
                st.success(f"Updated {symbol} to {shares} shares")
            elif symbol and shares == 0:
                if symbol in st.session_state.portfolio:
                    # --- DB WRITE ---
                    with conn.session as s:
                        # --- FIX: Wrap in text() ---
                        s.execute(
                            text("DELETE FROM portfolio WHERE symbol = :symbol"), 
                            {"symbol": symbol}
                        )
                        s.commit()
                    del st.session_state.portfolio[symbol]
                    st.success(f"Removed {symbol}")
            else:
                st.warning("Please enter a valid symbol.")
    
    with col2:
        if st.button("Clear All"):
            # --- DB WRITE ---
            with conn.session as s:
                # --- FIX: Wrap in text() ---
                s.execute(text("DELETE FROM portfolio"))
                s.commit()
            st.session_state.portfolio = {}
            st.rerun()

    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("Current Holdings")
        for sym, num_shares in st.session_state.portfolio.items():
            st.write(f"- **{sym}**: {num_shares} shares")
            
    st.divider()
    st.markdown(
        "**Example Queries:**\n"
        "- *What's the current price of Tesla?*\n"
        "- *Show me the last 6 months' performance of Apple.*\n"
    )

# --- 6. Chat Interface (with DB) ---

# Load chat history from DB into session_state on first run
if "messages" not in st.session_state:
    try:
        # NOTE: conn.query() does NOT need text()
        messages_df = conn.query("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
        st.session_state.messages = messages_df.to_dict("records")
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
        st.session_state.messages = []
        
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "ai", "content": "Hello! Ask me about any stock to get started."}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plot" in msg:
            st.plotly_chart(msg["plot"], width='stretch')

# Get user input
if prompt := st.chat_input("Ask your question... (e.g., 'Analyze TSLA')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # --- DB WRITE (User Message) ---
    try:
        with conn.session as s:
            # --- FIX: Wrap in text() ---
            s.execute(
                text("INSERT INTO chat_history (role, content) VALUES ('user', :content)"), 
                {"content": prompt}
            )
            s.commit()
    except Exception as e:
        st.error(f"Failed to save user message: {e}")

    # --- 7. Call the LangGraph Agent ---
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🤖")
        
        inputs = {"query": prompt}
        
        try:
            final_state = app.invoke(inputs)
            response_content = final_state.get("response", "I encountered an error.")
            plot_figure = final_state.get("plot")
            
            message_placeholder.markdown(response_content)
            
            # --- DB WRITE (AI Message) ---
            try:
                with conn.session as s:
                    # --- FIX: Wrap in text() ---
                    s.execute(
                        text("INSERT INTO chat_history (role, content) VALUES ('ai', :content)"), 
                        {"content": response_content}
                    )
                    s.commit()
            except Exception as e:
                st.error(f"Failed to save AI message: {e}")
            
            ai_msg = {"role": "ai", "content": response_content}
            if plot_figure:
                st.plotly_chart(plot_figure, width='stretch')
                ai_msg["plot"] = plot_figure
                
            st.session_state.messages.append(ai_msg)

        except Exception as e:
            error_msg = f"Sorry, I ran into an error: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "ai", "content": error_msg})