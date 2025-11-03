import streamlit as st
from agent.graph import app  # Import the compiled LangGraph app

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Stock Analysis Chatbot",
    page_icon="📈",
    layout="wide"
)

# --- 2. Page Title ---
st.title("📈 AI Stock Analysis Chatbot")
st.caption(f"Powered by Gemini-2.5-Flash, LangGraph, and Python 3.10")

# --- 3. Portfolio Management (Simple) ---
with st.sidebar:
    st.header("My Portfolio")
    
    # Initialize portfolio in session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}

    # Input for new stock
    symbol = st.text_input("Stock Symbol (e.g., AAPL)", max_chars=5).upper()
    shares = st.number_input("Number of Shares", min_value=0, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add/Update"):
            if symbol and shares > 0:
                st.session_state.portfolio[symbol] = shares
                st.success(f"Updated {symbol} to {shares} shares")
            elif symbol and shares == 0:
                if symbol in st.session_state.portfolio:
                    del st.session_state.portfolio[symbol]
                    st.success(f"Removed {symbol}")
            else:
                st.warning("Please enter a valid symbol.")
    
    with col2:
        if st.button("Clear All"):
            st.session_state.portfolio = {}
            st.experimental_rerun()

    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("Current Holdings")
        total_value = 0  # Placeholder for future enhancement
        for sym, num_shares in st.session_state.portfolio.items():
            st.write(f"- **{sym}**: {num_shares} shares")
        st.info("Portfolio tracking is a demo. Data is not saved.")
            
    st.divider()
    st.markdown(
        "**Example Queries:**\n"
        "- *What's the current price of Tesla?*\n"
        "- *Show me the last 6 months' performance of Apple.*\n"
        "- *What is the RSI of Microsoft?*\n"
        "- *What's the market sentiment around NVIDIA?*\n"
        "- *Compare Apple and Google on P/E ratio.*\n"
    )


# --- 4. Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hello! Ask me about any stock to get started."}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the message has a plot, display it
        if "plot" in msg:
            st.plotly_chart(msg["plot"], use_container_width=True)

# Get user input
if prompt := st.chat_input("Ask your question... (e.g., 'Analyze TSLA')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 5. Call the LangGraph Agent ---
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🤖")
        
        # Prepare the input for the graph
        inputs = {"query": prompt}
        
        try:
            # invoke() runs the graph and returns the final state
            final_state = app.invoke(inputs)
            
            response_content = final_state.get("response", "I encountered an error.")
            plot_figure = final_state.get("plot")
            
            # Display the final response
            message_placeholder.markdown(response_content)
            
            # Store the AI response and plot (if any)
            ai_msg = {"role": "ai", "content": response_content}
            if plot_figure:
                st.plotly_chart(plot_figure, use_container_width=True)
                ai_msg["plot"] = plot_figure
                
            st.session_state.messages.append(ai_msg)

        except Exception as e:
            error_msg = f"Sorry, I ran into an error: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "ai", "content": error_msg})