import streamlit as st
from sqlalchemy import text  # <--- IMPORT THIS

def init_db():
    """
    Initializes the database by creating tables if they don't exist.
    """
    # Use the 'url' parameter for an unambiguous SQLite connection
    conn = st.connection("stock_db", type="sql", url="sqlite:///stock_app.db")
    
    with conn.session as s:
        # --- FIX: Wrap the SQL string in text() ---
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolio (
                symbol TEXT PRIMARY KEY,
                shares INTEGER NOT NULL
            );
        """))
        
        # --- FIX: Wrap the SQL string in text() ---
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT_NULL,
                content TEXT NOT_NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """))
        s.commit()