import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import finnhub_client  # Import our Finnhub client

# --- yfinance Data Functions ---

def get_ticker(symbol: str):
    """Gets the yfinance Ticker object."""
    return yf.Ticker(symbol)

def get_current_price(symbol: str) -> float:
    """Gets the last closing price for a stock."""
    ticker = get_ticker(symbol)
    data = ticker.history(period="1d")
    if not data.empty:
        return data['Close'].iloc[-1]
    
    try:
        return ticker.info['currentPrice']
    except Exception:
        raise ValueError(f"Could not retrieve current price for {symbol}.")

def get_historical_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Gets historical stock data."""
    ticker = get_ticker(symbol)
    return ticker.history(period=period)

def get_stock_info(symbol: str) -> dict:
    """Gets fundamental data for a stock."""
    ticker = get_ticker(symbol)
    info = ticker.info
    
    fundamentals = {
        "companyName": info.get("longName"),
        "marketCap": info.get("marketCap"),
        "peRatio": info.get("trailingPE"),
        "forwardPeRatio": info.get("forwardPE"),
        "dividendYield": info.get("dividendYield"),
        "eps": info.get("trailingEps"),
        "beta": info.get("beta"),
        "52WeekHigh": info.get("fiftyTwoWeekHigh"),
        "52WeekLow": info.get("fiftyTwoWeekLow"),
        "averageVolume": info.get("averageVolume"),
    }
    return fundamentals

# --- Finnhub News Function ---

def get_stock_news(symbol: str) -> list:
    """
    Gets recent company news headlines for a stock using Finnhub.
    """
    try:
        # Get dates for the last 7 days
        today = datetime.now().strftime('%Y-%m-%d')
        one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        # Fetch news from Finnhub
        news_list = finnhub_client.company_news(symbol, _from=one_week_ago, to=today)

        if not news_list:
            return []  # No news found

        # Extract just the headlines, limiting to top 5
        headlines = [article['headline'] for article in news_list[:5] if 'headline' in article]
        
        return headlines

    except Exception as e:
        print(f"Warning: Could not fetch news for {symbol} from Finnhub: {e}")
        return []  # Return empty list on any error