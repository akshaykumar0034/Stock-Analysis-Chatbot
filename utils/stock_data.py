import yfinance as yf
import pandas as pd

def get_ticker(symbol: str):
    """Gets the yfinance Ticker object."""
    return yf.Ticker(symbol)

def get_current_price(symbol: str) -> float:
    """Gets the last closing price for a stock."""
    ticker = get_ticker(symbol)
    # Use '1d' period to get the most recent data
    data = ticker.history(period="1d")
    if not data.empty:
        return data['Close'].iloc[-1]
    
    # Fallback for delisted or invalid tickers
    try:
        # Try to get 'currentPrice' from info
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
    
    # Extract key fundamental metrics
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

def get_stock_news(symbol: str) -> list:
    """Gets recent news article titles for a stock."""
    ticker = get_ticker(symbol)
    news = ticker.news
    # Return top 5 news titles
    return [article['title'] for article in news[:5]]