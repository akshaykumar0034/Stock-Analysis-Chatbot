import pandas_ta as ta
import pandas as pd

def calculate_ta_indicators(historical_data: pd.DataFrame) -> dict:
    """Calculates key technical indicators and returns the latest values."""
    if historical_data.empty:
        return {}
        
    df = historical_data.copy()
    
    # Calculate indicators
    df.ta.rsi(append=True)       # RSI (RSI_14)
    df.ta.macd(append=True)      # MACD (MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
    df.ta.bbands(append=True)    # Bollinger Bands (BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
    
    # Get the very last row of data
    latest_indicators = df.iloc[-1]
    
    ta_results = {
        "rsi": latest_indicators.get("RSI_14"),
        "macd": latest_indicators.get("MACD_12_26_9"),
        "macd_signal": latest_indicators.get("MACDs_12_26_9"),
        "macd_hist": latest_indicators.get("MACDh_12_26_9"),
        "bollinger_upper": latest_indicators.get("BBU_20_2.0"),
        "bollinger_mid": latest_indicators.get("BBM_20_2.0"),
        "bollinger_lower": latest_indicators.get("BBL_20_2.0"),
    }
    
    # Clean up NaN values for JSON compatibility
    return {k: (v if pd.notna(v) else None) for k, v in ta_results.items()}