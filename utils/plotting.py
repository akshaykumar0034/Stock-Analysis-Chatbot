import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_stock_trend(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Creates an interactive Candlestick chart with Volume and Moving Averages."""
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f'{symbol} Candlestick', 'Volume'),
                        row_heights=[0.7, 0.3])

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    # 2. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', 
                             name='SMA 20', line=dict(color='yellow', width=1)),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', 
                             name='SMA 50', line=dict(color='orange', width=1)),
                  row=1, col=1)

    # 3. Volume Chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'),
                  row=2, col=1)
    
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        legend_title='Legend',
        height=600,
        template='plotly_dark' # Use a dark theme
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig