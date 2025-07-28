import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# 你的API Key（已集成，不要改）
FINNHUB_KEY = 'd1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180'
FMP_KEY = '8n2nsHP2Lj1uHkPRrtcQ8a63Lf95VjbU'
POLYGON_KEY = '2CDgF277xEhkhKndj5yFMVONxBGFFShg'

# 页面配置
st.set_page_config(page_title="Stock Analysis MVP", layout="wide")

# 侧边栏：股票符号输入
st.sidebar.title("Stock Analyzer")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL").upper()

# 数据获取函数（用你的API）
@st.cache_data
def get_real_time_quote(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
    response = requests.get(url)
    return response.json()

@st.cache_data
def get_fundamentals(ticker):
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_KEY}"
    response = requests.get(url)
    return response.json()[0] if response.json() else {}

@st.cache_data
def get_historical_data(ticker):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2025-07-28?apiKey={POLYGON_KEY}"
    response = requests.get(url)
    data = response.json().get('results', [])
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'})

# 多页导航
pages = ["Home", "Fundamentals", "Alerts"]
page = st.sidebar.radio("Navigate", pages)

if page == "Home":
    st.title(f"{ticker} Stock Dashboard")
    if ticker:
        hist = get_historical_data(ticker)
        if not hist.empty:
            # K线图
            fig = go.Figure(data=[go.Candlestick(x=hist['Date'],
                                                open=hist['Open'], high=hist['High'],
                                                low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            
            # 实时报价（Finnhub）
            quote = get_real_time_quote(ticker)
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${quote.get('c', 'N/A'):.2f}")
            col2.metric("High Today", f"${quote.get('h', 'N/A'):.2f}")
            col3.metric("Low Today", f"${quote.get('l', 'N/A'):.2f}")
        else:
            st.error("Invalid ticker or no data.")

elif page == "Fundamentals":
    st.title(f"{ticker} Fundamentals")
    if ticker:
        fundamentals = get_fundamentals(ticker)
        if fundamentals:
            df = pd.DataFrame({
                "Metric": ["Market Cap", "PE Ratio", "EPS", "Dividend Yield"],
                "Value": [fundamentals.get('mktCap', 'N/A'),
                          fundamentals.get('priceEarningsRatio', 'N/A'),
                          fundamentals.get('eps', 'N/A'),
                          fundamentals.get('dividendYield', 'N/A')]
            })
            st.table(df)
        else:
            st.error("No fundamental data.")

elif page == "Alerts":
    st.title("Price Alerts")
    threshold = st.number_input("Set Alert Threshold (e.g., price below)", value=100.0)
    if ticker:
        quote = get_real_time_quote(ticker)
        current_price = quote.get('c', 0)
        if current_price < threshold:
            st.warning(f"Alert: {ticker} price ${current_price:.2f} is below {threshold}!")
        else:
            st.success(f"{ticker} price is above threshold.")
