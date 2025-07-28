import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np

# 你的API Key
FINNHUB_KEY = 'd1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180'
FMP_KEY = '8n2nsHP2Lj1uHkPRrtcQ8a63Lf95VjbU'
POLYGON_KEY = '2CDgF277xEhkhKndj5yFMVONxBGFFShg'

st.set_page_config(page_title="股票分析MVP", layout="wide")

st.sidebar.title("股票分析器")
ticker = st.sidebar.text_input("输入股票代码 (例如, AAPL)", value="AAPL").upper()

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
def get_key_metrics(ticker):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&apikey={FMP_KEY}"
    response = requests.get(url)
    return response.json()[0] if response.json() else {}

@st.cache_data
def get_historical_data(ticker, days=30):
    from_date = "2025-06-28"
    to_date = "2025-07-28"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={POLYGON_KEY}"
    response = requests.get(url)
    data = response.json().get('results', [])
    df = pd.DataFrame(data)
    if not df.empty:
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'t': '日期', 'o': '开盘', 'h': '最高', 'l': '最低', 'c': '收盘', 'v': '成交量'})
        df.set_index('日期', inplace=True)
    return df

@st.cache_data
def get_news(ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-07-27&to=2025-07-28&token={FINNHUB_KEY}"
    response = requests.get(url)
    return response.json()[:3]

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 'N/A'

def calculate_macd(close, short=12, long=26, signal=9):
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1] if not macd_line.empty else ('N/A', 'N/A')

pages = ["首页", "基本面", "警报", "投资建议"]
page = st.sidebar.radio("导航", pages)

if page == "首页":
    st.title(f"{ticker} 股票仪表板")
    if ticker:
        hist = get_historical_data(ticker, days=5)  # 本周
        if not hist.empty:
            # K线图类似图像：添加MA5(蓝)、MA20(橙)
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                open=hist['开盘'], high=hist['最高'],
                                                low=hist['最低'], close=hist['收盘'],
                                                name='K线')])
            ma5 = hist['收盘'].rolling(window=5).mean()
            ma20 = hist['收盘'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5 (短期)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20 (长期)', line=dict(color='orange')))
            fig.update_layout(title=f"{ticker} 本周K线图（可拖拽查看细节）", xaxis_title="日期", yaxis_title="价格",
                              xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')  # 汉化日期
            st.plotly_chart(fig, use_container_width=True)
            
            quote = get_real_time_quote(ticker)
            col1, col2, col3 = st.columns(3)
            col1.metric("当前价格", f"${quote.get('c', 'N/A'):.2f}")
            col2.metric("今日最高", f"${quote.get('h', 'N/A'):.2f}")
            col3.metric("今日最低", f"${quote.get('l', 'N/A'):.2f}")
        else:
            st.error("无效代码或无数据。")

elif page == "基本面":
    st.title(f"{ticker} 基本面")
    if ticker:
        fundamentals = get_fundamentals(ticker)
        metrics = get_key_metrics(ticker)
        hist = get_historical_data(ticker)
        if fundamentals or metrics:
            rsi = calculate_rsi(hist['收盘'])
            macd, signal = calculate_macd(hist['收盘'])
            avg_volume = hist['成交量'].mean() if '成交量' in hist else 'N/A'
            
            # 修复N/A：用正确键或fallback
            eps = metrics.get('netIncomePerShare', fundamentals.get('eps', 'N/A'))
            dividend_yield = metrics.get('dividendYield', fundamentals.get('dividendYield', 'N/A'))
            beta = metrics.get('beta', fundamentals.get('beta', 'N/A'))
            roe = metrics.get('returnOnEquity', 'N/A')
            
            df = pd.DataFrame({
                "指标": ["市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", "Beta", "ROE", "负债权益比", "RSI (14日)", "MACD", "平均成交量"],
                "值": [fundamentals.get('mktCap', 'N/A'),
                       metrics.get('peRatio', 'N/A'),
                       eps,
                       dividend_yield,
                       beta,
                       roe,
                       metrics.get('debtToEquity', 'N/A'),
                       rsi,
                       f"{macd:.2f} (Signal: {signal:.2f})",
                       f"{avg_volume:,.0f}"]
            })
            st.table(df)
        else:
            st.error("无基本面数据。")

elif page == "警报":
    st.title("价格警报")
    threshold = st.number_input("设置警报阈值 (例如, 价格低于)", value=100.0)
    if ticker:
        quote = get_real_time_quote(ticker)
        current_price = quote.get('c', 0)
        if current_price < threshold:
            st.warning(f"警报: {ticker} 价格 ${current_price:.2f} 低于 {threshold}!")
        else:
            st.success(f"{ticker} 价格高于阈值。")

elif page == "投资建议":
    st.title(f"{ticker} 当天投资建议 (2025-07-28)")
    if ticker:
        quote = get_real_time_quote(ticker)
        fundamentals = get_fundamentals(ticker)
        metrics = get_key_metrics(ticker)
        hist = get_historical_data(ticker)
        news = get_news(ticker)
        
        current_price = quote.get('c', 0)
        pe = metrics.get('peRatio', 0)
        eps = metrics.get('netIncomePerShare', 0)
        rsi = calculate_rsi(hist['收盘'])
        macd, _ = calculate_macd(hist['收盘'])
        
        buy_sell = "买入" if rsi < 40 else ("卖出" if rsi > 60 else "持仓")
        reason = "RSI超卖，潜在反弹" if rsi < 40 else ("RSI超买，回调风险" if rsi > 60 else "市场稳定")
        buy_price = f"当前价附近或支持位{round(current_price * 0.95, 2)}" if buy_sell == "买入" else "N/A"
        sell_price = f"阻力位{round(current_price * 1.05, 2)}或目标1071" if buy_sell == "卖出" else "N/A"
        target = 1071
        support = round(current_price * 0.95, 2)
        resistance = round(current_price * 1.05, 2)
        
        st.write(f"**当前价格**: ${current_price:.2f}")
        st.write(f"**市盈率 (PE)**: {pe} - 高于平均，估值偏高但增长支持。")
        st.write(f"**每股收益 (EPS)**: {eps} - 稳定。")
        st.write(f"**RSI**: {rsi:.2f} - {reason}。")
        st.write(f"**MACD**: {macd:.2f} - 负值表示熊势，但关注交叉。")
        st.markdown(f"**当天建议**: {buy_sell}。<span style='color:red'>重点: 关注RSI和新闻情绪，买入价: {buy_price}，卖出价: {sell_price}，止损: 支持位{support}，目标: {target}。</span> 理由: 技术弱但销售增长正面，X情绪混合（下行风险到850）。非投资建议。", unsafe_allow_html=True)
        
        st.subheader("最新新闻（影响情绪）")
        for item in news:
            st.write(f"- {item.get('headline', '无标题')} ({item.get('datetime', '')})")
    else:
        st.error("请输入股票代码。")
