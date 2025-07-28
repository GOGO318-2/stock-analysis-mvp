import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import datetime  # 新增：用于时间计算

# 你的API Key
FINNHUB_KEY = 'd1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180'
FMP_KEY = '8n2nsHP2Lj1uHkPRrtcQ8a63Lf95VjbU'
POLYGON_KEY = '2CDgF277xEhkhKndj5yFMVONxBGFFShg'  # 保留，但不用于历史

st.set_page_config(page_title="股票分析MVP", layout="wide")

st.sidebar.title("股票分析器")
st.sidebar.markdown("支持港股：输入如'0700.HK'")
ticker = st.sidebar.text_input("输入股票代码 (例如, AAPL 或 0700.HK)", value="AAPL").upper()

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
    # 使用Finnhub历史蜡烛图，支持港股
    to_date = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=days)
    from_unix = int(datetime.datetime.combine(from_date, datetime.time()).timestamp())
    to_unix = int(datetime.datetime.combine(to_date, datetime.time()).timestamp())
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={from_unix}&to={to_unix}&token={FINNHUB_KEY}"
    response = requests.get(url).json()
    if 'c' in response:
        df = pd.DataFrame({
            '日期': pd.to_datetime(response['t'], unit='s'),
            '开盘': response['o'],
            '最高': response['h'],
            '最低': response['l'],
            '收盘': response['c'],
            '成交量': response['v']
        })
        df.set_index('日期', inplace=True)
        return df
    else:
        return pd.DataFrame()  # 空DataFrame处理错误

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
        hist = get_historical_data(ticker, days=5)
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                open=hist['开盘'], high=hist['最高'],
                                                low=hist['最低'], close=hist['收盘'],
                                                name='K线')])
            ma5 = hist['收盘'].rolling(window=5).mean()
            ma20 = hist['收盘'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5 (短期)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20 (长期)', line=dict(color='orange')))
            fig.update_layout(title=f"{ticker} 本周K线图（可拖拽查看细节）", xaxis_title="日期", yaxis_title="价格",
                              xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')
            st.plotly_chart(fig, use_container_width=True)
            
            quote = get_real_time_quote(ticker)
            col1, col2, col3 = st.columns(3)
            col1.metric("当前价格", f"${quote.get('c', 'N/A'):.2f}")
            col2.metric("今日最高", f"${quote.get('h', 'N/A'):.2f}")
            col3.metric("今日最低", f"${quote.get('l', 'N/A'):.2f}")
        else:
            st.error("无效代码或无数据（检查API限额）。")

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
        
        # 表格数据：短期、趋势、波段
        data = [
            {"阶段": "短期交易 (日内/短期)", "时机": "入场", "价位": f"{round(current_price * 0.98, 2)}-{round(current_price * 1.02, 2)}", "触发电号": "RSI<40且MACD金叉", "仓位": "60%", "备忘": "分批买入，每批0.5张。持仓1-3天，关注成交量放大。"},
            {"阶段": "短期交易 (日内/短期)", "时机": "止盈", "价位": f"{round(current_price * 1.05, 2)}", "触发电号": "RSI>60或MA5死叉", "仓位": "减仓40%", "备忘": "无量冲高减仓，目标区间5-10%。"},
            {"阶段": "趋势交易 (长期)", "时机": "入场", "价位": f"{round(current_price * 0.95, 2)}-{current_price}", "触发电号": "MA20上穿且ROE>30%", "仓位": "加仓40%", "备忘": "长期持仓，忽略短期波动。目标1100+，持仓3-6月。"},
            {"阶段": "趋势交易 (长期)", "时机": "止损", "价位": f"{round(current_price * 0.90, 2)}", "触发电号": "跌破支持位907", "仓位": "清仓", "备忘": "如果新闻负面，快速止损。"},
            {"阶段": "波段交易 (中短期)", "时机": "入场", "价位": f"{round(current_price * 0.97, 2)}-{round(current_price * 1.03, 2)}", "触发电号": "MACD正向交叉且成交量>平均", "仓位": "70%", "备忘": "波段捕捉，持仓1-4周。分两批，关注X情绪72%正面。"},
            {"阶段": "波段交易 (中短期)", "时机": "止盈/止损", "价位": f"{round(current_price * 1.10, 2)} / {round(current_price * 0.92, 2)}", "触发电号": "突破阻力1225或跌破MA20", "仓位": "减仓50%", "备忘": "目标区间10-15%，风险包括tech卖压。"}
        ]
        df = pd.DataFrame(data)
        st.table(df)
        
        st.markdown("<span style='color:red'>重点: RSI超卖建议短期买入，支持位907，目标1101。非投资建议，请咨询专业人士。</span>", unsafe_allow_html=True)
        
        st.subheader("最新新闻（影响情绪）")
        for item in news:
            st.write(f"- {item.get('headline', '无标题')} ({item.get('datetime', '')})")
    else:
        st.error("请输入股票代码。")
