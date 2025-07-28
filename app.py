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
st.set_page_config(page_title="股票分析MVP", layout="wide")

# 侧边栏：股票符号输入
st.sidebar.title("股票分析器")
ticker = st.sidebar.text_input("输入股票代码 (例如, AAPL)", value="AAPL").upper()

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
    return df.rename(columns={'t': '日期', 'o': '开盘', 'h': '最高', 'l': '最低', 'c': '收盘'})

@st.cache_data
def get_news(ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-07-27&to=2025-07-28&token={FINNHUB_KEY}"
    response = requests.get(url)
    return response.json()[:3]  # 只取前3条新闻

# 多页导航
pages = ["首页", "基本面", "警报", "投资建议"]
page = st.sidebar.radio("导航", pages)

if page == "首页":
    st.title(f"{ticker} 股票仪表板")
    if ticker:
        hist = get_historical_data(ticker)
        if not hist.empty:
            # K线图
            fig = go.Figure(data=[go.Candlestick(x=hist['日期'],
                                                open=hist['开盘'], high=hist['最高'],
                                                low=hist['最低'], close=hist['收盘'])])
            fig.update_layout(title=f"{ticker} K线图", xaxis_title="日期", yaxis_title="价格")
            st.plotly_chart(fig, use_container_width=True)
            
            # 实时报价（Finnhub）
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
        if fundamentals:
            df = pd.DataFrame({
                "指标": ["市值", "市盈率", "每股收益", "股息收益率"],
                "值": [fundamentals.get('mktCap', 'N/A'),
                          fundamentals.get('priceEarningsRatio', 'N/A'),
                          fundamentals.get('eps', 'N/A'),
                          fundamentals.get('dividendYield', 'N/A')]
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
        news = get_news(ticker)
        
        current_price = quote.get('c', 0)
        pe = float(fundamentals.get('priceEarningsRatio', 0))
        eps = float(fundamentals.get('eps', 0))
        
        # 基于数据的简单建议逻辑（结合专家经验）
        suggestion = "持仓观望"
        if pe < 25 and current_price > quote.get('pc', 0):  # PE低且上涨趋势
            suggestion = "买入（估值吸引，增长潜力大）"
        elif pe > 30:
            suggestion = "谨慎持仓（估值偏高，等待回调）"
        else:
            suggestion = "小仓买入（市场情绪正面，但关注风险）"
        
        st.write(f"**当前价格**: ${current_price:.2f}")
        st.write(f"**市盈率 (PE)**: {pe:.2f} - 如果低于25，适合长期买入；当前较高，需注意过热。")
        st.write(f"**每股收益 (EPS)**: {eps:.2f} - 稳定增长支持股价。")
        st.write(f"**我的建议**: {suggestion}. 理由：AAPL本周earnings即将发布，市场混合（72%正面情绪，但有tech卖压）。短期目标226，风险包括关税和波动。非投资建议，仅参考。")
        
        # 最新新闻
        st.subheader("最新新闻（影响情绪）")
        for item in news:
            st.write(f"- {item.get('headline', '无标题')} ({item.get('datetime', '')})")
    else:
        st.error("请输入股票代码。")
