import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json  # Added for Grok API

st.set_page_config(page_title="股票分析MVP", layout="wide")

# Sidebar
st.sidebar.title("股票分析器")
st.sidebar.markdown("支持港股：输入如0700")
ticker_input = st.sidebar.text_input("输入股票代码 (例如, TSLA 或 0700)", value="TSLA").upper()

# Auto append .HK for Hong Kong stocks, remove leading zeros
if ticker_input.isdigit():
    ticker_clean = ticker_input.lstrip('0')
    if 1 <= len(ticker_clean) <= 5:
        ticker = ticker_clean + '.HK'
    else:
        ticker = ticker_input
else:
    ticker = ticker_input

# Watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if st.sidebar.button("添加到Watchlist"):
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.sidebar.success("添加成功！")
st.sidebar.subheader("Watchlist")
for wl_ticker in st.session_state.watchlist:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(wl_ticker)
    if col2.button("移除", key=f"remove_{wl_ticker}"):
        st.session_state.watchlist.remove(wl_ticker)
        st.rerun()
    if col1.button("查询", key=f"query_{wl_ticker}"):
        ticker_input = wl_ticker
        st.rerun()

# API keys
API_KEYS = {
    "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",
    "alpha_vantage": "Z45S0SLJGM378PIO",
    "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",
    "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
}
API_ORDER = ["yfinance", "alpha_vantage", "finnhub", "polygon"]

# Data fetching
@st.cache_data
def get_stock_data(ticker):
    for api in API_ORDER:
        try:
            time.sleep(1)  # Rate limit
            if api == "yfinance":
                stock = yf.Ticker(ticker)
                info = stock.info
                rec = stock.recommendations_summary if hasattr(stock, 'recommendations_summary') and not stock.recommendations.empty else pd.DataFrame()
                if info:
                    return info, rec
            elif api == "finnhub":
                quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={API_KEYS['finnhub']}"
                quote_resp = requests.get(quote_url).json()
                if 'c' in quote_resp:
                    info = {'currentPrice': quote_resp['c'], 'dayHigh': quote_resp['h'], 'dayLow': quote_resp['l'], 'preMarketPrice': 'N/A', 'postMarketPrice': 'N/A'}
                    return info, pd.DataFrame()
        except Exception as e:
            st.warning(f"{api} 获取数据失败: {e}")
            continue
    st.error("所有API和备用数据源均失败，请稍后重试或检查网络。")
    return {}, pd.DataFrame()

@st.cache_data
def get_historical_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist if not hist.empty else pd.DataFrame()
    except Exception as e:
        st.warning(f"历史数据获取失败: {e}")
        return pd.DataFrame()

# Technical indicators
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1] if not rs.empty else 50

def calculate_macd(close, short=12, long=26, signal=9):
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1] if not macd_line.empty else (0, 0)

def calculate_bollinger_bands(close, window=20, std_dev=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

# News and sentiment
def get_news_and_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        news_list = []
        positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'growth']
        negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'decline']
        for item in news:
            title_lower = item.get('title', '').lower()
            sent_label = "正面" if any(kw in title_lower for kw in positive_keywords) else "负面" if any(kw in title_lower for kw in negative_keywords) else "中性"
            news_list.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'publish_date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment': sent_label
            })
        return news_list
    except Exception as e:
        st.warning(f"新闻获取失败: {e}")
        return []

def get_x_sentiment(ticker):
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEYS['xai']}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"What is the current sentiment on X for stock {ticker}? One word: 正面, 负面, or 中性."}]
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.warning(f"Grok API 情绪分析失败: {e}")
        return "中性"

def get_grok_remark(ticker):
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEYS['xai']}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"Provide a short remark for investing in {ticker}."}]
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.warning(f"Grok 备注失败: {e}")
        return "No advice."

# Pages
pages = ["首页", "基本面", "投资建议", "公共市场"]
page = st.sidebar.radio("导航", pages)

info, rec = get_stock_data(ticker)
currency = info.get('currency', 'USD')
company_name = info.get('longName', ticker) or ticker

if page == "首页":
    st.title(f"{company_name} ({ticker}) 股票仪表板")
    period_options = {"1日": "1d", "5日": "5d", "日K": "1mo", "周K": "3mo", "月K": "1y", "季K": "5y"}
    default_index = list(period_options.keys()).index("月K")
    selected_label = st.selectbox("选择时间范围", list(period_options.keys()), index=default_index)
    selected_period = period_options[selected_label]
    hist = get_historical_data(ticker, selected_period)
    if not hist.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K线')])
        ma5 = hist['Close'].rolling(window=5).mean()
        ma20 = hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20', line=dict(color='orange')))
        upper, middle, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(go.Scatter(x=hist.index, y=upper, mode='lines', name='Upper BB', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=lower, mode='lines', name='Lower BB', line=dict(color='green', dash='dash')))
        fig.update_layout(title=f"{ticker} {selected_label}K线图", xaxis_title="日期", yaxis_title="价格", xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("当前价格", f"{info.get('currentPrice', 'N/A'):.2f} {currency}")
        col2.metric("今日最高", f"{info.get('dayHigh', 'N/A'):.2f} {currency}")
        col3.metric("今日最低", f"{info.get('dayLow', 'N/A'):.2f} {currency}")
        if st.button("收藏"):
            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.success("收藏成功！")
    else:
        st.error("无历史数据可用。")
    if currency == 'USD':
        st.subheader("盘前/盘后")
        if st.button("刷新"):
            with st.spinner('刷新中...'):
                time.sleep(1)
                try:
                    new_info, _ = get_stock_data(ticker)
                    info = new_info
                    st.success("刷新成功！")
                except:
                    st.error("刷新失败。")
                st.rerun()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("盘前价格", f"{info.get('preMarketPrice', 'N/A'):.2f} {currency}")
        col2.metric("盘前变化", f"{info.get('preMarketChange', 0):.2f}")
        col3.metric("盘后价格", f"{info.get('postMarketPrice', 'N/A'):.2f} {currency}")
        col4.metric("盘后变化", f"{info.get('postMarketChange', 0):.2f}")

elif page == "基本面":
    st.title(f"{company_name} ({ticker}) 基本面")
    if info:
        hist = get_historical_data(ticker, "1mo")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close'])
            macd, signal = calculate_macd(hist['Close'])
            avg_volume = hist['Volume'].mean()
            returns = hist['Close'].pct_change()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 'N/A'
            df = pd.DataFrame({
                "指标": ["市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", "Beta", "ROE", "负债权益比", "RSI (14日)", "MACD", "平均成交量", "Sharpe Ratio"],
                "值": [info.get('marketCap', 'N/A'), info.get('trailingPE', 'N/A'), info.get('trailingEps', 'N/A'), info.get('dividendYield', 'N/A'),
                       info.get('beta', 'N/A'), info.get('returnOnEquity', 'N/A'), info.get('debtToEquity', 'N/A'), rsi,
                       f"{macd:.2f} (Signal: {signal:.2f})", f"{avg_volume:,.0f}", f"{sharpe:.2f}" if sharpe != 'N/A' else 'N/A']
            })
            st.table(df)
        else:
            st.error("无历史数据，无法计算指标。")
    else:
        st.error("无基本面数据。")

elif page == "投资建议":
    st.title(f"{company_name} ({ticker}) 投资建议")
    if info:
        hist = get_historical_data(ticker, "1mo")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close'])
            macd, _ = calculate_macd(hist['Close'])
            news = get_news_and_sentiment(ticker)
            news_sentiment = "中性"
            if news:
                sentiments = [n['sentiment'] for n in news]
                if sentiments.count("正面") > sentiments.count("负面"):
                    news_sentiment = "正面"
                elif sentiments.count("负面") > sentiments.count("正面"):
                    news_sentiment = "负面"
            current_price = info.get('currentPrice', 0)
            target_price = info.get('targetMeanPrice', current_price * 1.1)
            support = current_price * 0.95
            resistance = current_price * 1.05
            x_sentiment = get_x_sentiment(ticker)
            remark = get_grok_remark(ticker)
            data = [
                {"阶段": "短期", "时机": "入场", "价位": f"{support:.0f}-{current_price:.0f}", "仓位": "60%" if rsi < 40 else "减仓40%", "备注": f"RSI {rsi:.0f} {news_sentiment}"},
                {"阶段": "短期", "时机": "止盈", "价位": f"{resistance:.0f}", "仓位": "60%" if rsi < 40 else "减仓40%", "备注": f"RSI {rsi:.0f} {news_sentiment}"},
                {"阶段": "趋势", "时机": "入场", "价位": f"{support:.0f}-{current_price:.0f}", "仓位": "加仓40%", "备注": f"长期持仓 {x_sentiment}"},
                {"阶段": "趋势", "时机": "止损", "价位": f"{support * 0.95:.0f}", "仓位": "清仓", "备注": f"长期持仓 {x_sentiment}"},
                {"阶段": "波段", "时机": "入场", "价位": f"{support:.0f}-{resistance:.0f}", "仓位": "70%" if macd > 0 else "减仓50%", "备注": f"MACD {macd:.2f} {x_sentiment}"},
                {"阶段": "波段", "时机": "止盈/止损", "价位": f"{target_price:.0f}/{support:.0f}", "仓位": "70%" if macd > 0 else "减仓50%", "备注": f"MACD {macd:.2f} {x_sentiment}"}
            ]
            df = pd.DataFrame(data)
            trade_type = st.selectbox("选择交易类型", ["所有", "短期", "趋势", "波段"], index=0)
            if trade_type != "所有":
                df = df[df["阶段"] == trade_type]
            st.table(df)
            st.markdown(f"<span style='color:red'>建议: RSI{rsi:.0f} {('买入' if rsi < 40 else '卖出' if rsi > 60 else '持仓')}，目标{target_price:.0f}。{remark}</span>", unsafe_allow_html=True)
            if st.button("收藏"):
                if ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker)
                    st.success("收藏成功！")
        else:
            st.error("无历史数据，无法生成建议。")
    else:
        st.error("无投资建议数据。")

elif page == "公共市场":
    st.title("公共市场 - Top 50 科技美股推荐")
    tech_stocks = ['NVDA', 'TSLA', 'AMD', 'GOOG', 'AAPL', 'MSFT', 'AMZN', 'META', 'INTC', 'QCOM', 'IBM', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'EBAY', 'NFLX', 'DIS', 'BABA', 'JD', 'BIDU', 'TSM', 'ASML', 'MU', 'KLAC', 'LRCX', 'AVGO', 'TXN', 'STM', 'ARM', 'SNPS', 'CDNS', 'ANSS', 'KEYS', 'TER', 'SWKS', 'QRVO', 'MPWR', 'MCHP', 'ON', 'NXPI', 'ADI', 'LSCC', 'SYNA', 'POWI', 'SLAB', 'CRUS', 'MTSI', 'RMBS', 'CEVA', 'SGH', 'SITM']
    if st.button("更新Top 50"):
        with st.spinner('筛选股票中...'):
            stock_data = []
            for tick in tech_stocks:
                try:
                    time.sleep(1)
                    info, _ = get_stock_data(tick)
                    if not info:
                        continue
                    hist = get_historical_data(tick, "1wk")
                    if hist.empty or len(hist) < 2:
                        continue
                    volume_avg = hist['Volume'].mean()
                    turnover_avg = volume_avg * hist['Close'].mean()
                    pct_change = (info['currentPrice'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                    sentiment = get_x_sentiment(tick)
                    activity_score = (volume_avg / 1e8) + (2 if sentiment == "正面" else -2 if sentiment == "负面" else 0)
                    buy_level = "高" if activity_score > 5 else "中" if activity_score > 2 else "低"
                    buy_price = info['currentPrice'] * 0.95
                    remark = get_grok_remark(tick)
                    stock_data.append({
                        '股票代码': tick, '公司名称': info.get('longName', tick), '市值': f"{info.get('marketCap', 0):,.0f}",
                        '成交额': f"{turnover_avg:,.0f}", '成交量': f"{volume_avg:,.0f}", '价格': info.get('currentPrice', 0),
                        '最高': info.get('dayHigh', 0), '最低': info.get('dayLow', 0), '涨幅': f"{pct_change:.2f}%",
                        '买入等级': buy_level, '买入价': f"{buy_price:.2f}", '备注': remark
                    })
                except:
                    continue
            if stock_data:
                df = pd.DataFrame(stock_data)
                df['排序'] = df['买入等级'].map({'高': 3, '中': 2, '低': 1})
                df = df.sort_values('排序', ascending=False).drop('排序', axis=1).head(50)
                st.session_state['top50'] = df
                st.success("更新完成！")
            else:
                st.error("无数据可显示，请稍后重试。")
    if 'top50' in st.session_state:
        st.table(st.session_state['top50'])
        if st.button("收藏"):
            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.success("收藏成功！")
    else:
        st.info("点击'更新Top 50'开始筛选。")
