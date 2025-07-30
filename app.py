import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 配置信息
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
    },
    'cache_timeout': 300,  # 5分钟缓存
    'news_api': {
        'url': 'https://newsapi.org/v2/everything',
        'key': '你的NewsAPI密钥'  # 去newsapi.org免费申请
    }
}

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 数据获取通用函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_yfinance_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """统一通过 yfinance 获取股票数据（兼容港股，如 0700.HK）"""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        logger.error(f"yfinance 获取 {ticker} 数据失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Dict:
    """获取股票基本信息"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        logger.error(f"获取 {ticker} 基本信息失败: {e}")
        return {}

@st.cache_data(ttl=3600)  # 新闻缓存1小时
def get_market_news(keyword: str) -> List[Dict]:
    """通过 NewsAPI 获取市场新闻（免费方案）"""
    params = {
        'q': keyword,
        'apiKey': CONFIG['news_api']['key'],
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 5
    }
    try:
        response = requests.get(CONFIG['news_api']['url'], params=params, timeout=10)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            # 简单处理新闻时间和情感
            news_list = []
            for art in articles:
                news_list.append({
                    'title': art.get('title', ''),
                    'link': art.get('url', ''),
                    'publish_date': art.get('publishedAt', '')[:16].replace('T', ' '),
                    'source': art.get('source', {}).get('name', 'Unknown'),
                    'sentiment': '中性'  # 免费版简化情感，可后期扩展
                })
            return news_list
        else:
            logger.error(f"NewsAPI 请求失败，状态码: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取新闻失败: {e}")
        return []

# -------------------- 技术分析函数 --------------------
def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    """计算 RSI 指标"""
    if len(close) < period:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_macd(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[float, float]:
    """计算 MACD 指标"""
    if len(close) < long:
        return 0.0, 0.0
    ema_short = close.ewm(span=short).mean()
    ema_long = close.ewm(span=long).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

# -------------------- AI 分析（容错处理） --------------------
@st.cache_data(ttl=600)
def get_ai_sentiment(ticker: str) -> str:
    """AI 情绪分析（增加容错）"""
    try:
        # 这里可替换为更稳定的免费 AI 服务，如 Hugging Face 免费模型
        return "中性"  # 先简化，后期可扩展
    except Exception as e:
        logger.error(f"AI 情绪分析失败: {e}")
        return "中性（分析失败）"

@st.cache_data(ttl=600)
def get_ai_advice(ticker: str, rsi: float, macd: float) -> str:
    """AI 投资建议（增加容错）"""
    try:
        # 同理，替换为免费稳定方案
        return "暂无详细建议（免费版简化）"
    except Exception as e:
        logger.error(f"AI 建议失败: {e}")
        return "API 错误，无法获取建议"

# -------------------- 热门股票动态推荐 --------------------
@st.cache_data(ttl=3600)  # 每小时更新
def get_trending_stocks() -> pd.DataFrame:
    """动态获取热门股票（示例：用美股+港股热门，可扩展）"""
    # 这里可对接免费的热门股票 API，如 https://finnhub.io 免费版
    sample_tickers = ['TSLA', 'AAPL', '0700.HK', 'TENCENT', 'NVDA']  # 示例，需替换
    trending_data = []
    for ticker in sample_tickers:
        info = get_stock_info(ticker)
        if not info:
            continue
        trending_data.append({
            '代码': ticker,
            '名称': info.get('longName', ticker),
            '价格': info.get('currentPrice', 0),
            '涨跌幅': info.get('regularMarketChangePercent', 0)
        })
    return pd.DataFrame(trending_data)

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    """实时数据页面"""
    hist = get_yfinance_data(ticker, '1y')
    info = get_stock_info(ticker)
    if hist.empty or not info:
        st.error("❌ 无法获取股票数据，请检查代码或稍后重试")
        return

    st.title(f"📊 {info.get('longName', ticker)} 实时数据")
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前价格", f"{info.get('currentPrice', 0):.2f} {info.get('currency', 'USD')}")
    col2.metric("今日最高", f"{info.get('dayHigh', 'N/A'):.2f}")
    col3.metric("今日最低", f"{info.get('dayLow', 'N/A'):.2f}")
    col4.metric("成交量", f"{info.get('volume', 0):,}")

    # K 线图
    fig = go.Figure(data=go.Candlestick(
        x=hist.index,
        open=hist['Open'], high=hist['High'],
        low=hist['Low'], close=hist['Close']
    ))
    fig.update_layout(title=f"{ticker} K 线图", height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_technical_page(ticker: str):
    """技术分析页面"""
    hist = get_yfinance_data(ticker, '1y')
    info = get_stock_info(ticker)
    if hist.empty or not info:
        st.error("❌ 数据获取失败")
        return

    st.title(f"📈 {ticker} 技术分析")
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])

    # 指标卡片
    col1, col2 = st.columns(2)
    col1.metric("RSI(14)", f"{rsi:.2f}")
    col2.metric("MACD", f"{macd:.2f} / {signal:.2f}")

    # RSI 趋势图
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(14).apply(calculate_rsi), name='RSI'))
    fig_rsi.update_layout(title="RSI 趋势", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

def render_advice_page(ticker: str):
    """投资建议页面"""
    hist = get_yfinance_data(ticker, '3mo')
    info = get_stock_info(ticker)
    if hist.empty or not info:
        st.error("❌ 数据不足，无法生成建议")
        return

    rsi = calculate_rsi(hist['Close'])
    macd, _ = calculate_macd(hist['Close'])
    sentiment = get_ai_sentiment(ticker)
    ai_advice = get_ai_advice(ticker, rsi, macd)

    st.title(f"🎯 {ticker} 投资建议")
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{rsi:.2f}")
    col2.metric("市场情绪", sentiment)
    col3.write(f"AI 建议：{ai_advice}")

    # 风险提示
    st.warning("⚠️ 投资有风险，建议仅供参考")

def render_trending_page():
    """热门股票页面"""
    st.title("🌟 动态热门股票推荐")
    if st.button("🔄 手动刷新热门股票"):
        with st.spinner("正在获取最新数据..."):
            trending_df = get_trending_stocks()
            st.session_state['trending_stocks'] = trending_df
            st.success("数据更新完成！")

    if 'trending_stocks' in st.session_state:
        st.dataframe(st.session_state['trending_stocks'], hide_index=True, use_container_width=True)
    else:
        st.info("点击上方按钮获取动态热门股票")

def render_news_page(ticker: str):
    """市场新闻页面"""
    st.title(f"📰 {ticker} 市场新闻")
    news_list = get_market_news(ticker)
    if not news_list:
        st.warning("暂无有效新闻，可尝试更换股票代码")
        return

    # 新闻统计
    sentiment_counts = pd.Series([n['sentiment'] for n in news_list]).value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("正面新闻", sentiment_counts.get('正面', 0))
    col2.metric("中性新闻", sentiment_counts.get('中性', 0))
    col3.metric("负面新闻", sentiment_counts.get('负面', 0))

    # 新闻列表
    for news in news_list:
        with st.expander(f"{news['title'][:50]}..."):
            st.write(f"来源：{news['source']} | 时间：{news['publish_date']}")
            st.write(f"情绪：{news['sentiment']}")
            if news['link']:
                st.markdown(f"[阅读原文]({news['link']})")

# -------------------- 主应用逻辑 --------------------
def main():
    """主应用流程"""
    st.set_page_config(page_title=CONFIG['page_title'], layout=CONFIG['layout'])

    # 侧边栏 - 股票代码 & 关注列表
    st.sidebar.title("🚀 智能股票分析")
    ticker = st.sidebar.text_input("输入股票代码（支持港股，如 0700.HK）", value="TSLA").upper()

    # 关注列表管理
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    st.sidebar.markdown("### ⭐ 关注列表")
    if st.sidebar.button("➕ 添加到关注列表"):
        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            st.sidebar.success("添加成功！")
        else:
            st.sidebar.warning("已在关注列表")

    # 渲染关注列表操作
    for i, wl_ticker in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(wl_ticker)
        if col2.button("❌", key=f"remove_{i}"):
            st.session_state.watchlist.remove(wl_ticker)
            st.experimental_rerun()  # 修复点击报错问题（替换为更稳定的刷新）

    # 侧边栏 - 导航
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📋 导航菜单", [
        "📊 实时数据", "📈 技术分析", 
        "🎯 投资建议", "🌟 热门股票", 
        "📰 市场新闻"
    ])

    # 渲染对应页面
    if page == "📊 实时数据":
        render_realtime_page(ticker)
    elif page == "📈 技术分析":
        render_technical_page(ticker)
    elif page == "🎯 投资建议":
        render_advice_page(ticker)
    elif page == "🌟 热门股票":
        render_trending_page()
    elif page == "📰 市场新闻":
        render_news_page(ticker)

if __name__ == "__main__":
    main()
