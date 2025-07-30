import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging

# -------------------- 配置信息 --------------------
CONFIG = {
    'page_title': '智能股票分析',
    'layout': 'wide',
    'api_keys': {
        "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",
        "yfinance": "default"  # yfinance 无需 API Key
    },
    'cache_timeout': 300,  # 5分钟缓存
    'news_api': {
        'url': 'https://finnhub.io/api/v1/company-news',
        'key': "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180"
    }
}

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 核心函数：港股代码处理 --------------------
def process_hk_ticker(ticker: str) -> str:
    """
    处理港股代码：
    - 5位数字自动补全 .HK 后缀（如 00700 → 00700.HK）
    - 已带后缀的保持不变（如 00700.HK → 00700.HK）
    """
    ticker = ticker.strip().upper()
    if ticker.isdigit() and len(ticker) == 5:
        return f"{ticker}.HK"
    return ticker

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> dict:
    """
    获取股票基本信息（适配港股）
    - 使用 yfinance，自动处理 .HK 后缀
    - 失败时返回空字典
    """
    try:
        ticker = process_hk_ticker(ticker)
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        return {}

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    获取历史数据（适配港股）
    - period 可选值：1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
    """
    try:
        ticker = process_hk_ticker(ticker)
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        logger.error(f"获取历史数据失败 {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_news(ticker: str) -> list:
    """
    获取股票新闻（适配港股）
    - 使用 Finnhub API，自动处理 .HK 后缀
    """
    try:
        ticker = process_hk_ticker(ticker)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        params = {
            'symbol': ticker,
            'from': start_date,
            'to': end_date,
            'token': CONFIG['news_api']['key']
        }
        
        response = requests.get(CONFIG['news_api']['url'], params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"新闻API失败 {ticker}: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取新闻失败 {ticker}: {e}")
        return []

# -------------------- 技术分析函数（保持不变） --------------------
def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    info = get_stock_info(ticker)
    if not info:
        st.error("❌ 无法获取股票数据，请检查代码（港股用 5 位数字，如 00700）")
        return
    
    st.title(f"{info.get('longName', ticker)} ({ticker})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前价格", f"{info.get('currentPrice', 0):.2f} {info.get('currency', 'USD')}")
    col2.metric("今日最高", f"{info.get('dayHigh', 0):.2f} {info.get('currency', 'USD')}")
    col3.metric("今日最低", f"{info.get('dayLow', 0):.2f} {info.get('currency', 'USD')}")
    col4.metric("成交量", f"{info.get('volume', 0):,}")
    
    period = st.selectbox("选择时间范围", ["1d", "5d", "1mo", "3mo", "1y"], index=2)
    hist = get_historical_data(ticker, period)
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
    else:
        fig = go.Figure(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']
        ))
        fig.update_layout(title=f"{ticker} K线图", height=500)
        st.plotly_chart(fig, use_container_width=True)

def render_news_page(ticker: str):
    news = get_news(ticker)
    st.title(f"{ticker} 新闻")
    
    if not news:
        st.info("暂无相关新闻")
        return
    
    for item in news[:5]:  # 显示前 5 条新闻
        st.write(f"### {item['headline']}")
        st.write(f"**来源:** {item['source']} | **时间:** {datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')}")
        st.write(f"[阅读原文]({item['url']})")
        st.markdown("---")

# -------------------- 主应用逻辑（修复收藏功能） --------------------
def main():
    st.set_page_config(page_title=CONFIG['page_title'], layout='wide')
    st.sidebar.title("智能股票分析")
    
    # 初始化会话状态
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "TSLA"  # 默认美股
    
    # 股票代码输入
    ticker_input = st.sidebar.text_input(
        "输入股票代码", 
        value=st.session_state.current_ticker,
        help="美股: TSLA | 港股: 00700（自动补全.HK）"
    )
    
    # 点击输入框时更新当前股票（解决收藏点击报错）
    if ticker_input != st.session_state.current_ticker:
        st.session_state.current_ticker = ticker_input
    
    # 关注列表管理
    st.sidebar.markdown("### ⭐ 关注列表")
    if st.sidebar.button("➕ 添加到关注"):
        processed_ticker = process_hk_ticker(ticker_input)
        if processed_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(processed_ticker)
            st.sidebar.success(f"已添加 {processed_ticker}")
    
    # 显示关注列表（支持点击切换）
    for i, wl_ticker in enumerate(st.session_state.watchlist):
        col1, col2 = st.sidebar.columns([3, 1])
        # 点击股票代码切换当前股票
        if col1.button(wl_ticker, key=f"wl_{i}"):
            st.session_state.current_ticker = wl_ticker
        # 删除按钮
        if col2.button("❌", key=f"del_{i}"):
            st.session_state.watchlist.remove(wl_ticker)
            st.experimental_rerun()  # 安全重启页面
    
    # 功能菜单
    page = st.sidebar.radio("功能菜单", ["实时数据", "新闻", "技术分析"])
    
    # 渲染页面（使用当前股票）
    active_ticker = st.session_state.current_ticker
    if page == "实时数据":
        render_realtime_page(active_ticker)
    elif page == "新闻":
        render_news_page(active_ticker)
    elif page == "技术分析":
        # 简单示例：显示 RSI
        hist = get_historical_data(active_ticker, "1mo")
        if not hist.empty:
            rsi = calculate_rsi(hist['Close'])
            st.title(f"{active_ticker} 技术分析")
            st.metric("RSI(14)", f"{rsi:.2f}")
        else:
            st.error("❌ 无法获取技术分析数据")

if __name__ == "__main__":
    main()
