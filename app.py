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

# -------------------- 配置信息 --------------------
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",
        "alpha_vantage": "Z45S0SLJGM378PIO",
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
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

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """获取股票基本信息，适配港股代码（自动补全.HK后缀）"""
    try:
        # 港股代码处理：5位数字自动补全.HK（如00700 → 00700.HK）
        if ticker.isdigit() and len(ticker) == 5 and not ticker.endswith('.HK'):
            ticker = f"{ticker}.HK"
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 获取推荐数据（容错处理）
        try:
            recommendations = stock.recommendations_summary
            if recommendations is None or recommendations.empty:
                recommendations = pd.DataFrame()
        except:
            recommendations = pd.DataFrame()
            
        return info, recommendations
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        # 备选方案：Finnhub API
        try:
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}"
            response = requests.get(url, params={"token": CONFIG['api_keys']['finnhub']}, timeout=10)
            if response.status_code == 200:
                return response.json(), pd.DataFrame()
        except:
            return {}, pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str) -> pd.DataFrame:
    """获取历史数据，适配港股代码"""
    try:
        if ticker.isdigit() and len(ticker) == 5 and not ticker.endswith('.HK'):
            ticker = f"{ticker}.HK"
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist if not hist.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"获取历史数据失败 {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_news(ticker: str) -> List[Dict]:
    """使用Finnhub获取新闻，适配港股代码"""
    try:
        if ticker.isdigit() and len(ticker) == 5 and not ticker.endswith('.HK'):
            ticker = f"{ticker}.HK"
        
        # 时间范围：近30天
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
            news_items = response.json()
            news_list = []
            
            # 情感分析关键词
            positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy']
            negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell']
            
            for item in news_items:
                title = item.get('headline', '')
                title_lower = title.lower()
                
                sentiment = "正面" if any(kw in title_lower for kw in positive_keywords) else \
                           "负面" if any(kw in title_lower for kw in negative_keywords) else "中性"
                
                # 格式化时间
                try:
                    publish_date = datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
                except:
                    publish_date = "未知时间"
                
                news_list.append({
                    'title': title,
                    'link': item.get('url', ''),
                    'publish_date': publish_date,
                    'sentiment': sentiment,
                    'source': item.get('source', 'Unknown'),
                    'summary': item.get('summary', '')
                })
            
            return news_list
        else:
            logger.error(f"Finnhub新闻API失败，状态码: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取新闻失败 {ticker}: {e}")
        return []

# -------------------- 技术分析函数 --------------------
def calculate_rsi(close: pd.Series, period: int = 14) -> float:
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
    if len(close) < long:
        return 0.0, 0.0
    ema_short = close.ewm(span=short).mean()
    ema_long = close.ewm(span=long).mean()
    return (ema_short - ema_long).iloc[-1], (ema_short - ema_long).ewm(span=signal).mean().iloc[-1]

def calculate_bollinger_bands(close: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if len(close) < window:
        return pd.Series(), pd.Series(), pd.Series()
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    return rolling_mean + rolling_std * std_dev, rolling_mean, rolling_mean - rolling_std * std_dev

def calculate_support_resistance(close: pd.Series) -> Tuple[float, float]:
    if len(close) < 20:
        current_price = close.iloc[-1] if not close.empty else 0
        return current_price * 0.95, current_price * 1.05
    recent_data = close.tail(20)
    return recent_data.min(), recent_data.max()

# -------------------- AI分析函数 --------------------
@st.cache_data(ttl=600)
def get_sentiment(ticker: str) -> str:
    try:
        # X.ai API调用
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['api_keys']['xai']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"股票{ticker}市场情绪：正面、负面或中性？"}]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content'].strip()
            return "正面" if "正面" in result or "positive" in result.lower() else \
                   "负面" if "负面" in result or "negative" in result.lower() else "中性"
        else:
            # Finnhub备选
            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={CONFIG['api_keys']['finnhub']}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                score = res.json().get('companyNewsScore', 0.5)
                return "正面" if score > 0.6 else "负面" if score < 0.4 else "中性"
            return "中性（API错误）"
    except:
        return "中性（分析失败）"

@st.cache_data(ttl=600)
def get_investment_advice(ticker: str, rsi: float, macd: float) -> str:
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CONFIG['api_keys']['xai']}"}
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"基于RSI={rsi:.1f}、MACD={macd:.2f}，给{ticker}的50字内投资建议"}]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()['choices'][0]['message']['content'].strip() if response.status_code == 200 else \
               "RSI超卖可关注" if rsi < 30 else "RSI超买需谨慎" if rsi > 70 else "观望为主"
    except:
        return "RSI超卖可关注" if rsi < 30 else "RSI超买需谨慎" if rsi > 70 else "观望为主"

# -------------------- 热门股票函数 --------------------
@st.cache_data(ttl=3600)
def get_trending_stocks() -> pd.DataFrame:
    try:
        # Finnhub热门股票API
        url = "https://finnhub.io/api/v1/stock/most-active"
        params = {"token": CONFIG['api_keys']['finnhub']}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get('mostActiveStock', [])
            trending_data = []
            
            for item in data:
                ticker = item.get('symbol', '')
                info, _ = get_stock_info(ticker)
                if not info:
                    continue
                
                trending_data.append({
                    '股票代码': ticker,
                    '公司名称': info.get('longName', ticker),
                    '当前价格': info.get('currentPrice', 0),
                    '涨跌幅': info.get('regularMarketChangePercent', 0),
                    '成交量': info.get('volume', 0),
                    '市场情绪': get_sentiment(ticker)
                })
            
            return pd.DataFrame(trending_data) if trending_data else pd.DataFrame()
        else:
            # 静态热门列表（含港股）
            return pd.DataFrame([
                {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': 2.3, '成交量': 12345678, '市场情绪': '正面'},
                {'股票代码': 'AAPL', '公司名称': '苹果', '当前价格': 180.2, '涨跌幅': 0.8, '成交量': 23456789, '市场情绪': '中性'},
                {'股票代码': '00700.HK', '公司名称': '腾讯控股', '当前价格': 300.0, '涨跌幅': 1.5, '成交量': 56789012, '市场情绪': '正面'},
                {'股票代码': 'BABA', '公司名称': '阿里巴巴', '当前价格': 80.3, '涨跌幅': -0.5, '成交量': 87654321, '市场情绪': '中性'}
            ])
    except:
        return pd.DataFrame([
            {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': 2.3, '成交量': 12345678, '市场情绪': '正面'},
            {'股票代码': '00700.HK', '公司名称': '腾讯控股', '当前价格': 300.0, '涨跌幅': 1.5, '成交量': 56789012, '市场情绪': '正面'}
        ])

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    info, _ = get_stock_info(ticker)
    if not info:
        st.error("❌ 无法获取股票数据，请检查代码（港股请用5位数字，如00700）")
        return
    
    company_name = info.get('longName', ticker)
    currency = info.get('currency', 'USD')
    
    st.title(f"📊 {company_name} ({ticker})")
    
    # 关键指标（容错处理，避免N/A报错）
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = info.get('currentPrice', 0)
    prev_close = info.get('previousClose', current_price)
    change = current_price - prev_close if prev_close != 0 else 0
    change_percent = (change / prev_close * 100) if prev_close != 0 else 0
    
    with col1:
        st.metric(
            "当前价格", 
            f"{current_price:.2f} {currency}" if current_price != 0 else "N/A",
            delta=f"{change:.2f} ({change_percent:+.2f}%)" if prev_close != 0 else "N/A"
        )
    
    with col2:
        day_high = info.get('dayHigh', 'N/A')
        st.metric("今日最高", f"{day_high:.2f} {currency}" if isinstance(day_high, (int, float)) else day_high)
    
    with col3:
        day_low = info.get('dayLow', 'N/A')
        st.metric("今日最低", f"{day_low:.2f} {currency}" if isinstance(day_low, (int, float)) else day_low)
    
    with col4:
        volume = info.get('volume', 'N/A')
        st.metric("成交量", f"{volume:,}" if isinstance(volume, (int, float)) else volume)
    
    # K线图
    st.markdown("---")
    period_options = {"1日": "1d", "5日": "5d", "1月": "1mo", "3月": "3mo", "1年": "1y", "5年": "5y"}
    selected_period = st.selectbox("选择时间范围", list(period_options.keys()), index=2)
    hist = get_historical_data(ticker, period_options[selected_period])
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
        return
    
    fig = go.Figure(go.Candlestick(
        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K线'
    ))
    
    # 均线
    if len(hist) >= 5:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(5).mean(), name='MA5', line=dict(color='blue')))
    if len(hist) >= 20:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(20).mean(), name='MA20', line=dict(color='orange')))
        upper, mid, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(go.Scatter(x=hist.index, y=upper, name='布林上轨', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=lower, name='布林下轨', line=dict(color='green', dash='dash')))
    
    fig.update_layout(title=f"{ticker} K线图", height=500, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # 盘前盘后数据（仅美股）
    if currency == 'USD':
        st.markdown("### 📈 盘前/盘后交易")
        col1, col2 = st.columns(2)
        with col1:
            pre_price = info.get('preMarketPrice')
            st.metric("盘前价格", f"{pre_price:.2f} {currency}" if pre_price else "暂无数据")
        with col2:
            post_price = info.get('postMarketPrice')
            st.metric("盘后价格", f"{post_price:.2f} {currency}" if post_price else "暂无数据")

def render_technical_page(ticker: str):
    hist = get_historical_data(ticker, "1y")
    info = get_stock_info(ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据获取失败")
        return
    
    st.title(f"📈 {ticker} 技术分析")
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])
    support, resistance = calculate_support_resistance(hist['Close'])
    
    # 指标卡片
    col1, col2 = st.columns(2)
    col1.metric("RSI(14)", f"{rsi:.2f}", "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常")
    col2.metric("MACD", f"{macd:.2f} / {signal:.2f}", "看涨" if macd > signal else "看跌")
    
    # 技术指标表格
    tech_data = {
        "指标": ["支撑位", "阻力位", "RSI状态", "MACD状态"],
        "数值/描述": [
            f"{support:.2f}", f"{resistance:.2f}",
            "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常",
            "看涨" if macd > signal else "看跌"
        ]
    }
    st.dataframe(pd.DataFrame(tech_data), hide_index=True)
    
    # RSI趋势图
    if len(hist) >= 14:
        fig = go.Figure(go.Scatter(x=hist.index, y=hist['Close'].rolling(14).apply(calculate_rsi), name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
        fig.update_layout(title="RSI趋势", height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_advice_page(ticker: str):
    hist = get_historical_data(ticker, "3mo")
    info = get_stock_info(ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据不足，无法生成建议")
        return
    
    rsi = calculate_rsi(hist['Close'])
    macd, _ = calculate_macd(hist['Close'])
    sentiment = get_sentiment(ticker)
    ai_advice = get_investment_advice(ticker, rsi, macd)
    
    st.title(f"🎯 {ticker} 投资建议")
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{rsi:.2f}")
    col2.metric("市场情绪", sentiment)
    col3.metric("AI建议", ai_advice[:10] + "..." if len(ai_advice) > 10 else ai_advice)
    
    # 评分与建议
    score = 0
    score += 2 if rsi < 30 else -2 if rsi > 70 else 0
    score += 1 if macd > 0 else -1
    score += 1 if sentiment == "正面" else -1 if sentiment == "负面" else 0
    
    recommendation = {
        score >= 2: "强烈买入",
        score == 1: "买入",
        score == 0: "持有",
        score == -1: "卖出",
        score <= -2: "强烈卖出"
    }[True]
    
    st.markdown(f"### 综合建议: **{recommendation}**")
    st.warning("⚠️ 投资有风险，建议仅供参考")

def render_trending_page():
    st.title("🌟 热门股票")
    if st.button("🔄 更新热门股票"):
        with st.spinner("加载中..."):
            st.session_state['trending'] = get_trending_stocks()
            st.success("更新完成！")
    
    if 'trending' in st.session_state and not st.session_state['trending'].empty:
        st.dataframe(
            st.session_state['trending'],
            hide_index=True,
            column_config={
                "涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
                "当前价格": st.column_config.NumberColumn(format="$%.2f")
            }
        )
    else:
        st.info("点击上方按钮获取热门股票")

def render_news_page(ticker: str):
    st.title(f"📰 {ticker} 新闻")
    news_list = get_news(ticker)
    
    if not news_list:
        st.warning("暂无相关新闻")
        return
    
    # 情绪统计
    sentiment_counts = pd.Series([n['sentiment'] for n in news_list]).value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("正面新闻", sentiment_counts.get('正面', 0))
    col2.metric("中性新闻", sentiment_counts.get('中性', 0))
    col3.metric("负面新闻", sentiment_counts.get('负面', 0))
    
    # 新闻列表
    for news in news_list:
        with st.expander(f"{news['title'][:60]}..."):
            st.write(f"**来源:** {news['source']} | **时间:** {news['publish_date']}")
            st.write(f"**情绪:** {news['sentiment']}")
            if news['summary']:
                st.write(f"**摘要:** {news['summary']}")
            if news['link']:
                st.link_button("阅读原文", news['link'])

# -------------------- 主应用 --------------------
def main():
    st.set_page_config(page_title=CONFIG['page_title'], layout='wide')
    st.sidebar.title("🚀 智能股票分析")
    st.sidebar.markdown("---")
    
    # 股票代码输入
    ticker = st.sidebar.text_input(
        "输入股票代码", 
        value="00700",  # 默认港股示例
        help="美股: TSLA | 港股: 00700（自动补全.HK）"
    ).upper()
    
    # 收藏列表管理
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    st.sidebar.markdown("### ⭐ 关注列表")
    if st.sidebar.button("➕ 添加到关注"):
        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            st.sidebar.success(f"已添加 {ticker}")
        else:
            st.sidebar.warning("已在关注列表")
    
    # 显示收藏列表
    if st.session_state.watchlist:
        for i, wl_ticker in enumerate(st.session_state.watchlist):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(wl_ticker)
            if col2.button("❌", key=f"del_{i}"):
                st.session_state.watchlist.remove(wl_ticker)
                st.rerun()
    else:
        st.sidebar.info("暂无关注股票")
    
    # 导航菜单
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📋 功能菜单", [
        "📊 实时数据", "📈 技术分析", 
        "🎯 投资建议", "🌟 热门股票", "📰 新闻"
    ])
    
    # 页面渲染
    if page == "📊 实时数据":
        render_realtime_page(ticker)
    elif page == "📈 技术分析":
        render_technical_page(ticker)
    elif page == "🎯 投资建议":
        render_advice_page(ticker)
    elif page == "🌟 热门股票":
        render_trending_page()
    elif page == "📰 新闻":
        render_news_page(ticker)

if __name__ == "__main__":
    main()
