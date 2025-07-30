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

# 配置信息（更新API配置）
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",  # 原有API
        "alpha_vantage": "Z45S0SLJGM378PIO",  # 原有API
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",  # 新增热门股票API
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"  # 原有AI API
    },
    'cache_timeout': 300,  # 5分钟缓存
    'news_api': {
        'url': 'https://newsapi.org/v2/everything',
        'key': 'b5c3e5a5e6f34f34b4b3b6e4e6f3e5a'  # 新增免费新闻API（需自行申请）
    }
}

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """获取股票基本信息和推荐（修复港股兼容问题）"""
    try:
        # 处理港股代码格式
        if '.' in ticker and not ticker.endswith('.HK'):
            ticker = ticker.replace('.', '-')  # 某些API需要这种格式
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        try:
            recommendations = stock.recommendations_summary
            if recommendations is None or recommendations.empty:
                recommendations = pd.DataFrame()
        except:
            recommendations = pd.DataFrame()
            
        return info, recommendations
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        # 增加重试机制
        try:
            # 尝试使用finnhub API作为备选
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}"
            response = requests.get(url, params={"token": CONFIG['api_keys']['finnhub']})
            if response.status_code == 200:
                return response.json(), pd.DataFrame()
        except:
            return {}, pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str) -> pd.DataFrame:
    """获取历史数据（修复港股兼容问题）"""
    try:
        # 处理港股代码格式
        if '.' in ticker and not ticker.endswith('.HK'):
            ticker = ticker.replace('.', '-')
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist if not hist.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"获取历史数据失败 {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_news(ticker: str) -> List[Dict]:
    """获取股票新闻（改用NewsAPI作为备选）"""
    try:
        # 尝试使用yfinance获取新闻
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        
        news_list = []
        positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'growth', 'strong']
        negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'decline', 'weak']
        
        for item in news:
            title = item.get('title', '')
            title_lower = title.lower()
            
            sentiment = "正面" if any(kw in title_lower for kw in positive_keywords) else \
                        "负面" if any(kw in title_lower for kw in negative_keywords) else "中性"
            
            publish_time = item.get('providerPublishTime', 0)
            try:
                publish_date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
            except:
                publish_date = "未知时间"
            
            news_list.append({
                'title': title,
                'link': item.get('link', ''),
                'publish_date': publish_date,
                'sentiment': sentiment,
                'source': item.get('publisher', {}).get('name', 'Unknown')
            })
        
        return news_list
    except Exception as e:
        logger.error(f"获取新闻失败 {ticker}: {e}")
        # 备选方案：使用NewsAPI
        try:
            params = {
                'q': ticker,
                'apiKey': CONFIG['news_api']['key'],
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }
            response = requests.get(CONFIG['news_api']['url'], params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                news_list = []
                for art in articles:
                    news_list.append({
                        'title': art.get('title', ''),
                        'link': art.get('url', ''),
                        'publish_date': art.get('publishedAt', '')[:16].replace('T', ' '),
                        'source': art.get('source', {}).get('name', 'Unknown'),
                        'sentiment': '中性'  # 简化情感分析
                    })
                return news_list
        except:
            return []

# -------------------- 技术分析函数 --------------------
def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    """计算RSI指标"""
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
    """计算MACD指标"""
    if len(close) < long:
        return 0.0, 0.0
    
    ema_short = close.ewm(span=short).mean()
    ema_long = close.ewm(span=long).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal).mean()
    
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger_bands(close: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    if len(close) < window:
        return pd.Series(), pd.Series(), pd.Series()
    
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return upper_band, rolling_mean, lower_band

def calculate_support_resistance(close: pd.Series) -> Tuple[float, float]:
    """计算支撑位和阻力位"""
    if len(close) < 20:
        current_price = close.iloc[-1] if not close.empty else 0
        return current_price * 0.95, current_price * 1.05
    
    recent_data = close.tail(20)
    support = recent_data.min()
    resistance = recent_data.max()
    
    return support, resistance

# -------------------- AI分析函数（增强容错） --------------------
@st.cache_data(ttl=600)
def get_sentiment(ticker: str) -> str:
    """获取情感分析（增强容错）"""
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['api_keys']['xai']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-beta",
            "messages": [{
                "role": "user", 
                "content": f"分析股票 {ticker} 当前市场情绪，回答：正面、负面或中性"
            }],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content'].strip()
            return "正面" if any(word in result for word in ['正面', 'positive', '看涨', '乐观']) else \
                   "负面" if any(word in result for word in ['负面', 'negative', '看跌', '悲观']) else "中性"
        else:
            # 备选方案：使用简单的价格变动判断
            hist = get_historical_data(ticker, "1mo")
            if len(hist) > 5:
                recent_change = hist['Close'].pct_change().dropna().mean()
                return "正面" if recent_change > 0.01 else "负面" if recent_change < -0.01 else "中性"
            return "中性（API错误）"
    except Exception as e:
        logger.error(f"AI情感分析失败 {ticker}: {e}")
        # 备选方案：使用简单的价格变动判断
        hist = get_historical_data(ticker, "1mo")
        if len(hist) > 5:
            recent_change = hist['Close'].pct_change().dropna().mean()
            return "正面" if recent_change > 0.01 else "负面" if recent_change < -0.01 else "中性"
        return "中性（分析失败）"

@st.cache_data(ttl=600)
def get_investment_advice(ticker: str, rsi: float, macd: float) -> str:
    """获取投资建议（增强容错）"""
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['api_keys']['xai']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-beta",
            "messages": [{
                "role": "user",
                "content": f"基于技术指标：RSI={rsi:.1f}, MACD={macd:.2f}，为股票{ticker}提供简短投资建议（50字内）"
            }],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            # 备选方案：基于RSI和MACD生成简单建议
            advice = []
            if rsi < 30: advice.append("RSI超卖，短期可能反弹")
            elif rsi > 70: advice.append("RSI超买，短期可能回调")
            
            if macd > 0: advice.append("MACD为正，趋势向上")
            else: advice.append("MACD为负，趋势向下")
            
            return "; ".join(advice) if advice else "暂无明确信号，建议观望"
    except Exception as e:
        logger.error(f"AI投资建议失败 {ticker}: {e}")
        # 备选方案：基于RSI和MACD生成简单建议
        advice = []
        if rsi < 30: advice.append("RSI超卖，短期可能反弹")
        elif rsi > 70: advice.append("RSI超买，短期可能回调")
        
        if macd > 0: advice.append("MACD为正，趋势向上")
        else: advice.append("MACD为负，趋势向下")
        
        return "; ".join(advice) if advice else "暂无明确信号，建议观望"

# -------------------- 热门股票动态推荐（使用Polygon API） --------------------
@st.cache_data(ttl=3600)  # 每小时更新
def get_trending_stocks() -> pd.DataFrame:
    """动态获取热门股票（使用Polygon API）"""
    try:
        # 获取美股热门股票
        url = "https://api.polygon.io/v2/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "sort": "volume",
            "order": "desc",
            "limit": 20,  # 获取前20只交易量最大的股票
            "apiKey": CONFIG['api_keys']['polygon']
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json().get('tickers', [])
            trending_data = []
            
            for item in data:
                ticker = item.get('ticker', '')
                if not ticker:
                    continue
                
                # 获取股票详情
                info, _ = get_stock_info(ticker)
                if not info:
                    continue
                
                trending_data.append({
                    '股票代码': ticker,
                    '公司名称': info.get('longName', ticker),
                    '当前价格': info.get('currentPrice', 0),
                    '涨跌幅': info.get('regularMarketChangePercent', 0),
                    '成交量': info.get('volume', 0),
                    '市值': info.get('marketCap', 0)
                })
            
            if trending_data:
                return pd.DataFrame(trending_data)
        
        # 如果API失败，使用备用列表
        return pd.DataFrame([
            {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': 2.3, '成交量': 12345678, '市值': 750000000000},
            {'股票代码': 'AAPL', '公司名称': '苹果', '当前价格': 180.2, '涨跌幅': 0.8, '成交量': 23456789, '市值': 2800000000000},
            # 其他备用股票...
        ])
    except Exception as e:
        logger.error(f"获取热门股票失败: {e}")
        # 如果API失败，使用备用列表
        return pd.DataFrame([
            {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': 2.3, '成交量': 12345678, '市值': 750000000000},
            {'股票代码': 'AAPL', '公司名称': '苹果', '当前价格': 180.2, '涨跌幅': 0.8, '成交量': 23456789, '市值': 2800000000000},
            # 其他备用股票...
        ])

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    """渲染实时数据页面"""
    info, _ = get_stock_info(ticker)
    
    if not info:
        st.error("❌ 无法获取股票数据，请检查股票代码")
        return
    
    company_name = info.get('longName', ticker)
    currency = info.get('currency', 'USD')
    
    st.title(f"📊 {company_name} ({ticker})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = info.get('currentPrice', 0)
    previous_close = info.get('previousClose', current_price)
    
    change = current_price - previous_close
    change_percent = (change / previous_close * 100) if previous_close != 0 else 0.0
    
    with col1:
        st.metric(
            "当前价格", 
            f"{current_price:.2f} {currency}",
            delta=f"{change:.2f} ({change_percent:+.2f}%)"
        )
    
    with col2:
        st.metric("今日最高", f"{info.get('dayHigh', 'N/A'):.2f} {currency}")
    
    with col3:
        st.metric("今日最低", f"{info.get('dayLow', 'N/A'):.2f} {currency}")
    
    with col4:
        st.metric("成交量", f"{info.get('volume', 0):,.0f}")
    
    st.markdown("---")
    
    period_options = {
        "1日": "1d", "5日": "5d", "1月": "1mo", 
        "3月": "3mo", "1年": "1y", "5年": "5y"
    }
    
    selected_period = st.selectbox(
        "选择时间范围",
        list(period_options.keys()),
        index=2
    )
    
    hist = get_historical_data(ticker, period_options[selected_period])
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='K线'
    ))
    
    if len(hist) >= 5:
        ma5 = hist['Close'].rolling(window=5).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma5, 
            mode='lines', name='MA5',
            line=dict(color='blue', width=1)
        ))
    
    if len(hist) >= 20:
        ma20 = hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma20,
            mode='lines', name='MA20',
            line=dict(color='orange', width=1)
        ))
        
        upper, middle, lower = calculate_bollinger_bands(hist['Close'])
        if not upper.empty:
            fig.add_trace(go.Scatter(
                x=hist.index, y=upper,
                mode='lines', name='布林上轨',
                line=dict(color='red', dash='dash', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=hist.index, y=lower,
                mode='lines', name='布林下轨',
                line=dict(color='green', dash='dash', width=1)
            ))
    
    fig.update_layout(
        title=f"{ticker} K线图表",
        xaxis_title="时间",
        yaxis_title="价格",
        xaxis_rangeslider_visible=True,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if currency == 'USD':
        st.markdown("### 📈 盘前/盘后交易")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("盘前交易")
            pre_price = info.get('preMarketPrice')
            pre_change = info.get('preMarketChange', 0)
            
            if pre_price:
                st.metric(
                    "盘前价格",
                    f"{pre_price:.2f} {currency}",
                    delta=f"{pre_change:.2f}"
                )
            else:
                st.info("暂无盘前数据")
        
        with col2:
            st.subheader("盘后交易")
            post_price = info.get('postMarketPrice')
            post_change = info.get('postMarketChange', 0)
            
            if post_price:
                st.metric(
                    "盘后价格",
                    f"{post_price:.2f} {currency}",
                    delta=f"{post_change:.2f}"
                )
            else:
                st.info("暂无盘后数据")

def render_technical_page(ticker: str):
    """渲染技术分析页面"""
    st.title(f"📈 {ticker} 技术分析")
    
    info, _ = get_stock_info(ticker)
    hist = get_historical_data(ticker, "1y")
    
    if hist.empty:
        st.error("❌ 无法获取历史数据")
        return
    
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])
    support, resistance = calculate_support_resistance(hist['Close'])
    
    returns = hist['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else 0.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty and returns.std() != 0 else 0.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 技术指标")
        technical_data = {
            "指标": ["RSI (14日)", "MACD", "信号线", "支撑位", "阻力位"],
            "数值": [
                f"{rsi:.1f}",
                f"{macd:.3f}",
                f"{signal:.3f}",
                f"{support:.2f}",
                f"{resistance:.2f}"
            ],
            "状态": [
                "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常",
                "看涨" if macd > signal else "看跌",
                "-",
                "强支撑" if support > hist['Close'].iloc[-1] * 0.9 else "弱支撑",
                "强阻力" if resistance < hist['Close'].iloc[-1] * 1.1 else "弱阻力"
            ]
        }
        st.dataframe(pd.DataFrame(technical_data), hide_index=True)
    
    with col2:
        st.subheader("📊 风险指标")
        risk_data = {
            "指标": ["年化波动率", "夏普比率", "市盈率", "市净率", "Beta系数"],
            "数值": [
                f"{volatility:.1f}%",
                f"{sharpe:.2f}",
                f"{info.get('trailingPE', 'N/A')}",
                f"{info.get('priceToBook', 'N/A')}",
                f"{info.get('beta', 'N/A')}"
            ]
        }
        st.dataframe(pd.DataFrame(risk_data), hide_index=True)
    
    st.subheader("📈 RSI 趋势")
    if len(hist) >= 14:
        rsi_values = []
        for i in range(14, len(hist)):
            rsi_val = calculate_rsi(hist['Close'].iloc[:i+1])
            rsi_values.append(rsi_val)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index[14:],
            y=rsi_values,
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
        fig.update_layout(
            title="RSI指标趋势", 
            yaxis_title="RSI值", 
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_advice_page(ticker: str):
    """渲染投资建议页面"""
    st.title(f"🎯 {ticker} 投资建议")
    
    info, _ = get_stock_info(ticker)
    hist = get_historical_data(ticker, "3mo")
    
    if hist.empty:
        st.error("❌ 无法获取数据生成建议")
        return
    
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])
    current_price = info.get('currentPrice', hist['Close'].iloc[-1])
    support, resistance = calculate_support_resistance(hist['Close'])
    
    with st.spinner("🤖 AI分析中..."):
        sentiment = get_sentiment(ticker)
        ai_advice = get_investment_advice(ticker, rsi, macd)
    
    score = 0
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    
    if macd > signal: score += 1
    else: score -= 1
    
    if sentiment == "正面": score += 1
    elif sentiment == "负面": score -= 1
    
    if score >= 2:
        recommendation = "强烈买入"
        color = "green"
    elif score == 1:
        recommendation = "买入"
        color = "lightgreen"
    elif score == 0:
        recommendation = "持有"
        color = "yellow"
    elif score == -1:
        recommendation = "卖出"
        color = "orange"
    else:
        recommendation = "强烈卖出"
        color = "red"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("综合评分", f"{score}/5")
    
    with col2:
        st.markdown(f"### 投资建议: <span style='color:{color}'>{recommendation}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("市场情绪", sentiment)
    
    st.markdown("---")
    st.subheader("📋 详细建议")
    
    advice_data = {
        "策略类型": ["短线交易", "中线持有", "长线投资"],
        "建议操作": [
            "买入" if rsi < 40 else "卖出" if rsi > 60 else "观望",
            "持有" if -1 <= score <= 1 else "调仓",
            recommendation
        ],
        "目标价位": [
            f"{current_price * 1.05:.2f}",
            f"{current_price * 1.15:.2f}", 
            f"{current_price * 1.3:.2f}"
        ],
        "止损位": [
            f"{support:.2f}",
            f"{support * 0.95:.2f}",
            f"{support * 0.9:.2f}"
        ],
        "持仓建议": ["20-30%", "30-50%", "50-70%"]
    }
    
    st.dataframe(pd.DataFrame(advice_data), hide_index=True)
    
    st.markdown("---")
    st.subheader("🤖 AI 深度分析")
    st.info(ai_advice)
    
    st.markdown("---")
    st.warning("⚠️ 风险提示：以上建议仅供参考，投资有风险，入市需谨慎！")

def render_trending_page():
    """渲染热门股票页面"""
    st.title("🌟 热门科技股推荐")
    
    if st.button("🔄 更新数据", type="primary"):
        with st.spinner("正在获取最新数据..."):
            trending_data = get_trending_stocks()
            st.session_state['trending_data'] = trending_data
            st.success("数据更新完成！")
    
    if 'trending_data' in st.session_state:
        df = st.session_state['trending_data']
        
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.slider("最低价格", 0, 1000, 0)
        with col2:
            sentiment_filter = st.selectbox("情绪筛选", ["全部", "正面", "中性", "负面"])
        
        filtered_df = df[df['当前价格'] >= min_price]
        if sentiment_filter != "全部":
            filtered_df = filtered_df[filtered_df['市场情绪'] == sentiment_filter]
        
        st.dataframe(
            filtered_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "涨跌幅": st.column_config.NumberColumn(
                    "涨跌幅",
                    format="%.2f%%"
                ),
                "当前价格": st.column_config.NumberColumn(
                    "当前价格",
                    format="$%.2f"
                )
            }
        )
    else:
        st.info("点击'更新数据'获取最新热门股票信息")

def render_news_page(ticker: str):
    """渲染市场新闻页面"""
    st.title(f"📰 {ticker} 市场新闻")
    
    with st.spinner("获取最新新闻..."):
        news_list = get_news(ticker)
    
    if not news_list:
        st.warning("暂无相关新闻")
        return
    
    sentiments = [news['sentiment'] for news in news_list]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("正面新闻", sentiment_counts.get('正面', 0))
    with col2:
        st.metric("中性新闻", sentiment_counts.get('中性', 0))
    with col3:
        st.metric("负面新闻", sentiment_counts.get('负面', 0))
    
    st.markdown("---")
    
    for i, news in enumerate(news_list):
        with st.expander(f"📄 {news['title'][:80]}{'...' if len(news['title']) > 80 else ''}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**来源:** {news['source']}")
                st.write(f"**时间:** {news['publish_date']}")
            
            with col2:
                sentiment_color = {"正面": "🟢", "中性": "🟡", "负面": "🔴"}
                st.write(f"**情绪:** {sentiment_color.get(news['sentiment'], '⚪')} {news['sentiment']}")
            
            with col3:
                if news['link']:
                    st.link_button("阅读原文", news['link'])
            
            st.markdown("---")

# -------------------- 主应用 --------------------
def main():
    """主应用函数"""
    # 设置侧边栏
    st.sidebar.title("🚀 智能股票分析")
    st.sidebar.markdown("---")
    
    ticker_input = st.sidebar.text_input(
        "输入股票代码", 
        value="TSLA", 
        help="例如: TSLA (美股) 或 0700.HK (港股)"
    ).upper()
    
    # 处理港股代码格式
    ticker = ticker_input
    if '.' in ticker and not ticker.endswith('.HK'):
        ticker = ticker.replace('.', '-')
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    st.sidebar.markdown("### ⭐ 关注列表")
    
    if st.sidebar.button("➕ 添加到关注列表"):
        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            st.sidebar.success("添加成功！")
        else:
            st.sidebar.warning("已在关注列表中")
    
    if not st.session_state.watchlist:
        st.sidebar.info("暂无关注股票")
    else:
        for i, wl_ticker in enumerate(st.session_state.watchlist):
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            col1.text(wl_ticker)
            
            if col2.button("📊", key=f"view_{i}", help="查看"):
                ticker = wl_ticker
                st.experimental_rerun()
            
            if col3.button("🗑️", key=f"remove_{i}", help="移除"):
                st.session_state.watchlist.remove(wl_ticker)
                st.experimental_rerun()
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📋 导航菜单", 
        ["📊 实时数据", "📈 技术分析", "🎯 投资建议", "🌟 热门股票", "📰 市场新闻"]
    )
    
    # 渲染选择的页面
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
