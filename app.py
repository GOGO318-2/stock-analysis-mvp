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
import time

warnings.filterwarnings('ignore')

# -------------------- 配置信息 --------------------
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "finnhub": "ckq0dahr01qj3j9g4vrgckq0dahr01qj3j9g4vs0",
        "alpha_vantage": "Z45S0SLJGM378PIO",
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
    },
    'cache_timeout': 300,  # 5分钟缓存
    'news_api': {
        'url': 'https://finnhub.io/api/v1/company-news',
        'key': "ckq0dahr01qj3j9g4vrgckq0dahr01qj3j9g4vs0"
    }
}

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 港股代码处理函数 --------------------
def process_hk_ticker(ticker: str) -> str:
    """处理港股代码，将5位数字格式转为 .HK 后缀格式（如 00700 → 00700.HK）"""
    ticker = ticker.strip().upper()
    if ticker.isdigit() and len(ticker) == 5 and not ticker.endswith('.HK'):
        return f"{ticker}.HK"
    elif ticker.endswith('.HK') and len(ticker) == 8:
        return ticker
    return ticker

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """获取股票基本信息，适配港股代码（自动补全.HK后缀）"""
    try:
        processed_ticker = process_hk_ticker(ticker)
        
        # 尝试使用yfinance获取数据
        try:
            stock = yf.Ticker(processed_ticker)
            info = stock.info
            
            try:
                recommendations = stock.recommendations_summary
                if recommendations is None or recommendations.empty:
                    recommendations = pd.DataFrame()
            except:
                recommendations = pd.DataFrame()
                
            return info, recommendations
        except Exception as e:
            logger.warning(f"yfinance获取股票信息失败 {processed_ticker}: {e}")
            
        # yfinance失败时使用Finnhub作为备用
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={processed_ticker}"
        response = requests.get(url, params={"token": CONFIG['api_keys']['finnhub']}, timeout=10)
        if response.status_code == 200:
            info = response.json()
            # 获取实时报价
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={processed_ticker}"
            quote_response = requests.get(quote_url, params={"token": CONFIG['api_keys']['finnhub']}, timeout=10)
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                info['currentPrice'] = quote_data.get('c', 0)
                info['previousClose'] = quote_data.get('pc', 0)
                info['dayHigh'] = quote_data.get('h', 0)
                info['dayLow'] = quote_data.get('l', 0)
                info['volume'] = quote_data.get('v', 0)
            return info, pd.DataFrame()
        else:
            return {}, pd.DataFrame()
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        return {}, pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str) -> pd.DataFrame:
    """获取历史数据，适配港股代码"""
    try:
        processed_ticker = process_hk_ticker(ticker)
        
        # 尝试使用yfinance获取数据
        try:
            stock = yf.Ticker(processed_ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                return hist
        except Exception as e:
            logger.warning(f"yfinance获取历史数据失败 {processed_ticker}: {e}")
        
        # yfinance失败时使用Finnhub作为备用
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5年数据
        
        # 根据period调整时间范围
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "5d":
            start_date = end_date - timedelta(days=5)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': processed_ticker,
            'resolution': 'D',
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp()),
            'token': CONFIG['api_keys']['finnhub']
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data['s'] == 'ok':
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                df.set_index('Date', inplace=True)
                return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取历史数据失败 {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_news(ticker: str) -> List[Dict]:
    """使用Finnhub获取新闻，适配港股代码"""
    try:
        processed_ticker = process_hk_ticker(ticker)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        params = {
            'symbol': processed_ticker,
            'from': start_date,
            'to': end_date,
            'token': CONFIG['news_api']['key']
        }
        
        response = requests.get(CONFIG['news_api']['url'], params=params, timeout=10)
        if response.status_code == 200:
            news_items = response.json()
            news_list = []
            
            positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'strong', 'growth', 'beat', 'increase']
            negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'weak', 'decline', 'miss', 'decrease', 'cut']
            
            for item in news_items:
                title = item.get('headline', '')
                title_lower = title.lower()
                
                sentiment = "中性"
                if any(kw in title_lower for kw in positive_keywords):
                    sentiment = "正面"
                elif any(kw in title_lower for kw in negative_keywords):
                    sentiment = "负面"
                
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
    rs = gain.iloc[-1] / loss.iloc[-1]
    rsi = 100 - (100 / (1 + rs))
    return rsi if not pd.isna(rsi) else 50.0

def calculate_macd(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[float, float]:
    if len(close) < long:
        return 0.0, 0.0
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger_bands(close: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if len(close) < window:
        return pd.Series(), pd.Series(), pd.Series()
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

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
        # 使用Finnhub新闻情绪API
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={CONFIG['api_keys']['finnhub']}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            score = data.get('sentiment', {}).get('bullishPercent', 0.5)
            return "正面" if score > 0.6 else "负面" if score < 0.4 else "中性"
        return "中性"
    except:
        return "中性"

@st.cache_data(ttl=600)
def get_investment_advice(ticker: str, rsi: float, macd: float) -> str:
    try:
        if rsi < 30 and macd > 0:
            return "RSI超卖且MACD看涨，强烈建议买入"
        elif rsi < 30:
            return "RSI超卖，存在买入机会"
        elif rsi > 70 and macd < 0:
            return "RSI超买且MACD看跌，建议卖出"
        elif rsi > 70:
            return "RSI超买，需谨慎持有"
        elif macd > 0:
            return "MACD看涨，可考虑持有或加仓"
        elif macd < 0:
            return "MACD看跌，建议观望或减仓"
        else:
            return "技术指标中性，建议观望"
    except:
        return "RSI超卖可关注" if rsi < 30 else "RSI超买需谨慎" if rsi > 70 else "观望为主"

# -------------------- 热门股票函数 --------------------
@st.cache_data(ttl=3600)
def get_trending_stocks() -> pd.DataFrame:
    try:
        # 获取美股大盘指数成分股作为候选池
        url = "https://finnhub.io/api/v1/index/constituents?symbol=.SPX&token=" + CONFIG['api_keys']['finnhub']
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            constituents = response.json().get('constituents', [])[:50]
        else:
            # 备用股票池
            constituents = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'V', 
                           'PG', 'NVDA', 'HD', 'MA', 'DIS', 'ADBE', 'PYPL', 'NFLX', 'CRM', 
                           'INTC', 'CSCO', 'PEP', 'KO', 'T', 'VZ', 'WMT', 'MRK', 'PFE', 
                           'ABT', 'TMO', 'UNH', 'BAC', 'GS', 'MS', 'C', 'BA', 'CAT', 'MMM', 
                           'HON', 'GE', 'IBM', 'ORCL', 'QCOM', 'TXN', 'AMD', 'AVGO', 'AMAT', 
                           'MU', 'LRCX', 'ADI', 'XLNX']
        
        trending_data = []
        progress_bar = st.progress(0)
        total_stocks = len(constituents)
        
        for idx, ticker in enumerate(constituents):
            progress = (idx + 1) / total_stocks
            progress_bar.progress(progress)
            
            try:
                # 获取股票信息
                info, _ = get_stock_info(ticker)
                if not info or 'currentPrice' not in info:
                    continue
                
                # 获取历史数据
                hist = get_historical_data(ticker, "1y")
                if hist.empty:
                    continue
                
                # 计算技术指标
                rsi = calculate_rsi(hist['Close'])
                macd, _ = calculate_macd(hist['Close'])
                
                # 获取市场情绪
                sentiment = get_sentiment(ticker)
                
                # 计算推荐得分 (0-100)
                # RSI权重: 30%，MACD权重: 30%，情绪权重: 20%，价格动量权重: 20%
                score = 0
                
                # RSI评分：30以下满分，70以上0分
                if rsi < 30:
                    rsi_score = 100
                elif rsi > 70:
                    rsi_score = 0
                else:
                    rsi_score = 100 - ((rsi - 30) / 40 * 100)
                score += rsi_score * 0.3
                
                # MACD评分：正值加分，负值减分
                macd_score = 50 + (macd * 10)  # 每0.1的MACD值对应1分
                macd_score = max(0, min(100, macd_score))
                score += macd_score * 0.3
                
                # 情绪评分
                sentiment_score = 100 if sentiment == "正面" else 50 if sentiment == "中性" else 0
                score += sentiment_score * 0.2
                
                # 价格动量评分 (最近1个月涨幅)
                if len(hist) > 20:
                    monthly_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
                    momentum_score = min(100, max(0, 50 + monthly_return * 2))  # 每1%涨幅加2分
                    score += momentum_score * 0.2
                
                # 确保分数在0-100范围内
                score = max(0, min(100, score))
                
                trending_data.append({
                    '股票代码': ticker,
                    '公司名称': info.get('longName', ticker),
                    '当前价格': info.get('currentPrice', 0),
                    '涨跌幅': info.get('regularMarketChangePercent', 0),
                    'RSI': round(rsi, 2),
                    'MACD': round(macd, 4),
                    '市场情绪': sentiment,
                    '推荐得分': round(score),
                    '买入建议': "强烈买入" if score > 80 else "买入" if score > 60 else "观望" if score > 40 else "谨慎" if score > 20 else "卖出"
                })
            except Exception as e:
                logger.warning(f"处理股票 {ticker} 失败: {e}")
                continue
        
        # 按推荐得分降序排序
        df = pd.DataFrame(trending_data)
        if not df.empty:
            df = df.sort_values(by='推荐得分', ascending=False)
        return df
    except Exception as e:
        logger.error(f"获取热门股票失败: {e}")
        return pd.DataFrame([
            {'股票代码': 'AAPL', '公司名称': '苹果', '当前价格': 180.2, '涨跌幅': 0.8, 
             'RSI': 45.2, 'MACD': 0.12, '市场情绪': '正面', '推荐得分': 85, '买入建议': '强烈买入'},
            {'股票代码': 'MSFT', '公司名称': '微软', '当前价格': 340.5, '涨跌幅': 1.2, 
             'RSI': 38.7, 'MACD': 0.25, '市场情绪': '正面', '推荐得分': 82, '买入建议': '强烈买入'},
            {'股票代码': 'GOOGL', '公司名称': '谷歌', '当前价格': 138.2, '涨跌幅': -0.3, 
             'RSI': 52.1, 'MACD': -0.08, '市场情绪': '中性', '推荐得分': 65, '买入建议': '买入'},
            {'股票代码': 'AMZN', '公司名称': '亚马逊', '当前价格': 178.5, '涨跌幅': 2.1, 
             'RSI': 58.3, 'MACD': 0.15, '市场情绪': '正面', '推荐得分': 78, '买入建议': '买入'},
            {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': -1.5, 
             'RSI': 68.2, 'MACD': -0.12, '市场情绪': '中性', '推荐得分': 42, '买入建议': '观望'},
            {'股票代码': 'JPM', '公司名称': '摩根大通', '当前价格': 198.3, '涨跌幅': 0.7, 
             'RSI': 48.5, 'MACD': 0.08, '市场情绪': '正面', '推荐得分': 72, '买入建议': '买入'}
        ])

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    info, _ = get_stock_info(processed_ticker)
    if not info or 'currentPrice' not in info:
        st.error(f"❌ 无法获取股票数据，请检查代码（港股请用5位数字，如00700）")
        return
    
    company_name = info.get('longName', processed_ticker)
    currency = info.get('currency', 'USD')
    
    st.title(f"📊 {company_name} ({processed_ticker})")
    
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
    
    st.markdown("---")
    period_options = {"1日": "1d", "5日": "5d", "1月": "1mo", "3月": "3mo", "1年": "1y", "5年": "5y"}
    selected_period = st.selectbox("选择时间范围", list(period_options.keys()), index=2)
    hist = get_historical_data(processed_ticker, period_options[selected_period])
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
        return
    
    fig = go.Figure(go.Candlestick(
        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K线'
    ))
    
    if len(hist) >= 5:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(5).mean(), name='MA5', line=dict(color='blue')))
    if len(hist) >= 20:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(20).mean(), name='MA20', line=dict(color='orange')))
        upper, mid, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(go.Scatter(x=hist.index, y=upper, name='布林上轨', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=lower, name='布林下轨', line=dict(color='green', dash='dash')))
    
    fig.update_layout(title=f"{processed_ticker} K线图", height=500, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    
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
    processed_ticker = process_hk_ticker(ticker)
    hist = get_historical_data(processed_ticker, "1y")
    info = get_stock_info(processed_ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据获取失败")
        return
    
    st.title(f"📈 {processed_ticker} 技术分析")
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])
    support, resistance = calculate_support_resistance(hist['Close'])
    
    col1, col2 = st.columns(2)
    col1.metric("RSI(14)", f"{rsi:.2f}", "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常")
    col2.metric("MACD", f"{macd:.4f} / {signal:.4f}", "看涨" if macd > signal else "看跌")
    
    tech_data = {
        "指标": ["支撑位", "阻力位", "RSI状态", "MACD状态"],
        "数值/描述": [
            f"{support:.2f}", f"{resistance:.2f}",
            "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常",
            "看涨" if macd > signal else "看跌"
        ]
    }
    st.dataframe(pd.DataFrame(tech_data), hide_index=True)
    
    if len(hist) >= 14:
        # 计算RSI曲线
        rsi_values = []
        for i in range(14, len(hist)):
            rsi_values.append(calculate_rsi(hist['Close'].iloc[:i]))
        
        rsi_df = pd.DataFrame({
            'Date': hist.index[14:],
            'RSI': rsi_values
        }).set_index('Date')
        
        fig = go.Figure(go.Scatter(x=rsi_df.index, y=rsi_df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
        fig.update_layout(title="RSI趋势", height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_advice_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    hist = get_historical_data(processed_ticker, "3mo")
    info = get_stock_info(processed_ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据不足，无法生成建议")
        return
    
    rsi = calculate_rsi(hist['Close'])
    macd, _ = calculate_macd(hist['Close'])
    sentiment = get_sentiment(processed_ticker)
    ai_advice = get_investment_advice(processed_ticker, rsi, macd)
    
    st.title(f"🎯 {processed_ticker} 投资建议")
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{rsi:.2f}")
    col2.metric("市场情绪", sentiment)
    col3.metric("AI建议", ai_advice[:10] + "..." if len(ai_advice) > 10 else ai_advice)
    
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
    st.title("🌟 美股投资推荐")
    st.markdown("### 基于基本面与技术面的Top 50美股分析")
    st.info("评分标准：RSI(30%) + MACD(30%) + 市场情绪(20%) + 价格动量(20%)")
    
    if st.button("🔄 更新推荐列表"):
        with st.spinner("正在分析美股市场，可能需要1-2分钟..."):
            st.session_state['trending'] = get_trending_stocks()
            st.success("更新完成！")
    
    # 首次加载时初始化热门股票
    if 'trending' not in st.session_state:
        with st.spinner("首次加载美股推荐列表，请稍候..."):
            st.session_state['trending'] = get_trending_stocks()
    
    if not st.session_state['trending'].empty:
        # 添加颜色映射
        def color_score(val):
            color = 'green' if val > 80 else 'lightgreen' if val > 60 else 'gold' if val > 40 else 'orange' if val > 20 else 'red'
            return f'background-color: {color}'
        
        # 添加建议图标
        def advice_icon(advice):
            if "强烈买入" in advice:
                return "🚀"
            elif "买入" in advice:
                return "👍"
            elif "观望" in advice:
                return "👀"
            elif "谨慎" in advice:
                return "⚠️"
            else:
                return "👎"
        
        df = st.session_state['trending'].copy()
        df['建议'] = df['买入建议'].apply(advice_icon) + " " + df['买入建议']
        
        st.dataframe(
            df[['股票代码', '公司名称', '当前价格', '涨跌幅', 'RSI', 'MACD', '市场情绪', '推荐得分', '建议']],
            hide_index=True,
            column_config={
                "涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
                "当前价格": st.column_config.NumberColumn(format="$%.2f"),
                "推荐得分": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100
                )
            },
            height=800
        )
    else:
        st.info("暂无股票数据")

def render_news_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    st.title(f"📰 {processed_ticker} 新闻")
    news_list = get_news(processed_ticker)
    
    if not news_list:
        st.warning("暂无相关新闻")
        return
    
    sentiment_counts = pd.Series([n['sentiment'] for n in news_list]).value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("正面新闻", sentiment_counts.get('正面', 0))
    col2.metric("中性新闻", sentiment_counts.get('中性', 0))
    col3.metric("负面新闻", sentiment_counts.get('负面', 0))
    
    # 按情绪分组
    with st.expander("📈 新闻情绪分析", expanded=True):
        sentiment_df = pd.DataFrame({
            '情绪': ['正面', '中性', '负面'],
            '数量': [
                sentiment_counts.get('正面', 0),
                sentiment_counts.get('中性', 0),
                sentiment_counts.get('负面', 0)
            ]
        })
        st.bar_chart(sentiment_df.set_index('情绪'))
    
    # 显示新闻列表
    for news in news_list:
        sentiment_color = {
            "正面": "#d4f8d4",
            "中性": "#f0f0f0",
            "负面": "#f8d4d4"
        }.get(news['sentiment'], "#f0f0f0")
        
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: {sentiment_color};
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 5px solid {'green' if news['sentiment']=='正面' else 'gray' if news['sentiment']=='中性' else 'red'};
            ">
                <h4>{news['title']}</h4>
                <p><b>来源:</b> {news['source']} | <b>时间:</b> {news['publish_date']} | <b>情绪:</b> {news['sentiment']}</p>
                <p>{news['summary'][:200]}{'...' if len(news['summary']) > 200 else ''}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if news['link']:
                st.link_button("阅读原文", news['link'])
            st.markdown("---")

# -------------------- 主应用 --------------------
def main():
    st.set_page_config(page_title=CONFIG['page_title'], layout='wide')
    st.sidebar.title("🚀 智能股票分析")
    st.sidebar.markdown("---")
    
    # 使用会话状态跟踪当前选中的股票
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "00700"
    
    # 股票代码输入
    ticker = st.sidebar.text_input(
        "输入股票代码", 
        value=st.session_state.current_ticker,
        help="美股: AAPL | 港股: 00700（自动补全.HK）| A股: 600000.SS"
    ).upper()
    
    # 点击输入框时更新当前股票
    if ticker != st.session_state.current_ticker:
        st.session_state.current_ticker = ticker
    
    # 收藏列表管理
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["AAPL", "MSFT", "00700.HK", "TSLA"]
    
    st.sidebar.markdown("### ⭐ 关注列表")
    
    # 添加到关注按钮
    if st.sidebar.button("➕ 添加到关注"):
        processed_ticker = process_hk_ticker(ticker)
        if processed_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(processed_ticker)
            st.sidebar.success(f"已添加 {processed_ticker}")
        else:
            st.sidebar.warning("已在关注列表")
    
    # 显示收藏列表并添加点击事件
    if st.session_state.watchlist:
        for i, wl_ticker in enumerate(st.session_state.watchlist):
            col1, col2 = st.sidebar.columns([3, 1])
            
            # 使用按钮实现点击事件
            if col1.button(wl_ticker, key=f"wl_{i}"):
                st.session_state.current_ticker = wl_ticker
                st.experimental_rerun()
            
            # 删除按钮
            if col2.button("❌", key=f"del_{i}"):
                st.session_state.watchlist.remove(wl_ticker)
                st.rerun()
    else:
        st.sidebar.info("暂无关注股票")
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📋 功能菜单", [
        "📊 实时数据", "📈 技术分析", 
        "🎯 投资建议", "🌟 热门股票", "📰 新闻"
    ])
    
    # 使用会话状态中的当前股票进行查询
    active_ticker = st.session_state.current_ticker
    
    if page == "📊 实时数据":
        render_realtime_page(active_ticker)
    elif page == "📈 技术分析":
        render_technical_page(active_ticker)
    elif page == "🎯 投资建议":
        render_advice_page(active_ticker)
    elif page == "🌟 热门股票":
        render_trending_page()
    elif page == "📰 新闻":
        render_news_page(active_ticker)

if __name__ == "__main__":
    main()
