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

# -------------------- 关键修改：港股代码处理函数 --------------------
def process_hk_ticker(ticker: str) -> str:
    """处理港股代码，将5位数字格式转为 .HK 后缀格式（如 00700 → 00700.HK）"""
    if ticker.isdigit() and len(ticker) == 5 and not ticker.endswith('.HK'):
        return f"{ticker}.HK"
    return ticker.upper()  # 其他情况保持原格式（如美股、已带后缀的港股）

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """获取股票基本信息，适配港股代码（自动补全.HK后缀）"""
    try:
        ticker = process_hk_ticker(ticker)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 如果yfinance返回空数据，尝试使用备用API
        if not info or info.get('regularMarketPrice') is None:
            logger.warning(f"yfinance数据为空，尝试备用API: {ticker}")
            try:
                url = f"https://finnhub.io/api/v1/stock/profile2"
                params = {"symbol": ticker, "token": CONFIG['api_keys']['finnhub']}
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    backup_info = response.json()
                    if backup_info:
                        info = backup_info
            except Exception as e:
                logger.error(f"备用API也失败: {e}")
        
        try:
            recommendations = stock.recommendations_summary
            if recommendations is None or recommendations.empty:
                recommendations = pd.DataFrame()
        except:
            recommendations = pd.DataFrame()
            
        return info, recommendations
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        return {}, pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str) -> pd.DataFrame:
    """获取历史数据，适配港股代码"""
    try:
        ticker = process_hk_ticker(ticker)
        
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
            news_items = response.json()
            news_list = []
            
            positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'growth', 'profit']
            negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'decline', 'fall']
            
            for item in news_items:
                title = item.get('headline', '')
                title_lower = title.lower()
                
                sentiment = "正面" if any(kw in title_lower for kw in positive_keywords) else \
                           "负面" if any(kw in title_lower for kw in negative_keywords) else "中性"
                
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
    upper_band = rolling_mean + rolling_std * std_dev
    lower_band = rolling_mean - rolling_std * std_dev
    return upper_band, rolling_mean, lower_band

def calculate_support_resistance(close: pd.Series) -> Tuple[float, float]:
    """计算支撑位和阻力位"""
    if len(close) < 20:
        current_price = close.iloc[-1] if not close.empty else 0
        return current_price * 0.95, current_price * 1.05
    recent_data = close.tail(20)
    return recent_data.min(), recent_data.max()

# -------------------- AI分析函数 --------------------
@st.cache_data(ttl=600)
def get_sentiment(ticker: str) -> str:
    """获取市场情绪分析"""
    try:
        # 首先尝试使用XAI API
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['api_keys']['xai']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"分析股票{ticker}的市场情绪，回答：正面、负面或中性"}]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content'].strip()
            if "正面" in result or "positive" in result.lower():
                return "正面"
            elif "负面" in result or "negative" in result.lower():
                return "负面"
            else:
                return "中性"
        else:
            logger.warning(f"XAI API失败，状态码: {response.status_code}")
    except Exception as e:
        logger.error(f"XAI API错误: {e}")
    
    # 备用方案：使用Finnhub情绪分析
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment"
        params = {"symbol": ticker, "token": CONFIG['api_keys']['finnhub']}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            score = data.get('companyNewsScore', 0.5)
            if score > 0.6:
                return "正面"
            elif score < 0.4:
                return "负面"
            else:
                return "中性"
    except Exception as e:
        logger.error(f"Finnhub情绪API错误: {e}")
    
    return "中性"

@st.cache_data(ttl=600)
def get_investment_advice(ticker: str, rsi: float, macd: float) -> str:
    """获取投资建议"""
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['api_keys']['xai']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": f"基于RSI={rsi:.1f}、MACD={macd:.2f}，给股票{ticker}的50字内投资建议"}]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"AI建议API错误: {e}")
    
    # 备用逻辑建议
    if rsi < 30:
        return "RSI显示超卖状态，可考虑逢低买入"
    elif rsi > 70:
        return "RSI显示超买状态，建议谨慎操作"
    elif macd > 0:
        return "MACD呈现上涨趋势，可适度关注"
    else:
        return "技术指标中性，建议观望为主"

# -------------------- 热门股票函数 --------------------
@st.cache_data(ttl=3600)
def get_trending_stocks() -> pd.DataFrame:
    """获取热门股票列表"""
    try:
        url = "https://finnhub.io/api/v1/stock/most-active"
        params = {"token": CONFIG['api_keys']['finnhub']}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get('mostActiveStock', [])
            trending_data = []
            
            for item in data[:10]:  # 限制数量
                ticker = item.get('symbol', '')
                if not ticker:
                    continue
                    
                try:
                    info, _ = get_stock_info(ticker)
                    if not info:
                        continue
                    
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    change_percent = info.get('regularMarketChangePercent', 0)
                    volume = info.get('volume', 0)
                    
                    trending_data.append({
                        '股票代码': ticker,
                        '公司名称': info.get('longName', ticker),
                        '当前价格': current_price,
                        '涨跌幅': change_percent,
                        '成交量': volume,
                        '市场情绪': get_sentiment(ticker)
                    })
                except Exception as e:
                    logger.error(f"处理股票{ticker}数据时出错: {e}")
                    continue
            
            return pd.DataFrame(trending_data) if trending_data else get_default_trending()
        else:
            logger.error(f"Finnhub热门股票API失败，状态码: {response.status_code}")
            return get_default_trending()
    except Exception as e:
        logger.error(f"获取热门股票失败: {e}")
        return get_default_trending()

def get_default_trending() -> pd.DataFrame:
    """返回默认的热门股票数据"""
    return pd.DataFrame([
        {'股票代码': 'AAPL', '公司名称': '苹果公司', '当前价格': 180.2, '涨跌幅': 0.8, '成交量': 23456789, '市场情绪': '中性'},
        {'股票代码': 'TSLA', '公司名称': '特斯拉', '当前价格': 240.5, '涨跌幅': 2.3, '成交量': 12345678, '市场情绪': '正面'},
        {'股票代码': 'MSFT', '公司名称': '微软', '当前价格': 420.1, '涨跌幅': 1.2, '成交量': 18765432, '市场情绪': '正面'},
        {'股票代码': '00700.HK', '公司名称': '腾讯控股', '当前价格': 300.0, '涨跌幅': 1.5, '成交量': 56789012, '市场情绪': '正面'},
        {'股票代码': 'BABA', '公司名称': '阿里巴巴', '当前价格': 80.3, '涨跌幅': -0.5, '成交量': 87654321, '市场情绪': '中性'}
    ])

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    """渲染实时数据页面"""
    info, _ = get_stock_info(ticker)
    if not info:
        st.error("❌ 无法获取股票数据，请检查股票代码")
        st.info("💡 港股请使用5位数字代码（如：00700），美股请使用标准代码（如：AAPL）")
        return
    
    company_name = info.get('longName') or info.get('name', ticker)
    currency = info.get('currency', 'USD')
    
    st.title(f"📊 {company_name} ({ticker})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose', current_price)
    change = current_price - prev_close if prev_close != 0 else 0
    change_percent = (change / prev_close * 100) if prev_close != 0 else 0
    
    with col1:
        st.metric(
            "当前价格", 
            f"{current_price:.2f} {currency}" if current_price != 0 else "N/A",
            delta=f"{change:.2f} ({change_percent:+.2f}%)" if prev_close != 0 else "N/A"
        )
    
    with col2:
        day_high = info.get('dayHigh') or info.get('regularMarketDayHigh', 'N/A')
        st.metric("今日最高", f"{day_high:.2f} {currency}" if isinstance(day_high, (int, float)) else day_high)
    
    with col3:
        day_low = info.get('dayLow') or info.get('regularMarketDayLow', 'N/A')
        st.metric("今日最低", f"{day_low:.2f} {currency}" if isinstance(day_low, (int, float)) else day_low)
    
    with col4:
        volume = info.get('volume') or info.get('regularMarketVolume', 'N/A')
        st.metric("成交量", f"{volume:,}" if isinstance(volume, (int, float)) else volume)
    
    st.markdown("---")
    period_options = {"1日": "1d", "5日": "5d", "1月": "1mo", "3月": "3mo", "1年": "1y", "5年": "5y"}
    selected_period = st.selectbox("选择时间范围", list(period_options.keys()), index=2)
    hist = get_historical_data(ticker, period_options[selected_period])
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
        return
    
    # 创建K线图
    fig = go.Figure(go.Candlestick(
        x=hist.index, 
        open=hist['Open'], 
        high=hist['High'], 
        low=hist['Low'], 
        close=hist['Close'], 
        name='K线'
    ))
    
    # 添加移动平均线
    if len(hist) >= 5:
        ma5 = hist['Close'].rolling(5).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma5, name='MA5', line=dict(color='blue', width=1)))
    
    if len(hist) >= 20:
        ma20 = hist['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma20, name='MA20', line=dict(color='orange', width=1)))
        
        # 添加布林带
        upper, mid, lower = calculate_bollinger_bands(hist['Close'])
        if not upper.empty:
            fig.add_trace(go.Scatter(x=hist.index, y=upper, name='布林上轨', line=dict(color='red', dash='dash', width=1)))
            fig.add_trace(go.Scatter(x=hist.index, y=lower, name='布林下轨', line=dict(color='green', dash='dash', width=1)))
    
    fig.update_layout(
        title=f"{ticker} K线图", 
        height=500, 
        xaxis_rangeslider_visible=True,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 盘前盘后数据（仅限美股）
    if currency == 'USD' and not ticker.endswith('.HK'):
        st.markdown("### 📈 盘前/盘后交易")
        col1, col2 = st.columns(2)
        with col1:
            pre_price = info.get('preMarketPrice')
            pre_change = info.get('preMarketChangePercent', 0)
            if pre_price:
                st.metric("盘前价格", f"{pre_price:.2f} {currency}", f"{pre_change:+.2f}%")
            else:
                st.metric("盘前价格", "暂无数据")
        with col2:
            post_price = info.get('postMarketPrice')
            post_change = info.get('postMarketChangePercent', 0)
            if post_price:
                st.metric("盘后价格", f"{post_price:.2f} {currency}", f"{post_change:+.2f}%")
            else:
                st.metric("盘后价格", "暂无数据")

def render_technical_page(ticker: str):
    """渲染技术分析页面"""
    hist = get_historical_data(ticker, "1y")
    info = get_stock_info(ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据获取失败，无法进行技术分析")
        return
    
    st.title(f"📈 {ticker} 技术分析")
    
    # 计算技术指标
    rsi = calculate_rsi(hist['Close'])
    macd, signal = calculate_macd(hist['Close'])
    support, resistance = calculate_support_resistance(hist['Close'])
    
    # 显示主要指标
    col1, col2, col3 = st.columns(3)
    with col1:
        rsi_status = "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常"
        st.metric("RSI(14)", f"{rsi:.2f}", rsi_status)
    
    with col2:
        macd_status = "看涨" if macd > signal else "看跌"
        st.metric("MACD", f"{macd:.3f}", macd_status)
    
    with col3:
        st.metric("支撑/阻力", f"{support:.2f} / {resistance:.2f}")
    
    # 技术指标表格
    tech_data = {
        "指标": ["支撑位", "阻力位", "RSI状态", "MACD状态", "趋势方向"],
        "数值/描述": [
            f"{support:.2f}",
            f"{resistance:.2f}",
            rsi_status,
            macd_status,
            "上升" if hist['Close'].iloc[-1] > hist['Close'].iloc[-20] else "下降"
        ]
    }
    st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)
    
    # RSI趋势图
    if len(hist) >= 14:
        st.markdown("### RSI趋势图")
        rsi_values = hist['Close'].rolling(14).apply(lambda x: calculate_rsi(x) if len(x) == 14 else np.nan)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=rsi_values, name='RSI', line=dict(color='blue')))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线(70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线(30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="中位线(50)")
        
        fig.update_layout(
            title="RSI(14)指标趋势", 
            height=300,
            yaxis=dict(range=[0, 100]),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

def render_advice_page(ticker: str):
    """渲染投资建议页面"""
    hist = get_historical_data(ticker, "3mo")
    info = get_stock_info(ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据不足，无法生成投资建议")
        return
    
    st.title(f"🎯 {ticker} 投资建议")
    
    # 计算分析指标
    rsi = calculate_rsi(hist['Close'])
    macd, signal_line = calculate_macd(hist['Close'])
    sentiment = get_sentiment(ticker)
    ai_advice = get_investment_advice(ticker, rsi, macd)
    
    # 显示关键指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RSI指标", f"{rsi:.2f}")
    with col2:
        st.metric("市场情绪", sentiment)
    with col3:
        st.metric("MACD", f"{macd:.3f}")
    
    # 综合评分系统
    score = 0
    score += 2 if rsi < 30 else -2 if rsi > 70 else 0  # RSI评分
    score += 1 if macd > signal_line else -1  # MACD评分
    score += 1 if sentiment == "正面" else -1 if sentiment == "负面" else 0  # 情绪评分
    
    # 价格趋势评分
    if len(hist) >= 20:
        current_price = hist['Close'].iloc[-1]
        avg_price = hist['Close'].tail(20).mean()
        score += 1 if current_price > avg_price else -1
    
    # 生成建议
    if score >= 3:
        recommendation = "强烈买入"
        color = "green"
    elif score >= 1:
        recommendation = "买入"
        color = "lightgreen"
    elif score >= -1:
        recommendation = "持有"
        color = "orange"
    elif score >= -3:
        recommendation = "卖出"
        color = "lightcoral"
    else:
        recommendation = "强烈卖出"
        color = "red"
    
    # 显示综合建议
    st.markdown("---")
    st.markdown(f"### 综合建议: <span style='color: {color}; font-weight: bold;'>{recommendation}</span>", unsafe_allow_html=True)
    st.markdown(f"**评分**: {score}/7")
    
    # AI建议
    st.markdown("### 🤖 AI分析建议")
    st.info(ai_advice)
    
    # 风险提示
    st.markdown("### ⚠️ 风险提示")
    st.warning("以上建议仅供参考，投资有风险，入市需谨慎。请根据自身情况做出投资决策。")
    
    # 详细分析
    with st.expander("📊 详细分析"):
        st.write("**技术指标分析:**")
        st.write(f"- RSI: {rsi:.2f} ({'超卖' if rsi < 30 else '超买' if rsi > 70 else '正常'})")
        st.write(f"- MACD: {macd:.3f} ({'看涨' if macd > signal_line else '看跌'})")
        st.write(f"- 市场情绪: {sentiment}")

def render_trending_page():
    """渲染热门股票页面"""
    st.title("🌟 热门股票")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        st.info("数据每小时自动更新一次")
    
    with st.spinner("正在获取热门股票数据..."):
        trending_df = get_trending_stocks()
    
    if not trending_df.empty:
        st.dataframe(
            trending_df,
            hide_index=True,
            column_config={
                "涨跌幅": st.column_config.NumberColumn(
                    format="%.2f%%",
                    help="当日涨跌幅度"
                ),
                "当前价格": st.column_config.NumberColumn(
                    format="$%.2f",
                    help="当前交易价格"
                ),
                "成交量": st.column_config.NumberColumn(
                    format="%d",
                    help="当日成交量"
                )
            },
            use_container_width=True
        )
        
        # 股票选择
        st.markdown("### 📈 选择股票进行详细分析")
        selected_stock = st.selectbox(
            "选择要分析的股票:",
            options=trending_df['股票代码'].tolist(),
            format_func=lambda x: f"{x} - {trending_df[trending_df['股票代码']==x]['公司名称'].iloc[0]}"
        )
        
        if st.button("分析选中股票", use_container_width=True):
            st.session_state.current_ticker = selected_stock
            st.session_state.page = "📊 实时数据"
            st.rerun()
    else:
        st.error("暂无法获取热门股票数据，请稍后再试")

def render_news_page(ticker: str):
    """渲染新闻页面"""
    st.title(f"📰 {ticker} 相关新闻")
    
    with st.spinner("正在获取最新新闻..."):
        news_list = get_news(ticker)
    
    if not news_list:
        st.warning("暂无相关新闻数据")
        st.info("💡 可能是因为:")
        st.write("- 该股票新闻较少")
        st.write("- API访问限制")
        st.write("- 网络连接问题")
        return
    
    # 新闻情绪统计
    sentiment_counts = pd.Series([n['sentiment'] for n in news_list]).value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总新闻数", len(news_list))
    with col2:
        st.metric("正面新闻", sentiment_counts.get('正面', 0))
    with col3:
        st.metric("中性新闻", sentiment_counts.get('中性', 0))
    with col4:
        st.metric("负面新闻", sentiment_counts.get('负面', 0))
    
    # 情绪分布图
    if len(sentiment_counts) > 0:
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3
        )])
        fig.update_layout(title="新闻情绪分布", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 新闻列表
    for i, news in enumerate(news_list):
        sentiment_color = {
            '正面': 'green',
            '负面': 'red',
            '中性': 'gray'
        }.get(news['sentiment'], 'gray')
        
        with st.expander(f"📄 {news['title'][:80]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**来源:** {news['source']}")
                st.write(f"**发布时间:** {news['publish_date']}")
                
            with col2:
                st.markdown(f"**情绪:** <span style='color: {sentiment_color}'>●</span> {news['sentiment']}", 
                           unsafe_allow_html=True)
            
            if news.get('summary'):
                st.write(f"**摘要:** {news['summary']}")
            
            if news.get('link'):
                st.link_button("🔗 阅读原文", news['link'], use_container_width=True)

# -------------------- 主应用 --------------------
def main():
    """主应用函数"""
    st.set_page_config(
        page_title=CONFIG['page_title'], 
        layout=CONFIG['layout'],
        initial_sidebar_state="expanded"
    )
    
    # 侧边栏
    st.sidebar.title("🚀 智能股票分析")
    st.sidebar.markdown("---")
    
    # 初始化会话状态
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "00700"
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'page' not in st.session_state:
        st.session_state.page = "📊 实时数据"
    
    # 股票代码输入
    ticker_input = st.sidebar.text_input(
        "📝 输入股票代码", 
        value=st.session_state.current_ticker,
        help="美股示例: AAPL, TSLA | 港股示例: 00700, 00941",
        placeholder="输入股票代码..."
    ).strip().upper()
    
    # 更新当前股票
    if ticker_input and ticker_input != st.session_state.current_ticker:
        st.session_state.current_ticker = ticker_input
        st.rerun()
    
    # 快速选择热门股票
    st.sidebar.markdown("### 🔥 快速选择")
    popular_stocks = {
        "苹果 (AAPL)": "AAPL",
        "特斯拉 (TSLA)": "TSLA", 
        "微软 (MSFT)": "MSFT",
        "腾讯 (00700)": "00700",
        "阿里巴巴 (BABA)": "BABA"
    }
    
    for name, code in popular_stocks.items():
        if st.sidebar.button(name, key=f"pop_{code}", use_container_width=True):
            st.session_state.current_ticker = code
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # 关注列表管理
    st.sidebar.markdown("### ⭐ 我的关注")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("➕ 添加", key="add_watchlist", use_container_width=True):
            processed_ticker = process_hk_ticker(st.session_state.current_ticker)
            if processed_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(processed_ticker)
                st.success(f"已添加 {processed_ticker}")
                st.rerun()
            else:
                st.warning("已在关注列表")
    
    with col2:
        if st.button("🗑️ 清空", key="clear_watchlist", use_container_width=True):
            st.session_state.watchlist = []
            st.success("已清空关注列表")
            st.rerun()
    
    # 显示关注列表
    if st.session_state.watchlist:
        for i, wl_ticker in enumerate(st.session_state.watchlist):
            col1, col2 = st.sidebar.columns([4, 1])
            
            with col1:
                if st.button(f"📊 {wl_ticker}", key=f"wl_{i}", use_container_width=True):
                    st.session_state.current_ticker = wl_ticker
                    st.rerun()
            
            with col2:
                if st.button("❌", key=f"del_{i}", use_container_width=True):
                    st.session_state.watchlist.remove(wl_ticker)
                    st.rerun()
    else:
        st.sidebar.info("💡 点击"添加"收藏当前股票")
    
    st.sidebar.markdown("---")
    
    # 页面导航
    pages = ["📊 实时数据", "📈 技术分析", "🎯 投资建议", "🌟 热门股票", "📰 新闻资讯"]
    selected_page = st.sidebar.radio("📋 功能导航", pages, index=pages.index(st.session_state.page))
    
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    # 当前股票显示
    current_ticker = st.session_state.current_ticker
    processed_ticker = process_hk_ticker(current_ticker)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"🎯 当前分析: **{processed_ticker}**")
    
    # 渲染对应页面
    try:
        if selected_page == "📊 实时数据":
            render_realtime_page(current_ticker)
        elif selected_page == "📈 技术分析":
            render_technical_page(current_ticker)
        elif selected_page == "🎯 投资建议":
            render_advice_page(current_ticker)
        elif selected_page == "🌟 热门股票":
            render_trending_page()
        elif selected_page == "📰 新闻资讯":
            render_news_page(current_ticker)
    except Exception as e:
        st.error(f"页面加载出错: {str(e)}")
        logger.error(f"页面渲染错误: {e}")
        st.info("请尝试刷新页面或切换到其他功能")
    
    # 页脚信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 帮助信息")
    with st.sidebar.expander("💡 使用说明"):
        st.write("""
        **股票代码格式:**
        - 美股: AAPL, TSLA, MSFT
        - 港股: 00700, 00941 (自动转换为 .HK)
        
        **功能说明:**
        - 📊 实时数据: K线图、价格走势
        - 📈 技术分析: RSI、MACD、布林带
        - 🎯 投资建议: AI智能分析
        - 🌟 热门股票: 市场活跃度排行
        - 📰 新闻资讯: 相关新闻与情绪分析
        """)
    
    st.sidebar.success("💼 数据来源: yfinance + Finnhub API")

if __name__ == "__main__":
    main()
