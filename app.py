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
import plotly.express as px
import os
import json
import random
from urllib.parse import quote

warnings.filterwarnings('ignore')

# -------------------- 配置信息 --------------------
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "finnhub": os.getenv("FINNHUB_API_KEY", "ckq0dahr01qj3j9g4vrgckq0dahr01qj3j9g4vs0"),
        "alpha_vantage": "Z45S0SLJGM378PIO",
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg"
    },
    'cache_timeout': 300,  # 5分钟缓存
    'news_api': {
        'url': 'https://finnhub.io/api/v1/company-news',
        'key': os.getenv("FINNHUB_API_KEY", "ckq0dahr01qj3j9g4vrgckq0dahr01qj3j9g4vs0")
    }
}

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 港股代码处理函数 --------------------
def process_hk_ticker(ticker: str) -> str:
    """处理港股代码，转换为正确的yfinance格式（如 00700 → 0700.HK）"""
    ticker = ticker.strip().upper()
    
    # 移除.HK后缀（如果有）
    if ticker.endswith('.HK'):
        ticker = ticker.replace('.HK', '')
    
    # 确保是数字代码
    if not ticker.isdigit():
        return ticker
    
    # 转换格式：保留4位有效数字，不足4位前面补0
    # 港股代码在yfinance中要求4位数字（如0700.HK）
    ticker = ticker.lstrip('0')
    if not ticker:  # 全为0的情况
        return "0000.HK"
    
    # 确保4位长度
    ticker = ticker.zfill(4)
    
    return f"{ticker}.HK"

def process_finnhub_ticker(ticker: str) -> str:
    """处理港股代码用于Finnhub API（如 00700 → 0700-HK）"""
    ticker = ticker.strip().upper()
    
    if ticker.endswith('.HK'):
        ticker = ticker.replace('.HK', '')
    
    if not ticker.isdigit():
        return ticker
    
    ticker = ticker.lstrip('0')
    if not ticker:
        return "0000.HK"
    
    ticker = ticker.zfill(4)
    return f"{ticker}-HK"

# -------------------- 数据获取函数 --------------------
@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_stock_info(ticker: str) -> Tuple[Dict, pd.DataFrame]:
    """获取股票基本信息，优化港股支持"""
    try:
        processed_ticker = process_hk_ticker(ticker)
        
        # 尝试使用yfinance获取数据
        try:
            stock = yf.Ticker(processed_ticker)
            info = stock.info
            
            # 检查是否获取到有效数据
            if not info or 'currentPrice' not in info:
                raise ValueError("yfinance返回空数据")
                
            return info, pd.DataFrame()
        except Exception as e:
            logger.warning(f"yfinance获取股票信息失败 {processed_ticker}: {e}")
            
        # yfinance失败时使用Finnhub作为备用（特别是港股）
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={processed_ticker}"
        response = requests.get(url, params={"token": CONFIG['api_keys']['finnhub']}, timeout=10)
        if response.status_code == 200:
            info = response.json()
            if not info:
                return {}, pd.DataFrame()
            
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
            
            # 获取公司名称
            if 'name' not in info:
                info['longName'] = processed_ticker
                
            return info, pd.DataFrame()
        else:
            return {}, pd.DataFrame()
    except Exception as e:
        logger.error(f"获取股票信息失败 {ticker}: {e}")
        return {}, pd.DataFrame()

@st.cache_data(ttl=CONFIG['cache_timeout'])
def get_historical_data(ticker: str, period: str) -> pd.DataFrame:
    """获取历史数据，优化港股支持"""
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
        
        # 根据period调整时间范围
        if period == "1d":
            days = 1
        elif period == "5d":
            days = 5
        elif period == "1mo":
            days = 30
        elif period == "3mo":
            days = 90
        elif period == "1y":
            days = 365
        else:  # 5y
            days = 365 * 5
            
        start_date = end_date - timedelta(days=days)
        
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
            if data.get('s') == 'ok' and 't' in data:
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

# -------------------- 新闻获取函数（增强版） --------------------
@st.cache_data(ttl=3600)  # 新闻缓存1小时
def get_news(ticker: str) -> List[Dict]:
    """
    获取股票相关新闻，使用多种备用方案：
    1. 首先尝试Finnhub API
    2. 如果失败，尝试yfinance新闻
    3. 最后尝试Google搜索新闻
    """
    news_list = []
    
    # 方案1：尝试Finnhub API
    try:
        # 处理股票代码用于Finnhub API
        finnhub_ticker = process_finnhub_ticker(ticker)
        logger.info(f"使用Finnhub获取新闻: {finnhub_ticker}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 转换为时间戳（秒）
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        params = {
            'symbol': finnhub_ticker,
            'from': from_timestamp,
            'to': to_timestamp,
            'token': CONFIG['news_api']['key']
        }
        
        response = requests.get(CONFIG['news_api']['url'], params=params, timeout=15)
        logger.info(f"Finnhub新闻API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            news_items = response.json()
            # 过滤掉无效新闻
            news_items = [item for item in news_items if item.get('headline') and item.get('url')]
            
            # 只保留最新的10条新闻
            news_items = sorted(news_items, key=lambda x: x.get('datetime', 0), reverse=True)[:10]
            
            for item in news_items:
                title = item.get('headline', '')
                
                try:
                    publish_date = datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
                except:
                    publish_date = "未知时间"
                
                news_list.append({
                    'title': title,
                    'link': item.get('url', ''),
                    'publish_date': publish_date,
                    'source': item.get('source', 'Unknown')
                })
            
            # 如果Finnhub返回了新闻，直接返回
            if news_list:
                logger.info(f"从Finnhub获取到 {len(news_list)} 条新闻")
                return news_list
    except Exception as e:
        logger.error(f"Finnhub获取新闻失败 {ticker}: {e}")
    
    # 方案2：尝试yfinance新闻
    try:
        logger.info(f"尝试使用yfinance获取新闻: {ticker}")
        processed_ticker = process_hk_ticker(ticker)
        stock = yf.Ticker(processed_ticker)
        yf_news = stock.news
        
        if yf_news:
            # 只保留最新的10条新闻
            yf_news = sorted(yf_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:10]
            
            for item in yf_news:
                title = item.get('title', '')
                
                # 发布时间
                pub_time = item.get('providerPublishTime')
                if pub_time:
                    publish_date = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                else:
                    publish_date = "未知时间"
                
                news_list.append({
                    'title': title,
                    'link': item.get('link', ''),
                    'publish_date': publish_date,
                    'source': item.get('publisher', '未知来源')
                })
            
            if news_list:
                logger.info(f"从yfinance获取到 {len(news_list)} 条新闻")
                return news_list
    except Exception as e:
        logger.error(f"yfinance获取新闻失败 {ticker}: {e}")
    
    # 方案3：Google搜索作为最后备用
    try:
        logger.info(f"尝试使用Google搜索获取新闻: {ticker}")
        # 获取公司名称用于搜索
        info, _ = get_stock_info(ticker)
        company_name = info.get('longName', ticker)
        
        # 创建搜索查询
        query = f"{company_name} 股票 新闻"
        search_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        
        response = requests.get(search_url, timeout=15)
        if response.status_code == 200:
            from xml.etree import ElementTree as ET
            
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            # 只保留最新的10条新闻
            items = items[:10]
            
            for item in items:
                title = item.find('title').text if item.find('title') is not None else "无标题"
                link = item.find('link').text if item.find('link') is not None else "#"
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "未知时间"
                source = item.find('source').text if item.find('source') is not None else "未知来源"
                
                news_list.append({
                    'title': title,
                    'link': link,
                    'publish_date': pub_date,
                    'source': source
                })
            
            if news_list:
                logger.info(f"从Google搜索获取到 {len(news_list)} 条新闻")
                return news_list
    except Exception as e:
        logger.error(f"Google搜索获取新闻失败 {ticker}: {e}")
    
    # 所有方案都失败时返回空列表
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

def calculate_volatility(close: pd.Series, period: int = 20) -> float:
    """计算历史波动率"""
    if len(close) < period:
        return 0.0
    returns = close.pct_change().dropna()
    volatility = returns.rolling(period).std().iloc[-1] * np.sqrt(252)  # 年化波动率
    return volatility

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """计算平均真实波幅(ATR)"""
    if len(high) < period:
        return 0.0
    tr = pd.DataFrame(index=high.index)
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift(1))
    tr['l-pc'] = abs(low - close.shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = tr['tr'].rolling(period).mean().iloc[-1]
    return atr

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

# -------------------- 投资建议函数（分短期、中期、长期） --------------------
def get_investment_advice(ticker: str, hist: pd.DataFrame, info: dict) -> Tuple[str, str, str, List[float]]:
    """分短期、中期、长期给出投资建议和买入价格范围，基于实时数据分析"""
    try:
        # 获取当前价格
        current_price = info.get('currentPrice', 0)
        currency = info.get('currency', 'USD')
        
        # 计算技术指标
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        
        # 计算不同时间段的均线
        ma_short = hist['Close'].rolling(5).mean().iloc[-1] if len(hist) >= 5 else current_price
        ma_medium = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
        ma_long = hist['Close'].rolling(60).mean().iloc[-1] if len(hist) >= 60 else current_price
        
        # 获取市场情绪
        sentiment = get_sentiment(ticker)
        
        # 计算支撑位和阻力位
        support, resistance = calculate_support_resistance(hist['Close'])
        
        # 获取基本面数据
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # 计算波动率
        volatility = calculate_volatility(hist['Close'])
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'])
        
        # 计算精确价格范围
        def calculate_precise_price_range(base_price, volatility_factor=0.5):
            """根据波动率计算精确价格范围"""
            # 波动率调整因子 (0.3-0.7之间)
            volatility_adjustment = max(0.3, min(0.7, volatility_factor * (1 + volatility)))
            price_range = atr * volatility_adjustment
            return [
                max(0, base_price - price_range),
                base_price + price_range
            ]
        
        # 短期建议 (1周内) - 基于技术指标
        short_term = ""
        short_term_price = []
        if rsi < 30:
            short_term = "短期买入机会：RSI超卖，可能存在反弹机会"
            short_term_price = calculate_precise_price_range(support, 0.4)
        elif rsi > 70:
            short_term = "短期谨慎：RSI超买，可能有回调风险"
            short_term_price = calculate_precise_price_range(support, 0.5)
        else:
            short_term = "短期中性：技术指标未显示明显信号"
            short_term_price = calculate_precise_price_range(current_price, 0.6)
            
        # 中期建议 (1-3个月) - 结合技术和基本面
        medium_term = ""
        medium_term_price = []
        if macd > signal and sentiment == "正面":
            medium_term = "中期看涨：MACD金叉形成，市场情绪积极"
            medium_term_price = calculate_precise_price_range(ma_medium, 0.4)
        elif macd < signal and sentiment == "负面":
            medium_term = "中期谨慎：MACD死叉形成，市场情绪谨慎"
            medium_term_price = calculate_precise_price_range(ma_medium * 0.98, 0.5)
        else:
            medium_term = "中期中性：技术指标和市场情绪未形成明显趋势"
            medium_term_price = calculate_precise_price_range(ma_medium, 0.6)
            
        # 长期建议 (6个月以上) - 基于基本面和长期趋势
        long_term = ""
        long_term_price = []
        if current_price > ma_long and pe_ratio < 25 and pb_ratio < 3:
            long_term = "长期看涨：股价位于长期均线之上，估值合理"
            long_term_price = calculate_precise_price_range(ma_long * 0.98, 0.3)
        elif current_price < ma_long and pe_ratio > 30 and pb_ratio > 5:
            long_term = "长期谨慎：股价低于长期均线，估值偏高"
            long_term_price = calculate_precise_price_range(ma_long * 0.95, 0.5)
        else:
            long_term = "长期中性：基本面和技术面未显示明显优势或风险"
            long_term_price = calculate_precise_price_range(ma_long, 0.4)
            
        # 添加详细分析
        short_term += f"\n- RSI: {rsi:.2f}, 支撑位: {support:.2f}, 阻力位: {resistance:.2f}, 波动率: {volatility:.2%}"
        medium_term += f"\n- MACD: {macd:.4f}, Signal: {signal:.4f}, 市场情绪: {sentiment}, ATR: {atr:.2f}"
        long_term += f"\n- 市盈率: {pe_ratio:.2f}, 市净率: {pb_ratio:.2f}, 股息率: {dividend_yield:.2f}%"
            
        return short_term, medium_term, long_term, [
            short_term_price, 
            medium_term_price, 
            long_term_price
        ]
    except Exception as e:
        logger.error(f"生成投资建议失败: {e}")
        return (
            "短期建议：数据不足，无法生成建议",
            "中期建议：数据不足，无法生成建议",
            "长期建议：数据不足，无法生成建议",
            [[0, 0], [0, 0], [0, 0]]
        )

def get_hedge_advice(ticker: str, hist: pd.DataFrame, info: dict, quantity: int, cost_price: float) -> str:
    """获取对冲建议，包括期权策略"""
    try:
        if quantity <= 0:
            return "您没有持仓，无需对冲建议"
            
        # 获取当前价格
        current_price = info.get('currentPrice', 0)
        if current_price <= 0:
            return "无法获取当前价格，无法生成对冲建议"
            
        # 计算盈亏情况
        profit_percent = ((current_price - cost_price) / cost_price) * 100 if cost_price > 0 else 0
        profit_value = (current_price - cost_price) * quantity
        
        # 计算波动率
        volatility = calculate_volatility(hist['Close'])
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'])
        
        # 获取市场情绪
        sentiment = get_sentiment(ticker)
        
        # 计算技术指标
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        
        # 基本对冲建议
        advice = ""
        
        # 根据盈亏情况提供建议
        if profit_percent >= 20:
            advice += "📈 您当前持仓盈利显著（{:.2f}%），考虑部分获利了结或对冲保护收益。\n".format(profit_percent)
            advice += "📊 推荐策略：买入看跌期权保护盈利，或卖出备兑看涨期权增加收益。\n"
        elif profit_percent <= -10:
            advice += "📉 您当前持仓亏损（{:.2f}%），考虑风险管理或降低持仓成本。\n".format(profit_percent)
            advice += "📊 推荐策略：卖出虚值看跌期权降低持仓成本，或买入看涨期权对冲下行风险。\n"
        else:
            advice += "📊 您当前持仓盈亏平衡（{:.2f}%），考虑中性策略管理风险。\n".format(profit_percent)
            advice += "📊 推荐策略：卖出宽跨式期权组合收取权利金，或买入跨式期权捕捉波动。\n"
        
        # 根据波动率提供建议
        if volatility > 0.3:
            advice += "\n⚠️ 当前市场波动率较高（{:.2%}），建议加强风险管理。".format(volatility)
            advice += "\n📊 推荐策略：买入保护性期权或采用delta中性策略。"
        elif volatility < 0.15:
            advice += "\nℹ️ 当前市场波动率较低（{:.2%}），适合卖出期权策略。".format(volatility)
            advice += "\n📊 推荐策略：卖出备兑看涨期权或卖出看跌期权收取权利金。"
        else:
            advice += "\nℹ️ 当前市场波动率适中（{:.2%}），适合中性策略。".format(volatility)
        
        # 根据技术指标提供建议
        if rsi > 70 and macd < signal:
            advice += "\n⚠️ 技术指标显示超买信号，短期可能有回调风险。"
            advice += "\n📊 推荐策略：买入看跌期权保护持仓，或卖出虚值看涨期权。"
        elif rsi < 30 and macd > signal:
            advice += "\nℹ️ 技术指标显示超卖信号，短期可能有反弹机会。"
            advice += "\n📊 推荐策略：卖出看跌期权收取权利金，或买入看涨期权捕捉反弹。"
        
        # 具体期权策略建议
        advice += "\n\n### 具体期权策略建议："
        
        # 根据持仓数量计算期权合约数
        contracts = max(1, quantity // 100)  # 每100股对应1份期权合约
        
        if profit_percent >= 20:
            # 盈利较多时保护策略
            advice += f"\n1. **买入保护性看跌期权**"
            advice += f"\n   - 买入 {contracts} 份行权价为当前价格90%的看跌期权"
            advice += f"\n   - 保护持仓价值，限制下行风险"
            
            advice += f"\n2. **备兑看涨期权策略**"
            advice += f"\n   - 卖出 {contracts} 份行权价为当前价格110%的看涨期权"
            advice += f"\n   - 获取额外收益，同时锁定部分利润"
        elif profit_percent <= -10:
            # 亏损较多时降低成本策略
            advice += f"\n1. **卖出看跌期权策略**"
            advice += f"\n   - 卖出 {contracts} 份行权价为当前价格90%的看跌期权"
            advice += f"\n   - 收取权利金降低持仓成本"
            
            advice += f"\n2. **风险逆转策略**"
            advice += f"\n   - 卖出虚值看跌期权 + 买入虚值看涨期权"
            advice += f"\n   - 降低持仓成本同时保留上行潜力"
        else:
            # 中性策略
            advice += f"\n1. **卖出宽跨式策略**"
            advice += f"\n   - 卖出 {contracts} 份行权价为当前价格110%的看涨期权"
            advice += f"\n   - 卖出 {contracts} 份行权价为当前价格90%的看跌期权"
            advice += f"\n   - 在稳定市场中收取双倍权利金"
            
            advice += f"\n2. **铁鹰式策略**"
            advice += f"\n   - 同时卖出价内看跌和看涨期权 + 买入更价外期权"
            advice += f"\n   - 限制风险的同时获得稳定收益"
        
        # 添加综合分析因素
        advice += "\n\n### 综合分析因素："
        advice += f"\n- 当前价格: {current_price:.2f}"
        advice += f"\n- 持仓成本: {cost_price:.2f}"
        advice += f"\n- 持仓数量: {quantity}"
        advice += f"\n- 盈亏情况: {profit_value:.2f} ({profit_percent:.2f}%)"
        advice += f"\n- 波动率: {volatility:.2%}"
        advice += f"\n- ATR: {atr:.2f}"
        advice += f"\n- RSI: {rsi:.2f}"
        advice += f"\n- MACD: {macd:.4f}/{signal:.4f}"
        advice += f"\n- 市场情绪: {sentiment}"
        
        return advice
    except Exception as e:
        logger.error(f"生成对冲建议失败: {e}")
        return "无法生成对冲建议，请检查输入数据"

# -------------------- 热门股票函数（增强版） --------------------
@st.cache_data(ttl=3600)
def get_trending_stocks() -> pd.DataFrame:
    try:
        # 获取美股大盘指数成分股作为候选池
        url = "https://finnhub.io/api/v1/stock/symbol?exchange=US&token=" + CONFIG['api_keys']['finnhub']
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            all_stocks = response.json()
            # 过滤出活跃的普通股票（排除ETF、基金等）
            constituents = [
                stock['symbol'] for stock in all_stocks 
                if stock['type'] == 'Common Stock' and not stock['symbol'].endswith('.')
            ][:100]  # 取前100只股票
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
                if hist.empty or len(hist) < 20:  # 需要足够的数据
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
                monthly_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
                momentum_score = min(100, max(0, 50 + monthly_return * 2))  # 每1%涨幅加2分
                score += momentum_score * 0.2
                
                # 确保分数在0-100范围内
                score = max(0, min(100, score))
                
                trending_data.append({
                    '股票代码': ticker,
                    '公司名称': info.get('longName', ticker),
                    '当前价格': info.get('currentPrice', 0),
                    '涨跌幅': monthly_return,  # 使用实际计算的月度涨跌幅
                    'RSI': round(rsi, 2),
                    'MACD': round(macd, 4),
                    '市场情绪': sentiment,
                    '情绪分数': sentiment_score,
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
        return pd.DataFrame()

# -------------------- 页面渲染函数 --------------------
def render_realtime_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    info, _ = get_stock_info(processed_ticker)
    if not info or 'currentPrice' not in info:
        st.error(f"❌ 无法获取股票数据，请尝试以下解决方案：\n"
                 f"1. 港股使用4位数字代码（如'0700'代表腾讯）\n"
                 f"2. 美股使用股票代码（如'TSLA'）\n"
                 f"3. 确保输入正确股票代码")
        return
    
    company_name = info.get('longName', processed_ticker)
    currency = info.get('currency', 'USD')
    
    st.title(f"📊 {company_name} ({processed_ticker})")
    
    # 创建列布局
    col1, col2, col3, col4 = st.columns(4)
    
    # 获取并显示实时数据
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
    
    # 时间范围选择与K线图
    st.markdown("### 📈 价格走势")
    
    # 精简的时间范围选择器
    period_options = {"1日": "1d", "5日": "5d", "1月": "1mo", "3月": "3mo", "1年": "1y", "5年": "5y"}
    selected_period = st.selectbox("选择时间范围", list(period_options.keys()), index=2, 
                                  key='period_selector')
    
    hist = get_historical_data(processed_ticker, period_options[selected_period])
    
    if hist.empty:
        st.warning("⚠️ 无法获取历史数据")
        return
    
    # 优化K线图样式
    fig = go.Figure()
    
    # 添加K线
    fig.add_trace(go.Candlestick(
        x=hist.index, 
        open=hist['Open'], 
        high=hist['High'], 
        low=hist['Low'], 
        close=hist['Close'], 
        name='K线',
        increasing_line_color='#2ECC71',  # 上涨绿色
        decreasing_line_color='#E74C3C'   # 下跌红色
    ))
    
    # 添加均线
    if len(hist) >= 5:
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['Close'].rolling(5).mean(), 
            name='MA5', 
            line=dict(color='#3498DB', width=2)
        ))
    
    if len(hist) >= 20:
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist['Close'].rolling(20).mean(), 
            name='MA20', 
            line=dict(color='#F39C12', width=2)
        ))
    
    # 添加布林带
    if len(hist) >= 20:
        upper, mid, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=upper, 
            name='布林上轨', 
            line=dict(color='#E74C3C', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=lower, 
            name='布林下轨', 
            line=dict(color='#2ECC71', width=1, dash='dash'),
            fill='tonexty',  # 填充到下一个轨迹
            fillcolor='rgba(231, 76, 60, 0.1)'  # 半透明填充
        ))
    
    # 更新图表布局
    fig.update_layout(
        title=f"{processed_ticker} 价格走势",
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=3, label="3月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title="价格"
        )
    )
    
    # 添加成交量柱状图
    volume_fig = go.Figure(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='成交量',
        marker_color=np.where(hist['Close'] > hist['Open'], '#2ECC71', '#E74C3C')
    ))
    
    volume_fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=20, r=20, t=0, b=20),
        template='plotly_white'
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # 盘前/盘后交易数据（带刷新功能）放在页面底部
    if currency == 'USD':
        st.markdown("### 📈 盘前/盘后交易")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        # 使用会话状态存储盘前盘后数据
        if 'pre_post_data' not in st.session_state:
            st.session_state.pre_post_data = {
                'pre_price': info.get('preMarketPrice'),
                'post_price': info.get('postMarketPrice'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # 刷新按钮
        if col3.button("🔄 刷新盘前盘后数据"):
            try:
                # 重新获取股票信息
                new_info, _ = get_stock_info(processed_ticker)
                st.session_state.pre_post_data = {
                    'pre_price': new_info.get('preMarketPrice'),
                    'post_price': new_info.get('postMarketPrice'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.success("数据已刷新！")
            except:
                st.error("刷新失败")
        
        with col1:
            pre_price = st.session_state.pre_post_data['pre_price']
            st.metric("盘前价格", f"{pre_price:.2f} {currency}" if pre_price else "暂无数据")
        
        with col2:
            post_price = st.session_state.pre_post_data['post_price']
            st.metric("盘后价格", f"{post_price:.2f} {currency}" if post_price else "暂无数据")
        
        # 显示刷新时间
        st.caption(f"最后更新时间: {st.session_state.pre_post_data['last_updated']}")

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
    volatility = calculate_volatility(hist['Close'])
    atr = calculate_atr(hist['High'], hist['Low'], hist['Close'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI(14)", f"{rsi:.2f}", "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常")
    col2.metric("MACD", f"{macd:.4f} / {signal:.4f}", "看涨" if macd > signal else "看跌")
    col3.metric("波动率", f"{volatility:.2%}", "高波动" if volatility > 0.3 else "低波动" if volatility < 0.15 else "中波动")
    col4.metric("ATR(14)", f"{atr:.2f}")
    
    tech_data = {
        "指标": ["支撑位", "阻力位", "RSI状态", "MACD状态", "波动率", "ATR"],
        "数值/描述": [
            f"{support:.2f}", f"{resistance:.2f}",
            "超卖" if rsi < 30 else "超买" if rsi > 70 else "正常",
            "看涨" if macd > signal else "看跌",
            f"{volatility:.2%}",
            f"{atr:.2f}"
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
        
        fig = go.Figure(go.Scatter(
            x=rsi_df.index, 
            y=rsi_df['RSI'], 
            name='RSI',
            line=dict(color='#3498DB', width=2)
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
        fig.update_layout(
            title="RSI趋势", 
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_advice_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    hist = get_historical_data(processed_ticker, "1y")  # 获取1年数据用于分析
    info = get_stock_info(processed_ticker)[0]
    if hist.empty or not info:
        st.error("❌ 数据不足，无法生成建议")
        return
    
    # 获取当前价格
    current_price = info.get('currentPrice', 0)
    currency = info.get('currency', 'USD')
    
    # 持仓信息输入
    st.sidebar.markdown("### 📦 持仓信息")
    quantity = st.sidebar.number_input("持有数量", min_value=0, value=0, step=100)
    cost_price = st.sidebar.number_input("成本价格", min_value=0.0, value=0.0, step=0.1)
    
    # 获取分阶段投资建议和买入价格范围
    short_term, medium_term, long_term, price_ranges = get_investment_advice(processed_ticker, hist, info)
    
    # 获取对冲建议
    hedge_advice = get_hedge_advice(processed_ticker, hist, info, quantity, cost_price)
    
    st.title(f"🎯 {processed_ticker} 投资建议")
    st.caption(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示当前价格
    st.metric("当前价格", f"{current_price:.2f} {currency}")
    
    # 创建选项卡布局
    tab1, tab2, tab3, tab4 = st.tabs([
        "短期建议 (1周内)", "中期建议 (1-3个月)", 
        "长期建议 (6个月以上)", "持仓优化建议"
    ])
    
    with tab1:
        st.subheader("短期投资建议")
        st.info(short_term)
        st.markdown(f"**建议买入价格范围:** `{price_ranges[0][0]:.2f} - {price_ranges[0][1]:.2f} {currency}`")
        st.markdown("**关键因素:** RSI指标、短期支撑阻力位、市场情绪")
        
    with tab2:
        st.subheader("中期投资建议")
        st.info(medium_term)
        st.markdown(f"**建议买入价格范围:** `{price_ranges[1][0]:.2f} - {price_ranges[1][1]:.2f} {currency}`")
        st.markdown("**关键因素:** MACD指标、市场情绪、基本面趋势")
        
    with tab3:
        st.subheader("长期投资建议")
        st.info(long_term)
        st.markdown(f"**建议买入价格范围:** `{price_ranges[2][0]:.2f} - {price_ranges[2][1]:.2f} {currency}`")
        st.markdown("**关键因素:** 估值水平、长期趋势、基本面质量")
        
    with tab4:
        st.subheader("持仓优化建议")
        st.markdown("### 📊 持仓对冲建议")
        st.info(hedge_advice)
        st.markdown("### 📝 综合分析因素")
        st.markdown("""
        - **市场波动性**: 高波动性市场需要更积极的对冲策略
        - **技术指标**: RSI、MACD等指标显示市场短期方向
        - **持仓盈亏**: 盈利头寸可考虑获利了结，亏损头寸需风险管理
        - **期权策略**: 根据风险偏好选择保护性或收益增强策略
        - **时间因素**: 短期事件（财报、政策）可能影响策略选择
        - **整体市场趋势**: 大盘走势影响个股表现
        """)
    
    # 添加风险提示
    st.warning("""
    ⚠️ **投资决策综合考量因素**  
    1. 宏观经济环境（利率、通胀、GDP增长）  
    2. 行业趋势与竞争格局  
    3. 公司基本面（盈利能力、负债水平、现金流）  
    4. 技术面指标（趋势、支撑/阻力位）  
    5. 市场情绪与资金流向  
    6. 重大事件风险（财报、政策变化、地缘政治）  
    以上建议仅供参考，实际决策需结合个人风险承受能力
    """)

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
            df[['股票代码', '公司名称', '当前价格', '涨跌幅', 'RSI', 'MACD', '市场情绪', '情绪分数', '推荐得分', '建议']],
            hide_index=True,
            column_config={
                "涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
                "当前价格": st.column_config.NumberColumn(format="$%.2f"),
                "情绪分数": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100
                ),
                "推荐得分": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100
                )
            },
            height=800
        )
        
        # 情绪分析可视化
        st.markdown("### 市场情绪分布")
        sentiment_df = pd.DataFrame({
            '情绪': df['市场情绪'].value_counts().index,
            '数量': df['市场情绪'].value_counts().values
        })
        fig = px.pie(sentiment_df, names='情绪', values='数量', 
                     title='推荐股票情绪分布', 
                     color='情绪',
                     color_discrete_map={'正面':'#2ECC71', '中性':'#3498DB', '负面':'#E74C3C'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无股票数据")

def render_news_page(ticker: str):
    processed_ticker = process_hk_ticker(ticker)
    st.title(f"📰 {processed_ticker} 新闻")
    
    # 添加新闻加载状态
    with st.spinner("正在加载最新新闻..."):
        news_list = get_news(ticker)
    
    # 添加新闻刷新按钮
    if st.button("🔄 刷新新闻数据", key="refresh_news"):
        # 清除缓存并重新加载
        get_news.clear()
        st.rerun()
    
    if not news_list:
        st.warning("⚠️ 无法获取相关新闻，请稍后再试")
        return
    
    # 只显示标题、时间和外链
    st.markdown("### 最新10条新闻")
    
    # 创建简单的卡片式布局
    for news in news_list:
        # 创建卡片容器
        with st.container():
            # 标题和时间在同一行
            col1, col2 = st.columns([4, 1])
            col1.subheader(news['title'])
            col2.caption(f"📅 {news['publish_date']}")
            
            # 来源和链接
            st.markdown(f"来源: **{news['source']}**")
            st.markdown(f"[阅读原文 ↗]({news['link']})")
            
            # 分隔线
            st.divider()

# -------------------- 回调函数 --------------------
def update_current_ticker():
    """更新当前选中的股票代码"""
    if st.session_state.search_input and st.session_state.search_input != st.session_state.current_ticker:
        st.session_state.current_ticker = st.session_state.search_input

# -------------------- 主应用 --------------------
def main():
    st.set_page_config(page_title=CONFIG['page_title'], layout='wide')
    st.sidebar.title("🚀 智能股票分析")
    st.sidebar.markdown("---")
    
    # 使用会话状态跟踪当前选中的股票
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = "TSLA"  # 默认股票改为TSLA
    
    # 股票代码输入
    st.sidebar.markdown("### 🔍 股票查询")
    
    # 使用on_change回调处理回车提交
    st.sidebar.text_input(
        "输入股票代码", 
        value=st.session_state.current_ticker,
        help="美股: TSLA | 港股: 0700（4位数字）",
        key="search_input",
        on_change=update_current_ticker
    )
    
    # 热门股票快速访问
    st.sidebar.markdown("**🚀 热门股票**")
    hot_cols = st.sidebar.columns(3)
    hot_stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "0700"]
    for i, stock in enumerate(hot_stocks):
        if hot_cols[i % 3].button(stock, use_container_width=True):
            st.session_state.current_ticker = stock
            st.rerun()
    
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
