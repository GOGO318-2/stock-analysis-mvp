import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import time
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(page_title="股票分析MVP", layout="wide")

# 常量配置
class Config:
    API_KEYS = {
        "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",
        "alpha_vantage": "Z45S0SLJGM378PIO", 
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
    }
    API_ORDER = ["yfinance", "alpha_vantage", "finnhub", "polygon"]
    TECH_STOCKS = [
        'NVDA', 'TSLA', 'AMD', 'GOOG', 'AAPL', 'MSFT', 'AMZN', 'META', 'INTC', 'QCOM',
        'IBM', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'EBAY', 'NFLX', 'DIS', 'BABA',
        'JD', 'BIDU', 'TSM', 'ASML', 'MU', 'KLAC', 'LRCX', 'AVGO', 'TXN', 'STM',
        'ARM', 'SNPS', 'CDNS', 'ANSS', 'KEYS', 'TER', 'SWKS', 'QRVO', 'MPWR', 'MCHP',
        'ON', 'NXPI', 'ADI', 'LSCC', 'SYNA', 'POWI', 'SLAB', 'CRUS', 'MTSI', 'RMBS',
        'CEVA', 'SGH', 'SITM'
    ]
    PERIOD_OPTIONS = {"1日": "1d", "5日": "5d", "日K": "1mo", "周K": "3mo", "月K": "1y", "季K": "5y"}

class StockAnalyzer:
    def __init__(self):
        self.session_state_init()
    
    def session_state_init(self):
        """初始化session state"""
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        if 'top50' not in st.session_state:
            st.session_state.top50 = pd.DataFrame()
    
    def format_ticker(self, ticker_input: str) -> str:
        """格式化股票代码"""
        ticker_input = ticker_input.upper().strip()
        
        # 港股处理：自动添加.HK后缀，去除前导零
        if ticker_input.isdigit():
            ticker_clean = ticker_input.lstrip('0')
            if 1 <= len(ticker_clean) <= 5:
                return f"{ticker_clean}.HK"
        
        return ticker_input
    
    @st.cache_data(ttl=300)  # 5分钟缓存
    def get_stock_data(_self, ticker: str) -> Tuple[Dict, pd.DataFrame]:
        """获取股票数据，带错误处理和重试机制"""
        for api in Config.API_ORDER:
            try:
                time.sleep(0.5)  # 减少延迟
                
                if api == "yfinance":
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # 检查数据有效性
                    if not info or info.get('regularMarketPrice') is None:
                        continue
                        
                    # 获取推荐数据（如果可用）
                    rec = pd.DataFrame()
                    try:
                        if hasattr(stock, 'recommendations_summary'):
                            rec = stock.recommendations_summary
                    except:
                        pass
                    
                    return info, rec
                    
                elif api == "finnhub":
                    quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={Config.API_KEYS['finnhub']}"
                    response = requests.get(quote_url, timeout=10)
                    quote_resp = response.json()
                    
                    if 'c' in quote_resp and quote_resp['c'] > 0:
                        info = {
                            'currentPrice': quote_resp['c'],
                            'dayHigh': quote_resp['h'],
                            'dayLow': quote_resp['l'],
                            'preMarketPrice': quote_resp.get('pc', 'N/A'),
                            'postMarketPrice': 'N/A'
                        }
                        return info, pd.DataFrame()
                        
            except Exception as e:
                logger.warning(f"{api} 获取数据失败: {e}")
                continue
        
        st.error("所有API均失败，请稍后重试或检查网络连接。")
        return {}, pd.DataFrame()
    
    @st.cache_data(ttl=600)  # 10分钟缓存
    def get_historical_data(_self, ticker: str, period: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"未获取到 {ticker} 的历史数据")
                return pd.DataFrame()
                
            return hist
            
        except Exception as e:
            logger.error(f"历史数据获取失败 {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if hist.empty:
            return {}
        
        close = hist['Close']
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = self.calculate_rsi(close)
            
            # MACD
            macd, signal = self.calculate_macd(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            
            # 布林带
            upper, middle, lower = self.calculate_bollinger_bands(close)
            indicators['bb_upper'] = upper
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = lower
            
            # 移动平均线
            indicators['ma5'] = close.rolling(window=5).mean()
            indicators['ma20'] = close.rolling(window=20).mean()
            
            # 成交量相关
            indicators['avg_volume'] = hist['Volume'].mean()
            
            # 夏普比率
            returns = close.pct_change().dropna()
            if len(returns) > 1 and returns.std() != 0:
                indicators['sharpe'] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                indicators['sharpe'] = np.nan
                
        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
        
        return indicators
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def calculate_macd(self, close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[float, float]:
        """计算MACD"""
        try:
            ema_short = close.ewm(span=short, adjust=False).mean()
            ema_long = close.ewm(span=long, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            return macd_line.iloc[-1], signal_line.iloc[-1]
        except:
            return 0.0, 0.0
    
    def calculate_bollinger_bands(self, close: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        try:
            rolling_mean = close.rolling(window=window).mean()
            rolling_std = close.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return upper_band, rolling_mean, lower_band
        except:
            return pd.Series(), pd.Series(), pd.Series()
    
    @st.cache_data(ttl=1800)  # 30分钟缓存
    def get_news_and_sentiment(_self, ticker: str) -> List[Dict]:
        """获取新闻和情感分析"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:5] if hasattr(stock, 'news') else []
            
            news_list = []
            positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'growth', 'beat', 'strong']
            negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'decline', 'miss', 'weak']
            
            for item in news:
                title = item.get('title', '')
                title_lower = title.lower()
                
                # 改进的情感分析
                positive_score = sum(1 for kw in positive_keywords if kw in title_lower)
                negative_score = sum(1 for kw in negative_keywords if kw in title_lower)
                
                if positive_score > negative_score:
                    sentiment = "正面"
                elif negative_score > positive_score:
                    sentiment = "负面"
                else:
                    sentiment = "中性"
                
                news_list.append({
                    'title': title,
                    'link': item.get('link', ''),
                    'publish_date': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': sentiment
                })
                
            return news_list
            
        except Exception as e:
            logger.warning(f"新闻获取失败: {e}")
            return []
    
    def get_ai_sentiment(self, ticker: str) -> str:
        """获取AI情感分析"""
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.API_KEYS['xai']}", 
                "Content-Type": "application/json"
            }
            data = {
                "model": "grok-beta",
                "messages": [{
                    "role": "user", 
                    "content": f"What is the current market sentiment for stock {ticker}? Reply with only one word: 正面, 负面, or 中性."
                }],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()['choices'][0]['message']['content'].strip()
            
            # 验证返回结果
            valid_sentiments = ["正面", "负面", "中性"]
            return result if result in valid_sentiments else "中性"
            
        except Exception as e:
            logger.warning(f"AI情感分析失败: {e}")
            return "中性"
    
    def get_ai_remark(self, ticker: str) -> str:
        """获取AI投资建议"""
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.API_KEYS['xai']}", 
                "Content-Type": "application/json"
            }
            data = {
                "model": "grok-beta",
                "messages": [{
                    "role": "user", 
                    "content": f"Provide a brief investment remark for {ticker} in Chinese (max 50 characters)."
                }],
                "max_tokens": 100,
                "temperature": 0.3
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.warning(f"AI建议获取失败: {e}")
            return "无建议可用"
    
    def render_sidebar(self) -> str:
        """渲染侧边栏"""
        st.sidebar.title("📊 股票分析器")
        st.sidebar.markdown("💡 *支持港股：输入如0700*")
        
        # 股票代码输入
        ticker_input = st.sidebar.text_input(
            "🔍 输入股票代码", 
            value="TSLA",
            placeholder="例如: TSLA 或 0700"
        ).upper()
        
        ticker = self.format_ticker(ticker_input)
        
        # 添加到Watchlist按钮
        col1, col2 = st.sidebar.columns([2, 1])
        
        if col1.button("➕ 添加到收藏", use_container_width=True):
            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.sidebar.success("✅ 添加成功！")
            else:
                st.sidebar.info("ℹ️ 已在收藏中")
        
        if col2.button("🔄", help="刷新数据"):
            st.cache_data.clear()
            st.sidebar.success("🔄 已刷新")
        
        # Watchlist显示 - 改进样式
        if st.session_state.watchlist:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### ⭐ 我的收藏")
            
            # 创建收藏列表的HTML样式
            watchlist_html = """
            <div style="max-height: 300px; overflow-y: auto;">
            """
            
            for i, wl_ticker in enumerate(st.session_state.watchlist.copy()):
                # 为每个收藏项创建样式化的显示
                watchlist_html += f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 12px;
                    margin: 5px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    font-size: 14px;
                    font-weight: bold;
                ">
                    📈 {wl_ticker}
                </div>
                """
                
                # 为每个收藏项添加操作按钮
                col1, col2, col3 = st.sidebar.columns([2, 1, 1])
                
                if col1.button(f"🔍 查看", key=f"view_{wl_ticker}_{i}", use_container_width=True):
                    st.session_state.selected_ticker = wl_ticker
                    st.rerun()
                
                if col2.button("❌", key=f"remove_{wl_ticker}_{i}", help="移除"):
                    st.session_state.watchlist.remove(wl_ticker)
                    st.rerun()
                
                if col3.button("📊", key=f"chart_{wl_ticker}_{i}", help="图表"):
                    st.session_state.selected_ticker = wl_ticker
                    st.session_state.selected_page = "首页"
                    st.rerun()
            
            watchlist_html += "</div>"
            
            # 清空收藏按钮
            if st.sidebar.button("🗑️ 清空收藏", type="secondary"):
                st.session_state.watchlist = []
                st.sidebar.success("🧹 收藏已清空")
                st.rerun()
        else:
            st.sidebar.markdown("---")
            st.sidebar.info("💫 还没有收藏的股票\n点击上方按钮添加")
        
        # 页面导航 - 改进样式
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 页面导航")
        
        pages = {
            "🏠 首页": "首页", 
            "📊 基本面": "基本面", 
            "💡 投资建议": "投资建议", 
            "🌐 公共市场": "公共市场"
        }
        
        # 使用radio按钮，但带图标
        selected_page_display = st.sidebar.radio(
            "选择页面", 
            list(pages.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        page = pages[selected_page_display]
        
        # 添加分隔线和版本信息
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 12px;">
                📱 股票分析器 v2.0<br>
                💻 AI驱动的智能分析
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        return ticker, page
    
    def render_home_page(self, ticker: str, info: Dict, hist: pd.DataFrame):
        """渲染首页"""
        company_name = info.get('longName', ticker) or ticker
        currency = info.get('currency', 'USD')
        
        st.title(f"{company_name} ({ticker}) 股票仪表板")
        
        # 时间范围选择
        default_index = list(Config.PERIOD_OPTIONS.keys()).index("月K")
        selected_label = st.selectbox(
            "选择时间范围", 
            list(Config.PERIOD_OPTIONS.keys()), 
            index=default_index
        )
        selected_period = Config.PERIOD_OPTIONS[selected_label]
        
        # 获取历史数据
        hist = self.get_historical_data(ticker, selected_period)
        
        if not hist.empty:
            # 计算技术指标
            indicators = self.calculate_technical_indicators(hist)
            
            # 绘制K线图
            fig = go.Figure(data=[
                go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='K线'
                )
            ])
            
            # 添加移动平均线
            if 'ma5' in indicators and not indicators['ma5'].empty:
                fig.add_trace(go.Scatter(
                    x=hist.index, 
                    y=indicators['ma5'], 
                    mode='lines', 
                    name='MA5', 
                    line=dict(color='blue')
                ))
            
            if 'ma20' in indicators and not indicators['ma20'].empty:
                fig.add_trace(go.Scatter(
                    x=hist.index, 
                    y=indicators['ma20'], 
                    mode='lines', 
                    name='MA20', 
                    line=dict(color='orange')
                ))
            
            # 添加布林带
            if all(k in indicators for k in ['bb_upper', 'bb_lower']) and \
               not indicators['bb_upper'].empty:
                fig.add_trace(go.Scatter(
                    x=hist.index, 
                    y=indicators['bb_upper'], 
                    mode='lines', 
                    name='Upper BB', 
                    line=dict(color='red', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index, 
                    y=indicators['bb_lower'], 
                    mode='lines', 
                    name='Lower BB', 
                    line=dict(color='green', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{ticker} {selected_label}K线图",
                xaxis_title="日期",
                yaxis_title="价格",
                xaxis_rangeslider_visible=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示关键指标
            col1, col2, col3 = st.columns(3)
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            day_high = info.get('dayHigh', info.get('regularMarketDayHigh', 0))
            day_low = info.get('dayLow', info.get('regularMarketDayLow', 0))
            
            if current_price:
                col1.metric("当前价格", f"{current_price:.2f} {currency}")
            if day_high:
                col2.metric("今日最高", f"{day_high:.2f} {currency}")
            if day_low:
                col3.metric("今日最低", f"{day_low:.2f} {currency}")
            
            # 盘前盘后数据（仅美股）
            if currency == 'USD':
                st.subheader("盘前/盘后")
                
                if st.button("刷新"):
                    st.cache_data.clear()  # 清除缓存以获取最新数据
                    st.success("刷新成功！")
                    st.rerun()
                
                col1, col2, col3, col4 = st.columns(4)
                
                pre_market_price = info.get('preMarketPrice', 'N/A')
                pre_market_change = info.get('preMarketChange', 0)
                post_market_price = info.get('postMarketPrice', 'N/A')
                post_market_change = info.get('postMarketChange', 0)
                
                col1.metric("盘前价格", f"{pre_market_price} {currency}" if pre_market_price != 'N/A' else 'N/A')
                col2.metric("盘前变化", f"{pre_market_change:.2f}" if isinstance(pre_market_change, (int, float)) else 'N/A')
                col3.metric("盘后价格", f"{post_market_price} {currency}" if post_market_price != 'N/A' else 'N/A')
                col4.metric("盘后变化", f"{post_market_change:.2f}" if isinstance(post_market_change, (int, float)) else 'N/A')
        else:
            st.error("无历史数据可用，请检查股票代码是否正确。")
    
    def render_fundamentals_page(self, ticker: str, info: Dict):
        """渲染基本面页面"""
        company_name = info.get('longName', ticker) or ticker
        st.title(f"{company_name} ({ticker}) 基本面")
        
        if not info:
            st.error("无基本面数据可用。")
            return
        
        hist = self.get_historical_data(ticker, "1mo")
        if hist.empty:
            st.error("无历史数据，无法计算技术指标。")
            return
        
        # 计算技术指标
        indicators = self.calculate_technical_indicators(hist)
        
        # 构建基本面数据表
        fundamentals_data = {
            "指标": [
                "市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", 
                "Beta", "ROE", "负债权益比", "RSI (14日)", 
                "MACD", "平均成交量", "Sharpe Ratio"
            ],
            "值": [
                self.format_number(info.get('marketCap')),
                self.format_number(info.get('trailingPE')),
                self.format_number(info.get('trailingEps')),
                f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                self.format_number(info.get('beta')),
                f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                self.format_number(info.get('debtToEquity')),
                f"{indicators.get('rsi', 50):.2f}",
                f"{indicators.get('macd', 0):.2f} (Signal: {indicators.get('macd_signal', 0):.2f})",
                f"{indicators.get('avg_volume', 0):,.0f}",
                f"{indicators.get('sharpe', 0):.2f}" if not np.isnan(indicators.get('sharpe', np.nan)) else 'N/A'
            ]
        }
        
        df = pd.DataFrame(fundamentals_data)
        st.table(df)
        
        # 添加技术指标解读
        st.subheader("技术指标解读")
        rsi = indicators.get('rsi', 50)
        
        if rsi < 30:
            st.success("RSI显示超卖状态，可能是买入机会")
        elif rsi > 70:
            st.warning("RSI显示超买状态，注意风险")
        else:
            st.info("RSI处于正常范围")
    
    def format_number(self, value) -> str:
        """格式化数字显示"""
        if value is None or value == 'N/A':
            return 'N/A'
        
        try:
            if isinstance(value, (int, float)) and not np.isnan(value):
                if abs(value) >= 1e9:
                    return f"{value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    return f"{value/1e6:.2f}M"
                elif abs(value) >= 1e3:
                    return f"{value/1e3:.2f}K"
                else:
                    return f"{value:.2f}"
            else:
                return 'N/A'
        except:
            return 'N/A'
    
    def render_investment_advice_page(self, ticker: str, info: Dict):
        """渲染投资建议页面"""
        company_name = info.get('longName', ticker) or ticker
        st.title(f"{company_name} ({ticker}) 投资建议")
        
        if not info:
            st.error("无投资建议数据可用。")
            return
        
        hist = self.get_historical_data(ticker, "1mo")
        if hist.empty:
            st.error("无历史数据，无法生成投资建议。")
            return
        
        # 计算技术指标
        indicators = self.calculate_technical_indicators(hist)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        
        # 获取新闻情感
        news = self.get_news_and_sentiment(ticker)
        news_sentiment = self.analyze_news_sentiment(news)
        
        # 获取AI情感分析
        ai_sentiment = self.get_ai_sentiment(ticker)
        ai_remark = self.get_ai_remark(ticker)
        
        # 计算价位
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        target_price = info.get('targetMeanPrice', current_price * 1.1) if current_price else 0
        support = current_price * 0.95 if current_price else 0
        resistance = current_price * 1.05 if current_price else 0
        
        # 生成投资建议
        advice_data = self.generate_investment_advice(
            rsi, macd, news_sentiment, ai_sentiment, 
            current_price, support, resistance, target_price
        )
        
        # 交易类型筛选 - 默认选择短期
        trade_type = st.selectbox("选择交易类型", ["短期", "趋势", "波段"], index=0)
        
        # 过滤数据
        filtered_advice = [advice for advice in advice_data if advice["阶段"] == trade_type]
        
        if filtered_advice:
            df = pd.DataFrame(filtered_advice)
            
            # 使用自定义样式的表格
            st.markdown("### 📊 投资策略建议")
            
            # 创建HTML表格，解决样式问题
            table_html = """
            <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <thead>
                        <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">阶段</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">时机</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">价位</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">仓位</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">备注</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for _, row in df.iterrows():
                color = "#e8f5e8" if "入场" in str(row['时机']) else "#fff2e8" if "止盈" in str(row['时机']) else "#ffe8e8"
                table_html += f"""
                        <tr style="background-color: {color};">
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{row['阶段']}</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{row['时机']}</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{row['价位']}</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{row['仓位']}</td>
                            <td style="padding: 12px; border: 1px solid #dee2e6;">{row['备注']}</td>
                        </tr>
                """
            
            table_html += """
                    </tbody>
                </table>
            </div>
            """
            
            st.markdown(table_html, unsafe_allow_html=True)
        
        # 显示综合建议
        recommendation = self.get_recommendation(rsi, macd, news_sentiment)
        st.markdown("### 💡 综合分析建议")
        
        # 创建带样式的建议框
        advice_color = "#d4edda" if "买入" in recommendation else "#f8d7da" if "卖出" in recommendation else "#d1ecf1"
        border_color = "#c3e6cb" if "买入" in recommendation else "#f5c6cb" if "卖出" in recommendation else "#bee5eb"
        
        st.markdown(
            f"""
            <div style="
                background-color: {advice_color}; 
                border: 1px solid {border_color}; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0 0 10px 0; color: #333;">🎯 投资建议</h4>
                <p style="margin: 5px 0; font-size: 16px;"><strong>技术面:</strong> RSI {rsi:.0f} - {recommendation}</p>
                <p style="margin: 5px 0; font-size: 16px;"><strong>目标价位:</strong> ${target_price:.2f}</p>
                <p style="margin: 5px 0; font-size: 16px;"><strong>AI分析:</strong> {ai_remark}</p>
                <p style="margin: 5px 0; font-size: 14px; color: #666;"><strong>市场情绪:</strong> {ai_sentiment} | <strong>新闻情绪:</strong> {news_sentiment}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 新闻情感分析 - 改进显示
        if news:
            st.markdown("### 📰 相关新闻分析")
            
            for i, item in enumerate(news[:3]):
                sentiment_color = "#28a745" if item['sentiment'] == "正面" else "#dc3545" if item['sentiment'] == "负面" else "#6c757d"
                sentiment_icon = "📈" if item['sentiment'] == "正面" else "📉" if item['sentiment'] == "负面" else "➖"
                
                st.markdown(
                    f"""
                    <div style="
                        background: white; 
                        border-left: 4px solid {sentiment_color}; 
                        padding: 15px; 
                        margin: 10px 0; 
                        border-radius: 0 8px 8px 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <h5 style="margin: 0 0 8px 0; color: #333; font-size: 16px;">{item['title']}</h5>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                            <span style="color: {sentiment_color}; font-weight: bold;">{sentiment_icon} {item['sentiment']}</span>
                            <span style="color: #6c757d; font-size: 12px;">{item['publish_date']}</span>
                        </div>
                        {f'<a href="{item["link"]}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 14px;">📖 查看详情</a>' if item.get('link') else ''}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("📭 暂无相关新闻数据")
    
    def analyze_news_sentiment(self, news: List[Dict]) -> str:
        """分析新闻整体情感"""
        if not news:
            return "中性"
        
        sentiments = [item['sentiment'] for item in news]
        positive_count = sentiments.count("正面")
        negative_count = sentiments.count("负面")
        
        if positive_count > negative_count:
            return "正面"
        elif negative_count > positive_count:
            return "负面"
        else:
            return "中性"
    
    def generate_investment_advice(self, rsi: float, macd: float, news_sentiment: str, 
                                 ai_sentiment: str, current_price: float, support: float, 
                                 resistance: float, target_price: float) -> List[Dict]:
        """生成投资建议"""
        advice_data = []
        
        # 根据RSI确定仓位建议
        position_advice = "60%" if rsi < 40 else "30%" if rsi > 60 else "40%"
        rsi_signal = "买入" if rsi < 40 else "卖出" if rsi > 60 else "持仓"
        
        # 根据MACD确定趋势
        macd_signal = "看涨" if macd > 0 else "看跌"
        
        advice_data.extend([
            {
                "阶段": "短期",
                "时机": "入场",
                "价位": f"{support:.2f}-{current_price:.2f}",
                "仓位": position_advice,
                "备注": f"RSI {rsi:.0f} {rsi_signal}, 新闻{news_sentiment}"
            },
            {
                "阶段": "短期",
                "时机": "止盈",
                "价位": f"{resistance:.2f}",
                "仓位": "减仓50%",
                "备注": f"短期阻力位"
            },
            {
                "阶段": "趋势",
                "时机": "入场",
                "价位": f"{support:.2f}-{current_price:.2f}",
                "仓位": "加仓30%",
                "备注": f"长期持仓, AI情感{ai_sentiment}"
            },
            {
                "阶段": "趋势",
                "时机": "止损",
                "价位": f"{support * 0.90:.2f}",
                "仓位": "清仓",
                "备注": f"跌破重要支撑"
            },
            {
                "阶段": "波段",
                "时机": "入场",
                "价位": f"{support:.2f}-{resistance:.2f}",
                "仓位": "70%" if macd > 0 else "30%",
                "备注": f"MACD {macd_signal}, 波段操作"
            },
            {
                "阶段": "波段",
                "时机": "止盈/止损",
                "价位": f"{target_price:.2f}/{support:.2f}",
                "仓位": "分批减仓",
                "备注": f"目标价位/支撑位"
            }
        ])
        
        return advice_data
    
    def get_recommendation(self, rsi: float, macd: float, news_sentiment: str) -> str:
        """获取综合推荐"""
        if rsi < 30 and macd > 0 and news_sentiment == "正面":
            return "强烈买入"
        elif rsi < 40 and (macd > 0 or news_sentiment == "正面"):
            return "买入"
        elif rsi > 70 and macd < 0 and news_sentiment == "负面":
            return "强烈卖出"
        elif rsi > 60 and (macd < 0 or news_sentiment == "负面"):
            return "卖出"
        else:
            return "持仓观望"
    
    def render_market_page(self):
        """渲染公共市场页面"""
        st.title("🌐 公共市场 - Top 50 科技美股推荐")
        
        # 更新按钮
        col1, col2, col3 = st.columns([1, 1, 3])
        
        update_clicked = col1.button("🔄 更新Top 50", use_container_width=True)
        
        if col2.button("🗑️ 清除缓存", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ 缓存已清除")
        
        if update_clicked:
            self.update_top50_stocks()
        
        # 显示现有数据
        if not st.session_state.top50.empty:
            st.markdown("### 📈 当前Top 50科技股")
            
            # 添加筛选选项
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_level_filter = st.selectbox("💎 买入等级筛选", ["全部", "高", "中", "低"])
            
            with col2:
                min_price = st.number_input("💰 最低价格", min_value=0.0, value=0.0)
            
            with col3:
                max_price = st.number_input("💰 最高价格", min_value=0.0, value=1000.0)
            
            # 应用筛选
            filtered_df = st.session_state.top50.copy()
            
            if buy_level_filter != "全部":
                filtered_df = filtered_df[filtered_df['买入等级'] == buy_level_filter]
            
            if min_price > 0:
                filtered_df = filtered_df[filtered_df['价格'] >= min_price]
            
            if max_price < 1000:
                filtered_df = filtered_df[filtered_df['价格'] <= max_price]
            
            # 检查买入等级低的情况并给出说明
            low_level_count = len(filtered_df[filtered_df['买入等级'] == '低'])
            high_level_count = len(filtered_df[filtered_df['买入等级'] == '高'])
            
            if high_level_count == 0 and not filtered_df.empty:
                st.warning(
                    "⚠️ **当前市场分析说明**: 没有发现买入等级为'高'的股票。"
                    "这可能是由于以下原因：\n"
                    "- 🔴 市场整体情绪偏向谨慎\n"
                    "- 📉 技术指标显示超买状态\n"
                    "- 📰 近期负面新闻较多\n"
                    "- 💹 交易量相对较低\n\n"
                    "建议等待更好的入场时机或考虑分批建仓策略。"
                )
            
            if low_level_count > len(filtered_df) * 0.7:
                st.info(
                    "ℹ️ **市场提醒**: 当前大部分股票买入等级较低，建议谨慎操作，"
                    "可考虑关注基本面较好的个股进行长期布局。"
                )
            
            # 为每个股票添加收藏功能
            if not filtered_df.empty:
                st.markdown("---")
                
                # 创建自定义表格显示，每行添加收藏按钮
                for idx, (_, row) in enumerate(filtered_df.iterrows()):
                    # 根据买入等级设置颜色
                    level_colors = {
                        '高': ('#d4edda', '#155724', '🟢'),
                        '中': ('#fff3cd', '#856404', '🟡'), 
                        '低': ('#f8d7da', '#721c24', '🔴')
                    }
                    
                    bg_color, text_color, icon = level_colors.get(row['买入等级'], ('#f8f9fa', '#212529', '⚪'))
                    
                    # 创建股票信息卡片
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div style="
                                background: {bg_color}; 
                                border: 1px solid {text_color}20; 
                                border-radius: 10px; 
                                padding: 15px; 
                                margin: 10px 0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <h4 style="margin: 0; color: {text_color};">
                                            {icon} {row['股票代码']} - {row['公司名称']}
                                        </h4>
                                        <p style="margin: 5px 0; color: {text_color};">
                                            💰 价格: ${row['价格']:.2f} | 📊 涨幅: {row['涨幅']} | 
                                            🎯 买入等级: <strong>{row['买入等级']}</strong>
                                        </p>
                                        <p style="margin: 5px 0; color: {text_color};">
                                            📈 建议买入价: ${row['买入价']} | 💼 市值: {row['市值']}
                                        </p>
                                        <p style="margin: 5px 0; color: {text_color}; font-size: 12px;">
                                            💡 {row['备注']}
                                        </p>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        # 收藏按钮
                        is_favorited = row['股票代码'] in st.session_state.watchlist
                        
                        if is_favorited:
                            if st.button(f"⭐ 已收藏", key=f"fav_{row['股票代码']}_{idx}", 
                                       type="secondary", use_container_width=True):
                                st.session_state.watchlist.remove(row['股票代码'])
                                st.success(f"📤 已取消收藏 {row['股票代码']}")
                                st.rerun()
                        else:
                            if st.button(f"⭐ 收藏", key=f"unfav_{row['股票代码']}_{idx}", 
                                       use_container_width=True):
                                st.session_state.watchlist.append(row['股票代码'])
                                st.success(f"📥 已收藏 {row['股票代码']}")
                                st.rerun()
                        
                        # 快速查看按钮
                        if st.button(f"🔍 查看", key=f"view_{row['股票代码']}_{idx}", 
                                   use_container_width=True):
                            st.session_state.selected_ticker = row['股票代码']
                            st.session_state.selected_page = "首页"
                            st.rerun()
            
            # 添加统计信息
            st.markdown("---")
            st.markdown("### 📊 统计信息")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 总股票数", len(filtered_df))
            
            with col2:
                high_level_count = len(filtered_df[filtered_df['买入等级'] == '高'])
                st.metric("🟢 高等级股票", high_level_count)
            
            with col3:
                avg_price = filtered_df['价格'].mean() if not filtered_df.empty else 0
                st.metric("💰 平均价格", f"${avg_price:.2f}")
            
            with col4:
                positive_change = len(filtered_df[filtered_df['涨幅'].str.contains(r'^[^-]', na=False)])
                st.metric("📈 上涨股票", positive_change)
        
        else:
            st.markdown(
                """
                <div style="
                    text-align: center; 
                    padding: 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    border-radius: 15px;
                    margin: 20px 0;
                ">
                    <h2>🚀 开始分析</h2>
                    <p>点击上方 '🔄 更新Top 50' 按钮获取最新的科技股分析数据</p>
                    <p>我们将为您分析市场情绪、技术指标和AI建议</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def update_top50_stocks(self):
        """更新Top 50股票数据"""
        with st.spinner('正在筛选股票数据，请稍候...'):
            stock_data = []
            progress_bar = st.progress(0)
            
            for i, tick in enumerate(Config.TECH_STOCKS):
                try:
                    # 更新进度条
                    progress_bar.progress((i + 1) / len(Config.TECH_STOCKS))
                    
                    # 获取股票数据
                    info, _ = self.get_stock_data(tick)
                    if not info:
                        continue
                    
                    # 获取历史数据
                    hist = self.get_historical_data(tick, "1wk")
                    if hist.empty or len(hist) < 2:
                        continue
                    
                    # 计算指标
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    if not current_price:
                        continue
                    
                    volume_avg = hist['Volume'].mean()
                    turnover_avg = volume_avg * hist['Close'].mean()
                    
                    # 计算涨幅
                    if len(hist) >= 2:
                        pct_change = ((current_price - hist['Close'].iloc[-2]) / 
                                    hist['Close'].iloc[-2] * 100)
                    else:
                        pct_change = 0
                    
                    # AI情感分析
                    sentiment = self.get_ai_sentiment(tick)
                    
                    # 计算活跃度评分
                    activity_score = (volume_avg / 1e8) + (2 if sentiment == "正面" 
                                                         else -2 if sentiment == "负面" else 0)
                    
                    # 买入等级
                    buy_level = "高" if activity_score > 5 else "中" if activity_score > 2 else "低"
                    
                    # 建议买入价
                    buy_price = current_price * 0.95
                    
                    # AI建议
                    remark = self.get_ai_remark(tick)
                    
                    stock_data.append({
                        '股票代码': tick,
                        '公司名称': info.get('longName', tick)[:20] + '...' if len(info.get('longName', tick)) > 20 else info.get('longName', tick),
                        '市值': self.format_number(info.get('marketCap')),
                        '成交额': self.format_number(turnover_avg),
                        '成交量': self.format_number(volume_avg),
                        '价格': current_price,
                        '最高': info.get('dayHigh', info.get('regularMarketDayHigh', 0)),
                        '最低': info.get('dayLow', info.get('regularMarketDayLow', 0)),
                        '涨幅': f"{pct_change:.2f}%",
                        '买入等级': buy_level,
                        '买入价': f"{buy_price:.2f}",
                        '备注': remark[:30] + '...' if len(remark) > 30 else remark
                    })
                    
                    # 控制请求频率
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"处理股票 {tick} 时出错: {e}")
                    continue
            
            progress_bar.empty()
            
            if stock_data:
                # 创建DataFrame并排序
                df = pd.DataFrame(stock_data)
                
                # 按买入等级排序
                level_order = {'高': 3, '中': 2, '低': 1}
                df['排序'] = df['买入等级'].map(level_order)
                df = df.sort_values(['排序', '涨幅'], ascending=[False, False])
                df = df.drop('排序', axis=1).head(50)
                
                # 保存到session state
                st.session_state.top50 = df
                
                st.success(f"成功更新 {len(df)} 只股票数据！")
                
            else:
                st.error("无法获取股票数据，请稍后重试")
    
    def run(self):
        """运行应用"""
        try:
            # 渲染侧边栏
            ticker, page = self.render_sidebar()
            
            # 检查是否有选中的股票
            if hasattr(st.session_state, 'selected_ticker'):
                ticker = st.session_state.selected_ticker
                delattr(st.session_state, 'selected_ticker')
            
            # 获取股票数据
            if page != "公共市场":
                info, rec = self.get_stock_data(ticker)
            else:
                info, rec = {}, pd.DataFrame()
            
            # 根据页面渲染内容
            if page == "首页":
                hist = self.get_historical_data(ticker, "1mo")
                self.render_home_page(ticker, info, hist)
                
            elif page == "基本面":
                self.render_fundamentals_page(ticker, info)
                
            elif page == "投资建议":
                self.render_investment_advice_page(ticker, info)
                
            elif page == "公共市场":
                self.render_market_page()
            
            # 通用收藏功能
            if page != "公共市场" and info:
                if st.button("📌 收藏", key="main_favorite"):
                    if ticker not in st.session_state.watchlist:
                        st.session_state.watchlist.append(ticker)
                        st.success("✅ 收藏成功！")
                    else:
                        st.info("ℹ️ 已在收藏列表中")
                        
        except Exception as e:
            st.error(f"应用运行出错: {e}")
            logger.error(f"应用运行出错: {e}")

# 主程序入口
def main():
    """主程序"""
    try:
        analyzer = StockAnalyzer()
        analyzer.run()
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        logger.error(f"应用启动失败: {e}")

if __name__ == "__main__":
    main()
