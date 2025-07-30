import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import time
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'page_title': '智能股票分析平台',
    'layout': 'wide',
    'api_keys': {
        "finnhub": "d1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180",
        "alpha_vantage": "Z45S0SLJGM378PIO", 
        "polygon": "2CDgF277xEhkhKndj5yFMVONxBGFFShg",
        "xai": "xai-N36diIqx3wkZz6eBGQfjadqdNe3H84FYfPsXXauU02ag1s5k45zida3aYocHu5Bi9AhT6jO5kFpjW7CD"
    },
    'tech_stocks': [
        'NVDA', 'TSLA', 'AMD', 'GOOG', 'AAPL', 'MSFT', 'AMZN', 'META', 'INTC', 'QCOM',
        'IBM', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'NFLX', 'BABA', 'TSM', 'ASML'
    ],
    'cache_timeout': 300  # 5 minutes
}

# Setup
st.set_page_config(page_title=CONFIG['page_title'], layout=CONFIG['layout'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """统一的股票数据获取类"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'StockAnalyzer/1.0'})
    
    @st.cache_data(ttl=CONFIG['cache_timeout'])
    def get_stock_info(_self, ticker: str) -> Tuple[Dict, pd.DataFrame]:
        """获取股票基本信息和推荐"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 获取推荐信息（如果可用）
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
    def get_historical_data(_self, ticker: str, period: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist if not hist.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"获取历史数据失败 {ticker}: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CONFIG['cache_timeout'])
    def get_news(_self, ticker: str) -> List[Dict]:
        """获取股票新闻"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:5]
            
            news_list = []
            positive_keywords = ['positive', 'bullish', 'surge', 'gain', 'up', 'buy', 'growth', 'strong']
            negative_keywords = ['negative', 'bearish', 'drop', 'loss', 'down', 'sell', 'decline', 'weak']
            
            for item in news:
                title = item.get('title', '')
                title_lower = title.lower()
                
                # 简单情感分析
                if any(kw in title_lower for kw in positive_keywords):
                    sentiment = "正面"
                elif any(kw in title_lower for kw in negative_keywords):
                    sentiment = "负面"
                else:
                    sentiment = "中性"
                
                news_list.append({
                    'title': title,
                    'link': item.get('link', ''),
                    'publish_date': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M'),
                    'sentiment': sentiment,
                    'source': item.get('publisher', {}).get('name', 'Unknown')
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"获取新闻失败 {ticker}: {e}")
            return []

class TechnicalAnalyzer:
    """技术分析工具类"""
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        if len(close) < period:
            return 50.0
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[float, float]:
        """计算MACD指标"""
        if len(close) < long:
            return 0.0, 0.0
        
        ema_short = close.ewm(span=short).mean()
        ema_long = close.ewm(span=long).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.iloc[-1], signal_line.iloc[-1]
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        if len(close) < window:
            return pd.Series(), pd.Series(), pd.Series()
        
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        return upper_band, rolling_mean, lower_band
    
    @staticmethod
    def calculate_support_resistance(close: pd.Series) -> Tuple[float, float]:
        """计算支撑位和阻力位"""
        if len(close) < 20:
            current_price = close.iloc[-1]
            return current_price * 0.95, current_price * 1.05
        
        # 使用最近20天的数据
        recent_data = close.tail(20)
        support = recent_data.min()
        resistance = recent_data.max()
        
        return support, resistance

class AIAnalyzer:
    """AI分析工具类"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    
    @st.cache_data(ttl=600)  # 10分钟缓存
    def get_sentiment(_self, ticker: str) -> str:
        """获取情感分析"""
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {_self.api_key}",
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
            
            response = _self.session.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content'].strip()
                # 标准化回答
                if any(word in result for word in ['正面', 'positive', '看涨', '乐观']):
                    return "正面"
                elif any(word in result for word in ['负面', 'negative', '看跌', '悲观']):
                    return "负面"
                else:
                    return "中性"
            
        except Exception as e:
            logger.error(f"AI情感分析失败 {ticker}: {e}")
        
        return "中性"
    
    @st.cache_data(ttl=600)
    def get_investment_advice(_self, ticker: str, rsi: float, macd: float) -> str:
        """获取投资建议"""
        try:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {_self.api_key}",
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
            
            response = _self.session.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"AI投资建议失败 {ticker}: {e}")
        
        return "暂无建议"

class StockAnalyzerUI:
    """股票分析界面类"""
    
    def __init__(self):
        self.data_fetcher = StockDataFetcher(CONFIG['api_keys'])
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_analyzer = AIAnalyzer(CONFIG['api_keys']['xai'])
        self.setup_sidebar()
    
    def setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.title("🚀 智能股票分析")
        st.sidebar.markdown("---")
        
        # 股票代码输入
        ticker_input = st.sidebar.text_input(
            "输入股票代码", 
            value="TSLA", 
            help="例如: TSLA (美股) 或 0700 (港股)"
        ).upper()
        
        # 处理港股代码
        self.ticker = self.process_ticker(ticker_input)
        
        # Watchlist管理
        self.setup_watchlist()
        
        # 页面导航
        st.sidebar.markdown("---")
        self.page = st.sidebar.radio(
            "📋 导航菜单", 
            ["📊 实时数据", "📈 技术分析", "🎯 投资建议", "🌟 热门股票", "📰 市场新闻"]
        )
    
    def process_ticker(self, ticker_input: str) -> str:
        """处理股票代码"""
        if ticker_input.isdigit():
            ticker_clean = ticker_input.lstrip('0')
            if 1 <= len(ticker_clean) <= 5:
                return ticker_clean + '.HK'
        return ticker_input
    
    def setup_watchlist(self):
        """设置关注列表"""
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        
        st.sidebar.markdown("### ⭐ 关注列表")
        
        if st.sidebar.button("➕ 添加到关注列表"):
            if self.ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(self.ticker)
                st.sidebar.success("添加成功！")
            else:
                st.sidebar.warning("已在关注列表中")
        
        # 显示关注列表
        for i, wl_ticker in enumerate(st.session_state.watchlist):
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            col1.text(wl_ticker)
            
            if col2.button("📊", key=f"view_{i}", help="查看"):
                self.ticker = wl_ticker
                st.rerun()
            
            if col3.button("🗑️", key=f"remove_{i}", help="移除"):
                st.session_state.watchlist.remove(wl_ticker)
                st.rerun()
    
    def run(self):
        """运行主应用"""
        if self.page == "📊 实时数据":
            self.render_realtime_page()
        elif self.page == "📈 技术分析":
            self.render_technical_page()
        elif self.page == "🎯 投资建议":
            self.render_advice_page()
        elif self.page == "🌟 热门股票":
            self.render_trending_page()
        elif self.page == "📰 市场新闻":
            self.render_news_page()
    
    def render_realtime_page(self):
        """渲染实时数据页面"""
        info, _ = self.data_fetcher.get_stock_info(self.ticker)
        
        if not info:
            st.error("❌ 无法获取股票数据，请检查股票代码")
            return
        
        company_name = info.get('longName', self.ticker)
        currency = info.get('currency', 'USD')
        
        # 页面标题
        st.title(f"📊 {company_name} ({self.ticker})")
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', current_price)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0
        
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
        
        # 时间范围选择和K线图
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
        
        self.render_chart(period_options[selected_period])
        
        # 盘前盘后数据（仅美股）
        if currency == 'USD':
            self.render_extended_hours(info, currency)
    
    def render_chart(self, period: str):
        """渲染K线图"""
        hist = self.data_fetcher.get_historical_data(self.ticker, period)
        
        if hist.empty:
            st.warning("⚠️ 无法获取历史数据")
            return
        
        fig = go.Figure()
        
        # K线图
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='K线'
        ))
        
        # 移动平均线
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
            
            # 布林带
            upper, middle, lower = self.technical_analyzer.calculate_bollinger_bands(hist['Close'])
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
            title=f"{self.ticker} K线图表",
            xaxis_title="时间",
            yaxis_title="价格",
            xaxis_rangeslider_visible=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_extended_hours(self, info: Dict, currency: str):
        """渲染盘前盘后数据"""
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
    
    def render_technical_page(self):
        """渲染技术分析页面"""
        st.title(f"📈 {self.ticker} 技术分析")
        
        info, _ = self.data_fetcher.get_stock_info(self.ticker)
        hist = self.data_fetcher.get_historical_data(self.ticker, "1y")
        
        if hist.empty:
            st.error("❌ 无法获取历史数据")
            return
        
        # 计算技术指标
        rsi = self.technical_analyzer.calculate_rsi(hist['Close'])
        macd, signal = self.technical_analyzer.calculate_macd(hist['Close'])
        support, resistance = self.technical_analyzer.calculate_support_resistance(hist['Close'])
        
        # 风险指标
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # 技术指标表格
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
        
        # RSI图表
        st.subheader("📈 RSI 趋势")
        if len(hist) >= 14:
            rsi_values = []
            for i in range(14, len(hist)):
                rsi_val = self.technical_analyzer.calculate_rsi(hist['Close'].iloc[:i+1])
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
            fig.update_layout(title="RSI指标趋势", yaxis_title="RSI值", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_advice_page(self):
        """渲染投资建议页面"""
        st.title(f"🎯 {self.ticker} 投资建议")
        
        info, _ = self.data_fetcher.get_stock_info(self.ticker)
        hist = self.data_fetcher.get_historical_data(self.ticker, "3mo")
        
        if hist.empty:
            st.error("❌ 无法获取数据生成建议")
            return
        
        # 计算指标
        rsi = self.technical_analyzer.calculate_rsi(hist['Close'])
        macd, signal = self.technical_analyzer.calculate_macd(hist['Close'])
        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        support, resistance = self.technical_analyzer.calculate_support_resistance(hist['Close'])
        
        # AI分析
        with st.spinner("🤖 AI分析中..."):
            sentiment = self.ai_analyzer.get_sentiment(self.ticker)
            ai_advice = self.ai_analyzer.get_investment_advice(self.ticker, rsi, macd)
        
        # 综合评分
        score = 0
        if rsi < 30: score += 2  # 超卖加分
        elif rsi > 70: score -= 2  # 超买减分
        
        if macd > signal: score += 1  # MACD看涨
        else: score -= 1
        
        if sentiment == "正面": score += 1
        elif sentiment == "负面": score -= 1
        
        # 建议等级
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
        
        # 显示建议
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("综合评分", f"{score}/5")
        
        with col2:
            st.markdown(f"### 投资建议: <span style='color:{color}'>{recommendation}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.metric("市场情绪", sentiment)
        
        # 详细建议表
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
        
        # AI建议
        st.markdown("---")
        st.subheader("🤖 AI 深度分析")
        st.info(ai_advice)
        
        # 风险提示
        st.markdown("---")
        st.warning("⚠️ 风险提示：以上建议仅供参考，投资有风险，入市需谨慎！")
    
    def render_trending_page(self):
        """渲染热门股票页面"""
        st.title("🌟 热门科技股推荐")
        
        if st.button("🔄 更新数据", type="primary"):
            with st.spinner("正在获取最新数据..."):
                trending_data = self.get_trending_stocks()
                st.session_state['trending_data'] = trending_data
                st.success("数据更新完成！")
        
        if 'trending_data' in st.session_state:
            df = st.session_state['trending_data']
            
            # 添加筛选选项
            col1, col2 = st.columns(2)
            with col1:
                min_price = st.slider("最低价格", 0, 1000, 0)
            with col2:
                sentiment_filter = st.selectbox("情绪筛选", ["全部", "正面", "中性", "负面"])
            
            # 应用筛选
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
            st.info("点击"更新数据"获取最新热门股票信息")
    
    def get_trending_stocks(self) -> pd.DataFrame:
        """获取热门股票数据"""
        stock_data = []
        
        for ticker in CONFIG['tech_stocks'][:15]:  # 限制数量提高性能
            try:
                info, _ = self.data_fetcher.get_stock_info(ticker)
                if not info:
                    continue
                
                hist = self.data_fetcher.get_historical_data(ticker, "1mo")
                if hist.empty:
                    continue
                
                current_price = info.get('currentPrice', 0)
                previous_close = info.get('previousClose', current_price)
                change_percent = ((current_price - previous_close) / previous_close * 100) if previous_close else 0
                
                # 简化版情绪分析
                sentiment = "中性"  # 默认值，可选择性启用AI分析
                
                # 计算活跃度评分
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', volume)
                volume_ratio = (volume / avg_volume) if avg_volume else 1
                
                activity_score = min(5, volume_ratio + (1 if change_percent > 2 else -1 if change_percent < -2 else 0))
                buy_level = "高" if activity_score > 3 else "中" if activity_score > 1.5 else "低"
                
                stock_data.append({
                    '股票代码': ticker,
                    '公司名称': info.get('longName', ticker)[:20] + '...' if len(info.get('longName', ticker)) > 20 else info.get('longName', ticker),
                    '当前价格': current_price,
                    '涨跌幅': change_percent,
                    '成交量': volume,
                    '市值': info.get('marketCap', 0),
                    '市场情绪': sentiment,
                    '推荐等级': buy_level,
                    'P/E比率': info.get('trailingPE', 'N/A'),
                    '52周最高': info.get('fiftyTwoWeekHigh', 'N/A'),
                    '52周最低': info.get('fiftyTwoWeekLow', 'N/A')
                })
                
            except Exception as e:
                logger.error(f"获取股票数据失败 {ticker}: {e}")
                continue
        
        if stock_data:
            df = pd.DataFrame(stock_data)
            # 按推荐等级和涨跌幅排序
            level_order = {'高': 3, '中': 2, '低': 1}
            df['_sort_key'] = df['推荐等级'].map(level_order)
            df = df.sort_values(['_sort_key', '涨跌幅'], ascending=[False, False])
            df = df.drop('_sort_key', axis=1)
            return df
        
        return pd.DataFrame()
    
    def render_news_page(self):
        """渲染市场新闻页面"""
        st.title(f"📰 {self.ticker} 市场新闻")
        
        with st.spinner("获取最新新闻..."):
            news_list = self.data_fetcher.get_news(self.ticker)
        
        if not news_list:
            st.warning("暂无相关新闻")
            return
        
        # 情绪统计
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
        
        # 新闻列表
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

class PerformanceOptimizer:
    """性能优化工具"""
    
    @staticmethod
    def optimize_dataframe_display(df: pd.DataFrame, max_rows: int = 50) -> pd.DataFrame:
        """优化DataFrame显示"""
        if len(df) > max_rows:
            return df.head(max_rows)
        return df
    
    @staticmethod
    def batch_api_calls(tickers: List[str], batch_size: int = 5) -> List[List[str]]:
        """批量API调用优化"""
        return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

def main():
    """主函数"""
    try:
        # 页面头部
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='margin: 0; color: white;'>🚀 智能股票分析平台</h1>
            <p style='margin: 0; opacity: 0.9;'>基于AI的实时股票分析与投资建议系统</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 初始化应用
        app = StockAnalyzerUI()
        app.run()
        
        # 页面底部
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            <p>⚠️ 免责声明：本系统提供的所有信息仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
            <p>💡 数据来源：Yahoo Finance、AI分析 | 更新频率：实时</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"应用启动失败: {e}")
        logger.error(f"应用启动失败: {e}")

# 添加自定义CSS样式
def inject_custom_css():
    """注入自定义CSS样式"""
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 0.5rem;
    }
    
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stExpander {
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* 隐藏Streamlit默认样式 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .stColumns {
            flex-direction: column;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 启动应用
if __name__ == "__main__":
    inject_custom_css()
    main()
