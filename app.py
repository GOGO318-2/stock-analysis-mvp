import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import datetime
import yfinance as yf  # 新增fallback

# 你的API Key
FINNHUB_KEY = 'd1p1qv9r01qi9vk2517gd1p1qv9r01qi9vk25180'
FMP_KEY = '8n2nsHP2Lj1uHkPRrtcQ8a63Lf95VjbU'
POLYGON_KEY = '2CDgF277xEhkhKndj5yFMVONxBGFFShg'  # 备用

st.set_page_config(page_title="股票分析MVP", layout="wide")

st.sidebar.title("股票分析器")
st.sidebar.markdown("支持港股：输入如'0700.HK'")
ticker = st.sidebar.text_input("输入股票代码 (例如, AAPL 或 0700.HK)", value="AAPL").upper()

@st.cache_data
def get_real_time_quote(ticker):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
        return requests.get(url).json()
    except Exception as e:
        st.error(f"报价API失败: {e}. 检查Key或限额。")
        return {}  # 空返回

@st.cache_data
def get_fundamentals(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_KEY}"
        return requests.get(url).json()[0] if requests.get(url).json() else {}
    except Exception as e:
        st.error(f"基本面API失败: {e}.")
        return {}

@st.cache_data
def get_key_metrics(ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&apikey={FMP_KEY}"
        return requests.get(url).json()[0] if requests.get(url).json() else {}
    except Exception as e:
        st.error(f"指标API失败: {e}.")
        return {}

@st.cache_data
def get_historical_data(ticker, days=30):
    try:
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=days)
        from_unix = int(datetime.datetime.combine(from_date, datetime.time()).timestamp())
        to_unix = int(datetime.datetime.combine(to_date, datetime.time()).timestamp())
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={from_unix}&to={to_unix}&token={FINNHUB_KEY}"
        response = requests.get(url).json()
        if 'c' in response:
            df = pd.DataFrame({
                '日期': pd.to_datetime(response['t'], unit='s'),
                '开盘': response['o'],
                '最高': response['h'],
                '最低': response['l'],
                '收盘': response['c'],
                '成交量': response['v']
            })
            df.set_index('日期', inplace=True)
            return df
        else:
            raise ValueError("无历史数据。")
    except Exception as e:
        st.error(f"Finnhub历史API失败: {e}. 使用yfinance fallback。")
        # Fallback to yfinance
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False)
            if not df.empty:
                df = df.rename(columns={'Open': '开盘', 'High': '最高', 'Low': '最低', 'Close': '收盘', 'Volume': '成交量'})
                df.index.name = '日期'
                return df
            else:
                return pd.DataFrame()
        except Exception as yf_e:
            st.error(f"yfinance fallback失败: {yf_e}.")
            return pd.DataFrame()

@st.cache_data
def get_news(ticker):
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-07-27&to=2025-07-28&token={FINNHUB_KEY}"
        return requests.get(url).json()[:3]
    except Exception as e:
        st.error(f"新闻API失败: {e}.")
        return []

@st.cache_data
def get_recommendations(ticker):
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_KEY}"
        return requests.get(url).json()[0] if requests.get(url).json() else {}
    except Exception as e:
        st.error(f"推荐API失败: {e}.")
        return {}

def calculate_rsi(close, period=14):
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    except:
        return 50

def calculate_macd(close, short=12, long=26, signal=9):
    try:
        ema_short = close.ewm(span=short, adjust=False).mean()
        ema_long = close.ewm(span=long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1] if not macd_line.empty else (0, 0)
    except:
        return 0, 0

pages = ["首页", "基本面", "警报", "投资建议"]
page = st.sidebar.radio("导航", pages)

if page == "首页":
    st.title(f"{ticker} 股票仪表板")
    if ticker:
        hist = get_historical_data(ticker, days=5)
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                open=hist['开盘'], high=hist['最高'],
                                                low=hist['最低'], close=hist['收盘'],
                                                name='K线')])
            ma5 = hist['收盘'].rolling(window=5).mean()
            ma20 = hist['收盘'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5 (短期)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20 (长期)', line=dict(color='orange')))
            fig.update_layout(title=f"{ticker} 本周K线图（可拖拽查看细节）", xaxis_title="日期", yaxis_title="价格",
                              xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')
            st.plotly_chart(fig, use_container_width=True)
            
            quote = get_real_time_quote(ticker)
            col1, col2, col3 = st.columns(3)
            col1.metric("当前价格", f"${quote.get('c', 'N/A'):.2f}")
            col2.metric("今日最高", f"${quote.get('h', 'N/A'):.2f}")
            col3.metric("今日最低", f"${quote.get('l', 'N/A'):.2f}")
        else:
            st.error("无历史数据可用。")

elif page == "基本面":
    st.title(f"{ticker} 基本面")
    if ticker:
        fundamentals = get_fundamentals(ticker)
        metrics = get_key_metrics(ticker)
        hist = get_historical_data(ticker)
        if fundamentals or metrics:
            rsi = calculate_rsi(hist['收盘'])
            macd, signal = calculate_macd(hist['收盘'])
            avg_volume = hist['成交量'].mean() if '成交量' in hist else 'N/A'
            
            eps = metrics.get('netIncomePerShare', fundamentals.get('eps', 'N/A'))
            dividend_yield = metrics.get('dividendYield', fundamentals.get('dividendYield', 'N/A'))
            beta = metrics.get('beta', fundamentals.get('beta', 'N/A'))
            roe = metrics.get('returnOnEquity', 'N/A')
            
            df = pd.DataFrame({
                "指标": ["市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", "Beta", "ROE", "负债权益比", "RSI (14日)", "MACD", "平均成交量"],
                "值": [fundamentals.get('mktCap', 'N/A'),
                       metrics.get('peRatio', 'N/A'),
                       eps,
                       dividend_yield,
                       beta,
                       roe,
                       metrics.get('debtToEquity', 'N/A'),
                       rsi,
                       f"{macd:.2f} (Signal: {signal:.2f})",
                       f"{avg_volume:,.0f}"]
            })
            st.table(df)
        else:
            st.error("无基本面数据。")

elif page == "警报":
    st.title("价格警报")
    threshold = st.number_input("设置警报阈值 (例如, 价格低于)", value=100.0)
    if ticker:
        quote = get_real_time_quote(ticker)
        current_price = quote.get('c', 0)
        if current_price < threshold:
            st.warning(f"警报: {ticker} 价格 ${current_price:.2f} 低于 {threshold}!")
        else:
            st.success(f"{ticker} 价格高于阈值。")

elif page == "投资建议":
    st.title(f"{ticker} 当天投资建议 (2025-07-28)")
    if ticker:
        quote = get_real_time_quote(ticker)
        fundamentals = get_fundamentals(ticker)
        metrics = get_key_metrics(ticker)
        hist = get_historical_data(ticker)
        news = get_news(ticker)
        rec = get_recommendations(ticker)
        
        current_price = quote.get('c', 0)
        pe = metrics.get('peRatio', 0)
        eps = metrics.get('netIncomePerShare', 0)
        rsi = calculate_rsi(hist['收盘'])
        macd, _ = calculate_macd(hist['收盘'])
        buy_rating = rec.get('buy', 0)
        sell_rating = rec.get('sell', 0)
        target_price = rec.get('targetPrice', current_price * 1.1)
        support = current_price * 0.95
        resistance = current_price * 1.05
        news_sentiment = "正面" if any('positive' in item.get('headline', '').lower() for item in news) else ("负面" if any('negative' in item.get('headline', '').lower() for item in news) else "中性")
        
        short_memo = f"RSI {rsi:.0f}表示{('超卖反弹' if rsi < 40 else '超买回调' if rsi > 60 else '稳定波动')}，关注成交量放大，风险{news_sentiment}情绪。"
        trend_memo = f"PE {pe:.1f}支持长期{('增长' if buy_rating > sell_rating else '谨慎')}，ROE稳定，持仓3-6月忽略波动。"
        swing_memo = f"MACD {macd:.2f}交叉，波段捕捉{('上涨' if macd > 0 else '下行')}机会，分批操作，X情绪{news_sentiment}。"
        
        short_trigger = f"RSI<40且MACD金叉" if rsi < 50 else f"RSI>60或MA5死叉"
        trend_trigger = f"MA20上穿且ROE>20%" if buy_rating > sell_rating else f"跌破支持位{support:.0f}"
        swing_trigger = f"MACD正向交叉且成交量>平均" if macd > 0 else f"突破阻力{resistance:.0f}或跌破MA20"
        short_pos = "60%" if rsi < 40 else "减仓40%"
        trend_pos = "加仓40%" if buy_rating > sell_rating else "清仓"
        swing_pos = "70%" if macd > 0 else "减仓50%"
        
        data = [
            {"阶段": "短期交易 (日内/短期)", "时机": "入场", "价位": f"{support:.0f}-{current_price:.0f}", "触发电号": short_trigger, "仓位": short_pos, "备忘": short_memo},
            {"阶段": "短期交易 (日内/短期)", "时机": "止盈", "价位": f"{resistance:.0f}", "触发电号": short_trigger, "仓位": short_pos, "备忘": short_memo},
            {"阶段": "趋势交易 (长期)", "时机": "入场", "价位": f"{support:.0f}-{current_price:.0f}", "触发电号": trend_trigger, "仓位": trend_pos, "备忘": trend_memo},
            {"阶段": "趋势交易 (长期)", "时机": "止损", "价位": f"{support * 0.95:.0f}", "触发电号": trend_trigger, "仓位": trend_pos, "备忘": trend_memo},
            {"阶段": "波段交易 (中短期)", "时机": "入场", "价位": f"{support:.0f}-{resistance:.0f}", "触发电号": swing_trigger, "仓位": swing_pos, "备忘": swing_memo},
            {"阶段": "波段交易 (中短期)", "时机": "止盈/止损", "价位": f"{target_price:.0f} / {support:.0f}", "触发电号": swing_trigger, "仓位": swing_pos, "备忘": swing_memo}
        ]
        df = pd.DataFrame(data)
        st.table(df)
        
        st.markdown(f"<span style='color:red'>重点: RSI{rsi:.0f}建议{('买入' if rsi < 40 else '卖出' if rsi > 60 else '持仓')}，目标{target_price:.0f}。非投资建议。</span>", unsafe_allow_html=True)
        
        st.subheader("最新新闻（影响情绪）")
        for item in news:
            st.write(f"- {item.get('headline', '无标题')} ({item.get('datetime', '')})")
    else:
        st.error("请输入股票代码。")
