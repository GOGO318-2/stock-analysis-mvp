import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf

st.set_page_config(page_title="股票分析MVP", layout="wide")

st.sidebar.title("股票分析器")
st.sidebar.markdown("支持港股：输入如0700（自动加.HK）")
ticker_input = st.sidebar.text_input("输入股票代码 (例如, AAPL 或 0700)", value="AAPL").upper()

# 自动添加.HK for港股
if ticker_input.isdigit() and len(ticker_input) == 4:
    ticker = ticker_input + '.HK'
else:
    ticker = ticker_input

@st.cache_data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        news = stock.news[:3]
        recommendations = stock.recommendations_summary if not stock.recommendations.empty else pd.DataFrame()
        return info, hist, news, recommendations
    except Exception as e:
        st.error(f"数据拉取失败: {e}. 请检查代码或网络。")
        return {}, pd.DataFrame(), [], pd.DataFrame()

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

info, hist, news, rec = get_stock_data(ticker)

currency = info.get('currency', 'USD')  # 自动货币

if page == "首页":
    st.title(f"{ticker} 股票仪表板")
    if not hist.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                            open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'],
                                            name='K线')])
        ma5 = hist['Close'].rolling(window=5).mean()
        ma20 = hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5 (短期)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20 (长期)', line=dict(color='orange')))
        fig.update_layout(title=f"{ticker} 本周K线图（可拖拽查看细节）", xaxis_title="日期", yaxis_title="价格",
                          xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')
        st.plotly_chart(fig, use_container_width=True)
        
        current_price = info.get('currentPrice', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        col1, col2, col3 = st.columns(3)
        col1.metric("当前价格", f"{current_price:.2f} {currency}" if isinstance(current_price, (int, float)) else current_price)
        col2.metric("今日最高", f"{day_high:.2f} {currency}" if isinstance(day_high, (int, float)) else day_high)
        col3.metric("今日最低", f"{day_low:.2f} {currency}" if isinstance(day_low, (int, float)) else day_low)
    else:
        st.error("无历史数据可用。")

elif page == "基本面":
    st.title(f"{ticker} 基本面")
    if info:
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        avg_volume = hist['Volume'].mean() if 'Volume' in hist else 'N/A'
        
        df = pd.DataFrame({
            "指标": ["市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", "Beta", "ROE", "负债权益比", "RSI (14日)", "MACD", "平均成交量"],
            "值": [info.get('marketCap', 'N/A'),
                   info.get('trailingPE', 'N/A'),
                   info.get('trailingEps', 'N/A'),
                   info.get('dividendYield', 'N/A'),
                   info.get('beta', 'N/A'),
                   info.get('returnOnEquity', 'N/A'),
                   info.get('debtToEquity', 'N/A'),
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
    current_price = info.get('currentPrice', 0)
    if current_price < threshold:
        st.warning(f"警报: {ticker} 价格 {current_price:.2f} {currency} 低于 {threshold}!")
    else:
        st.success(f"{ticker} 价格高于阈值。")

elif page == "投资建议":
    st.title(f"{ticker} 当天投资建议 (2025-07-28)")
    if info:
        current_price = info.get('currentPrice', 0)
        pe = info.get('trailingPE', 0)
        eps = info.get('trailingEps', 0)
        rsi = calculate_rsi(hist['Close'])
        macd, _ = calculate_macd(hist['Close'])
        buy_rating = rec.get('Buy', 0) if not rec.empty else 0
        sell_rating = rec.get('Sell', 0) if not rec.empty else 0
        target_price = info.get('targetMeanPrice', current_price * 1.1)
        support = current_price * 0.95
        resistance = current_price * 1.05
        news_sentiment = "正面" if news and any('positive' in n.get('title', '').lower() for n in news) else ("负面" if news and any('negative' in n.get('title', '').lower() for n in news) else "中性")
        
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
            title = item.get('title', '无标题')
            link = item.get('link', '')
            date = item.get('publish_date', '')
            st.markdown(f"- [{title}]({link}) ({date})")
    else:
        st.error("请输入股票代码。")
