import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

st.set_page_config(page_title="股票分析MVP", layout="wide")

st.sidebar.title("股票分析器")
st.sidebar.markdown("支持港股：输入如0700（自动加.HK）")
ticker_input = st.sidebar.text_input("输入股票代码 (例如, AAPL 或 0700)", value="AAPL").upper()

# 自动添加.HK for港股
if ticker_input.isdigit() and 1 <= len(ticker_input) <= 5 and not ticker_input.endswith('.HK'):
    ticker = ticker_input + '.HK'
else:
    ticker = ticker_input

# Watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if st.button("添加到watchlist"):
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.success("添加成功！")
st.sidebar.subheader("Watchlist")
for wl_ticker in st.session_state.watchlist:
    st.sidebar.text(wl_ticker)
    if st.sidebar.button(f"移除 {wl_ticker}", key=wl_ticker):
        st.session_state.watchlist.remove(wl_ticker)
        st.rerun()

@st.cache_data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        recommendations = stock.recommendations_summary if not stock.recommendations.empty else pd.DataFrame()
        return info, recommendations
    except Exception as e:
        st.error(f"数据拉取失败: {e}. 请检查代码或网络。")
        return {}, pd.DataFrame()

@st.cache_data
def get_historical_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if not hist.empty:
            return hist
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"历史数据拉取失败: {e}.")
        return pd.DataFrame()

def get_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        one_month_ago = datetime.now() - timedelta(days=30)
        filtered_news = [n for n in news if 'publish_date' in n and datetime.fromtimestamp(n['publish_date']) >= one_month_ago]
        filtered_news.sort(key=lambda x: x.get('publish_date', 0), reverse=True)
        if filtered_news:
            return filtered_news[:5]
    except:
        pass
    # Fallback to Investing.com
    try:
        url = f"https://cn.investing.com/equities/{ticker}-news"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('article', class_='js-article-item')
        news_list = []
        for item in news_items[:5]:
            title_tag = item.find('a', class_='title')
            title = title_tag.text.strip() if title_tag else ''
            link = 'https://cn.investing.com' + title_tag['href'] if title_tag else ''
            date_tag = item.find('time')
            date = date_tag['datetime'] if date_tag else ''
            if title and datetime.fromisoformat(date.replace('Z', '+00:00')) >= one_month_ago:
                news_list.append({'title': title, 'link': link, 'publish_date': date})
        return news_list
    except:
        return []

def get_fed_rate():
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomc.htm"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        rate_text = soup.find(string=lambda text: "Target range for the federal funds rate" in text if text else None)
        if rate_text:
            return rate_text.parent.find_next_sibling('p').text.strip()
        else:
            return "无法获取Fed利率。"
    except:
        return "无法获取Fed利率。"

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

def calculate_bollinger_bands(close, window=20, std_dev=2):
    try:
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band
    except:
        return pd.Series(), pd.Series(), pd.Series()

def calculate_stochastic(close, high, low, period=14):
    try:
        l14 = low.rolling(window=period).min()
        h14 = high.rolling(window=period).max()
        k = 100 * ((close - l14) / (h14 - l14))
        return k.iloc[-1] if not k.empty else 50
    except:
        return 50

def backtest_ma_crossover(hist):
    try:
        ma5 = hist['Close'].rolling(5).mean()
        ma20 = hist['Close'].rolling(20).mean()
        signals = pd.DataFrame(index=hist.index)
        signals['signal'] = 0.0
        signals['signal'][ma5 > ma20] = 1.0
        signals['signal'][ma5 < ma20] = -1.0
        signals['positions'] = signals['signal'].diff()
        returns = hist['Close'].pct_change() * signals['signal'].shift(1)
        win_rate = (returns > 0).mean() * 100 if not returns.empty else 0
        return win_rate
    except:
        return 0

def get_x_sentiment(ticker):
    try:
        # 模拟x_semantic_search
        sentiment = "中性"  # placeholder
        return sentiment
    except:
        return "中性"

pages = ["首页", "基本面", "警报", "投资建议"]
page = st.sidebar.radio("导航", pages)

info, news, rec = get_stock_data(ticker)

currency = info.get('currency', 'USD')  # 自动货币
company_name = info.get('longName', ticker) or ticker

if page == "首页":
    st.title(f"{company_name} ({ticker}) 股票仪表板")
    period_options = {
        "1日": "1d",
        "5日": "5d",
        "日K": "1mo",
        "周K": "3mo",
        "月K": "1y",
        "季K": "5y"
    }
    default_index = list(period_options.keys()).index("日K")
    selected_label = st.selectbox("选择时间范围", list(period_options.keys()), index=default_index)
    selected_period = period_options[selected_label]
    hist = get_historical_data(ticker, selected_period)
    if not hist.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                            open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'],
                                            name='K线')])
        ma5 = hist['Close'].rolling(window=5).mean()
        ma20 = hist['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma5, mode='lines', name='MA5 (短期)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist.index, y=ma20, mode='lines', name='MA20 (长期)', line=dict(color='orange')))
        
        if st.checkbox("显示布林带"):
            upper, middle, lower = calculate_bollinger_bands(hist['Close'])
            fig.add_trace(go.Scatter(x=hist.index, y=upper, mode='lines', name='Upper BB', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=hist.index, y=lower, mode='lines', name='Lower BB', line=dict(color='green', dash='dash')))
        
        if st.checkbox("显示随机振荡器"):
            stochastic = calculate_stochastic(hist['Close'], hist['High'], hist['Low'])
            st.text(f"Stochastic: {stochastic:.2f}")

        fig.update_layout(title=f"{ticker} {selected_label}K线图（可拖拽查看细节）", xaxis_title="日期", yaxis_title="价格",
                          xaxis_rangeslider_visible=True, xaxis_tickformat='%Y年%m月%d日')
        st.plotly_chart(fig, use_container_width=True)
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
            st.session_state.prev_current = info.get('currentPrice', 0)
            st.session_state.prev_pre = info.get('preMarketPrice', 0)
            st.session_state.prev_post = info.get('postMarketPrice', 0)
        
        current_price = info.get('currentPrice', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        col1, col2, col3 = st.columns(3)
        col1.metric("当前价格", f"{current_price:.2f} {currency}" if isinstance(current_price, (int, float)) else current_price)
        col2.metric("今日最高", f"{day_high:.2f} {currency}" if isinstance(day_high, (int, float)) else day_high)
        col3.metric("今日最低", f"{day_low:.2f} {currency}" if isinstance(day_low, (int, float)) else day_low)
        
        if currency == 'USD':  # 美股
            st.subheader("盘前/盘后实时交易")
            if st.button("手动刷新"):
                with st.spinner('刷新中...'):
                    time.sleep(1)  # 模拟延迟
                    try:
                        new_info = yf.Ticker(ticker).info
                        st.session_state.prev_current = current_price
                        st.session_state.prev_pre = info.get('preMarketPrice', 0)
                        st.session_state.prev_post = info.get('postMarketPrice', 0)
                        info = new_info
                        st.success("刷新成功！")
                    except:
                        st.error("刷新失败，请重试。")
                st.session_state.last_refresh = time.time()
                st.rerun()
            
            last_refresh_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_refresh))
            st.text(f"最后刷新时间: {last_refresh_time}")
            
            pre_market_price = info.get('preMarketPrice', 'N/A')
            pre_market_change = info.get('preMarketChange', 0)
            post_market_price = info.get('postMarketPrice', 'N/A')
            post_market_change = info.get('postMarketChange', 0)
            current_change = current_price - st.session_state.prev_current if isinstance(current_price, (int, float)) else 0
            
            pre_delta_color = "normal" if pre_market_change >= 0 else "inverse"
            post_delta_color = "normal" if post_market_change >= 0 else "inverse"
            current_delta_color = "normal" if current_change >= 0 else "inverse"
            
            col4, col5, col6, col7 = st.columns(4)
            col4.metric("盘前价格", f"{pre_market_price:.2f} {currency}" if isinstance(pre_market_price, (int, float)) else pre_market_price)
            col5.metric("盘前变化", f"{pre_market_change:.2f}" if isinstance(pre_market_change, (int, float)) else pre_market_change, delta_color=pre_delta_color)
            col6.metric("盘后价格", f"{post_market_price:.2f} {currency}" if isinstance(post_market_price, (int, float)) else post_market_price)
            col7.metric("盘后变化", f"{post_market_change:.2f}" if isinstance(post_market_change, (int, float)) else post_market_change, delta_color=post_delta_color)
            
            # 自动刷新每60s
            time.sleep(60)
            st.rerun()
    else:
        st.error("无历史数据可用。")

elif page == "基本面":
    st.title(f"{company_name} ({ticker}) 基本面")
    if info:
        hist = get_historical_data(ticker, "1mo")
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        avg_volume = hist['Volume'].mean() if 'Volume' in hist else 'N/A'
        # Sharpe Ratio
        returns = hist['Close'].pct_change()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 'N/A'
        
        df = pd.DataFrame({
            "指标": ["市值", "市盈率 (PE)", "每股收益 (EPS)", "股息收益率", "Beta", "ROE", "负债权益比", "RSI (14日)", "MACD", "平均成交量", "Sharpe Ratio (风险回报)"],
            "值": [info.get('marketCap', 'N/A'),
                   info.get('trailingPE', 'N/A'),
                   info.get('trailingEps', 'N/A'),
                   info.get('dividendYield', 'N/A'),
                   info.get('beta', 'N/A'),
                   info.get('returnOnEquity', 'N/A'),
                   info.get('debtToEquity', 'N/A'),
                   rsi,
                   f"{macd:.2f} (Signal: {signal:.2f})",
                   f"{avg_volume:,.0f}",
                   f"{sharpe:.2f}" if isinstance(sharpe, float) else sharpe]
        })
        st.table(df)
    else:
        st.error("无基本面数据。")

elif page == "警报":
    st.title("价格警报")
    st.markdown("""
    **用法说明**：设置价格阈值和类型（低于/高于），app每分钟自动检查当前价格。如果触发阈值，会显示警告通知（页面弹窗）。通知方式：实时页面警告（红色），支持手动刷新检查。设置后保存，监控即时生效。
    """)
    alert_type = st.selectbox("警报类型", ["低于阈值", "高于阈值"])
    threshold = st.number_input("设置阈值", value=100.0)
    if st.button("保存设置"):
        st.session_state.alert_type = alert_type
        st.session_state.threshold = threshold
        st.success("设置保存！开始监控。")
    
    if 'alert_type' in st.session_state:
        current_price = info.get('currentPrice', 0)
        if (st.session_state.alert_type == "低于阈值" and current_price < st.session_state.threshold) or (st.session_state.alert_type == "高于阈值" and current_price > st.session_state.threshold):
            st.warning(f"警报触发: {ticker} 价格 {current_price:.2f} {currency} {st.session_state.alert_type} {st.session_state.threshold}!")
        else:
            st.success(f"{ticker} 价格正常。")
        
        # 自动刷新每60s
        time.sleep(60)
        st.rerun()

elif page == "投资建议":
    st.title(f"{company_name} ({ticker}) 当天投资建议 (2025-07-28)")
    if info:
        hist = get_historical_data(ticker, "1mo")
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
        
        if news:
            st.subheader("最新资讯（影响情绪）")
            for item in news:
                title = item.get('title', '')
                link = item.get('link', '')
                date = item.get('publish_date', '')
                if title:
                    st.markdown(f"- [{title}]({link}) ({date})")
    else:
        st.error("请输入股票代码。")
