"""
WarpCrypto-Trader v2: Sosyal Medya Odaklı Kripto Analiz Platformu
Çoklu kripto para desteği ve gelişmiş görselleştirme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Modülleri içe aktar
from data_processor import load_and_process_data, get_latest_data, generate_trading_signals
from strategy_backtester import backtest_strategy, backtest_multiple_strategies
from train_model_fixed import ImprovedCryptoPredictor

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="WarpCrypto-Trader v2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Stilleri
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    h1 {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(90deg, #FFD700 0%, #FF6B6B 50%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5em;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 30px rgba(255, 215, 0, 0.5); }
        to { text-shadow: 0 0 50px rgba(255, 107, 107, 0.8); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(103, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    .gauge-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        padding: 15px;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5);
    }
    
    .strategy-table {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cryptocurrency mapping
CRYPTO_OPTIONS = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD', 
    'Solana (SOL)': 'SOL-USD'
}

CRYPTO_COLORS = {
    'BTC-USD': '#F7931A',  # Bitcoin Orange
    'ETH-USD': '#627EEA',  # Ethereum Blue
    'SOL-USD': '#14F195'   # Solana Green
}

CRYPTO_EMOJIS = {
    'BTC-USD': '₿',
    'ETH-USD': '⟠',
    'SOL-USD': '◉'
}

# Title
st.markdown("# 🚀 WarpCrypto-Trader v2.0")
st.markdown("### 🤖 AI-Powered Multi-Crypto Analysis Platform")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    
    # Cryptocurrency selection
    st.markdown("### 🪙 Select Cryptocurrency")
    selected_crypto_name = st.selectbox(
        "Choose crypto to analyze:",
        options=list(CRYPTO_OPTIONS.keys()),
        index=0
    )
    selected_ticker = CRYPTO_OPTIONS[selected_crypto_name]
    crypto_color = CRYPTO_COLORS[selected_ticker]
    crypto_emoji = CRYPTO_EMOJIS[selected_ticker]
    
    st.info(f"Selected: {crypto_emoji} {selected_crypto_name}")
    
    # Model status check
    st.markdown("### 🧠 Model Status")
    model_file = f"{selected_ticker.lower().replace('-', '_')}_predictor_improved.keras"
    model_exists = os.path.exists(model_file)
    
    if model_exists:
        st.success(f"✅ {selected_crypto_name} model is ready!")
    else:
        st.warning(f"⚠️ {selected_crypto_name} model not trained yet")
        if st.button(f"🚀 Train {selected_crypto_name} Model"):
            with st.spinner(f"Training {selected_crypto_name} model..."):
                os.system(f"python train_model_fixed.py --ticker {selected_ticker}")
                st.rerun()
    
    # Display options
    st.markdown("### 📊 Display Settings")
    show_gauge = st.checkbox("RSI Gauge Indicator", value=True)
    show_strategies = st.checkbox("Strategy Leaderboard", value=True)
    show_confidence = st.checkbox("AI Confidence Score", value=True)
    
    # Time period
    st.markdown("### 📅 Analysis Period")
    period = st.selectbox(
        "Time Range",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    # Auto refresh
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)

# Helper fonksiyonlar
@st.cache_resource(show_spinner=False)
def load_improved_model(ticker):
    """Load improved model"""
    model_file = f"{ticker.lower().replace('-', '_')}_predictor_improved.keras"
    scaler_file = f"{ticker.lower().replace('-', '_')}_feature_scaler_improved.pkl"
    columns_file = f"{ticker.lower().replace('-', '_')}_feature_columns_improved.pkl"
    
    if not os.path.exists(model_file):
        return None, None, None
    
    try:
        model = tf.keras.models.load_model(model_file)
        scaler = joblib.load(scaler_file)
        columns = joblib.load(columns_file)
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data(ttl=300, show_spinner=False)
def load_data_cached(ticker, period):
    """Load data with cache"""
    return load_and_process_data(ticker, period=period)

def create_rsi_gauge(rsi_value):
    """Create RSI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = rsi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "RSI Indicator", 'font': {'size': 20, 'color': 'white'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#00ff00'},
                {'range': [30, 70], 'color': '#ffff00'},
                {'range': [70, 100], 'color': '#ff0000'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': rsi_value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def calculate_ai_confidence(predictions):
    """Calculate AI predictions confidence score"""
    if not predictions or 'predictions' not in predictions:
        return 0
    
    changes = []
    for period in predictions['predictions'].values():
        changes.append(abs(period['change']))
    
    # Standard deviation of changes - low deviation = high confidence
    std_dev = np.std(changes)
    # Confidence score: 0-100 (inverse proportion)
    confidence = max(0, min(100, 100 - std_dev * 10))
    
    return confidence

def run_strategy_comparison(data, ticker_name):
    """Run strategy comparison"""
    strategies = {
        'RSI': lambda d: np.where(d['RSI'] < 30, 1, np.where(d['RSI'] > 70, -1, 0)),
        'MACD': lambda d: np.where(d['MACD'] > d['MACD_Signal'], 1, -1),
        'RSI+MACD': lambda d: (
            np.where(d['RSI'] < 30, 1, np.where(d['RSI'] > 70, -1, 0)) * 0.5 +
            np.where(d['MACD'] > d['MACD_Signal'], 1, -1) * 0.5
        ),
        'Bollinger': lambda d: np.where(d['Close'] < d['BB_Low'], 1, 
                                       np.where(d['Close'] > d['BB_High'], -1, 0))
    }
    
    results = []
    for name, strategy_func in strategies.items():
        try:
            signals = strategy_func(data)
            data['Strategy_Signal'] = signals
            
            # Simple backtest
            returns = data['Returns'].shift(-1)
            strategy_returns = returns * data['Strategy_Signal']
            
            total_return = (1 + strategy_returns).prod() - 1
            win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum() * 100
            max_dd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            
            results.append({
                'Strategy': name,
                'Return (%)': f"{total_return * 100:.2f}",
                'Win Rate (%)': f"{win_rate:.1f}",
                'Max Drawdown (%)': f"{max_dd * 100:.2f}"
            })
        except:
            continue
    
    return pd.DataFrame(results)

# Main content
with st.spinner(f"📊 Loading {selected_crypto_name} data..."):
    latest_data = get_latest_data(selected_ticker)
    full_data = load_data_cached(selected_ticker, period)

if latest_data and full_data is not None:
    # Dynamic title
    st.markdown(f"## {crypto_emoji} {selected_crypto_name} Analysis Dashboard")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"💰 {selected_crypto_name} Price",
            f"${latest_data['price']:,.2f}",
            f"{latest_data['data']['Returns'].iloc[-1]*100:.2f}%"
        )
    
    with col2:
        if show_gauge:
            with st.container():
                st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                gauge_fig = create_rsi_gauge(latest_data['rsi'])
                st.plotly_chart(gauge_fig, use_container_width=True, key="rsi_gauge")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.metric(
                "📊 RSI",
                f"{latest_data['rsi']:.2f}",
                "Oversold" if latest_data['rsi'] < 30 else "Overbought" if latest_data['rsi'] > 70 else "Normal"
            )
    
    with col3:
        st.metric(
            "📈 Trend",
            "Uptrend" if latest_data['trend'] == 'Yükseliş' else "Downtrend",
            f"Volatility: {latest_data['volatility']:.2f}%"
        )
    
    with col4:
        signal_text = "BUY" if latest_data['signal'] == 'AL' else "SELL" if latest_data['signal'] == 'SAT' else "HOLD"
        signal_color = "🟢" if latest_data['signal'] == 'AL' else "🔴" if latest_data['signal'] == 'SAT' else "🟡"
        st.metric(
            "🎯 Signal",
            f"{signal_color} {signal_text}",
            "AI Recommendation"
        )
    
    # AI Predictions
    st.markdown("---")
    st.markdown("## 🔮 AI Predictions")
    
    model, scaler, columns = load_improved_model(selected_ticker)
    
    if model is not None:
        try:
            # Create predictor for model
            predictor = ImprovedCryptoPredictor(ticker=selected_ticker)
            predictor.model = model
            predictor.feature_scaler = scaler
            predictor.feature_columns = columns
            
            # Make predictions
            predictions = predictor.predict_next_days(full_data)
            
            if predictions:
                # AI Confidence Score
                if show_confidence:
                    confidence_score = calculate_ai_confidence(predictions)
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="color: {crypto_color};">🤖 AI Confidence Score</h3>
                        <h1 style="color: white;">{confidence_score:.1f}%</h1>
                        <p style="color: gray;">Prediction consistency level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction cards
                pred_cols = st.columns(3)
                horizons = ['1_day', '3_day', '7_day']
                icons = ['🎯', '📅', '📆']
                
                for col, horizon, icon in zip(pred_cols, horizons, icons):
                    with col:
                        pred_data = predictions['predictions'][horizon]
                        change_color = "#00ff00" if pred_data['change'] > 0 else "#ff0000"
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="text-align: center; color: {crypto_color};">
                                {icon} {horizon.replace('_', ' ').title()}
                            </h3>
                            <h2 style="text-align: center; color: white;">
                                ${pred_data['price']:,.2f}
                            </h2>
                            <p style="text-align: center; color: {change_color}; font-size: 1.5em; font-weight: bold;">
                                {pred_data['change']:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not make predictions: {e}")
    else:
        st.info(f"📊 Please train the model for {selected_crypto_name} first.")
    
    # Strategy Leaderboard
    if show_strategies:
        st.markdown("---")
        st.markdown("## 🏆 Strategy Leaderboard")
        
        with st.spinner("Analyzing strategies..."):
            full_data = generate_trading_signals(full_data)
            strategy_results = run_strategy_comparison(full_data, selected_crypto_name)
            
            if not strategy_results.empty:
                st.markdown('<div class="strategy-table">', unsafe_allow_html=True)
                
                # Highlight best strategy
                best_strategy = strategy_results.iloc[0]['Strategy']
                st.success(f"🥇 Best Strategy: **{best_strategy}**")
                
                # Display as table
                st.dataframe(
                    strategy_results,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Strategy": st.column_config.TextColumn("📊 Strategy"),
                        "Return (%)": st.column_config.TextColumn("💰 Return"),
                        "Win Rate (%)": st.column_config.TextColumn("🎯 Success"),
                        "Max Drawdown (%)": st.column_config.TextColumn("📉 Max Loss")
                    }
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chart
    st.markdown("---")
    st.markdown(f"## 📈 {selected_crypto_name} Price Chart")
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{selected_crypto_name} Price", "Volume")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=full_data.index,
            open=full_data['Open'],
            high=full_data['High'],
            low=full_data['Low'],
            close=full_data['Close'],
            name=selected_crypto_name,
            increasing_line_color=crypto_color,
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=full_data.index, 
            y=full_data['SMA_50'],
            name='SMA 50',
            line=dict(color='orange', width=1.5)
        ),
        row=1, col=1
    )
    
    # Volume
    colors = [crypto_color if row['Close'] >= row['Open'] else '#ff4444' 
             for idx, row in full_data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=full_data.index,
            y=full_data['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Chart layout
    fig.update_layout(
        title=f"{crypto_emoji} {selected_crypto_name} Technical Analysis",
        template="plotly_dark",
        height=600,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🚀 WarpCrypto-Trader v2.0 | AI-Powered Analysis</p>
        <p>Perfect for X.com/Twitter content creation! 📹</p>
        <p style="font-size: 1.2em; margin-top: 20px; color: #4ECDC4;">⚡ Powered by <strong>Turtir-AI</strong> ⚡</p>
    </div>
    """, unsafe_allow_html=True)

# Auto refresh
if 'auto_refresh' in locals() and auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
