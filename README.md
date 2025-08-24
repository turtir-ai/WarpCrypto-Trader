# 🚀 WarpCrypto-Trader v2.0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Powered by](https://img.shields.io/badge/Powered%20by-Turtir--AI-cyan)](https://github.com/turtir-ai)

## 🎯 Overview

**WarpCrypto-Trader** is an advanced AI-powered cryptocurrency analysis platform that provides real-time predictions, technical analysis, and trading strategy comparisons for Bitcoin, Ethereum, and Solana. Perfect for creating engaging content for social media platforms like X.com (Twitter)!

![Demo](https://via.placeholder.com/800x400/0f0c29/4ECDC4?text=WarpCrypto-Trader+v2.0+Demo)

## ✨ Features

### 🤖 AI-Powered Predictions
- **LSTM/GRU Hybrid Models** - Advanced deep learning architecture for accurate price predictions
- **Multi-Horizon Forecasting** - 1-day, 3-day, and 7-day price predictions
- **AI Confidence Score** - Measures prediction consistency and reliability
- **Per-Crypto Models** - Separately trained models for BTC, ETH, and SOL

### 📊 Technical Analysis
- **Real-time Data** - Live cryptocurrency data from Yahoo Finance
- **25+ Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Interactive RSI Gauge** - Visual representation of market conditions
- **Dynamic Price Charts** - Candlestick charts with volume analysis

### 🏆 Strategy Leaderboard
- **Multiple Trading Strategies** - RSI, MACD, Bollinger Bands, and combined strategies
- **Backtesting Engine** - Historical performance analysis
- **Win Rate & Returns** - Comprehensive strategy metrics
- **Real-time Comparison** - Find the best strategy for current market conditions

### 🎨 Modern UI/UX
- **Glassmorphism Design** - Beautiful, transparent card layouts
- **Dynamic Color Themes** - Each cryptocurrency has its unique color scheme
- **Responsive Layout** - Works perfectly on all screen sizes
- **Dark Mode** - Eye-friendly interface for extended use

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/WarpCrypto-Trader.git
cd WarpCrypto-Trader
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Models (Optional)
Train models for all cryptocurrencies:

**Windows PowerShell:**
```powershell
.\train_all_models.ps1
```

**Linux/Mac:**
```bash
chmod +x train_all_models.sh
./train_all_models.sh
```

Or train individually:
```bash
python train_model_fixed.py --ticker BTC-USD
python train_model_fixed.py --ticker ETH-USD
python train_model_fixed.py --ticker SOL-USD
```

### Step 4: Launch the Application
```bash
streamlit run app_v2.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
WarpCrypto-Trader/
│
├── app_v2.py                  # Main Streamlit application (English version)
├── data_processor.py          # Data fetching and processing module
├── train_model_fixed.py       # LSTM/GRU model training script
├── strategy_backtester.py     # Trading strategy backtesting engine
│
├── train_all_models.ps1       # Windows batch training script
├── train_all_models.sh        # Linux/Mac batch training script
│
├── models/                    # Trained models directory (auto-created)
│   ├── btc_usd_predictor_improved.keras
│   ├── eth_usd_predictor_improved.keras
│   └── sol_usd_predictor_improved.keras
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🎮 Usage Guide

### 1. Select Cryptocurrency
Use the sidebar dropdown to choose between Bitcoin, Ethereum, or Solana.

### 2. Train Models (First Time)
Click the "Train Model" button in the sidebar if the model hasn't been trained yet.

### 3. Customize Display
- **RSI Gauge Indicator** - Toggle the visual RSI gauge
- **Strategy Leaderboard** - Show/hide strategy comparison table
- **AI Confidence Score** - Display prediction reliability metric

### 4. Analyze Results
- Check current price and trend
- Review AI predictions for different time horizons
- Compare trading strategy performances
- Examine technical indicators

## 📹 Content Creation Tips

Perfect for creating engaging content on X.com (Twitter):

### Video Ideas
1. **"AI Predicts Crypto Prices!"** - Show AI predictions with confidence scores
2. **"Best Trading Strategy Revealed"** - Highlight the strategy leaderboard
3. **"Bitcoin vs Ethereum vs Solana"** - Quick comparison between cryptos
4. **"RSI Alert!"** - Focus on the RSI gauge when it's in extreme zones

### Recording Tips
- Use dark theme for better visuals
- Slow transitions between sections
- Highlight AI confidence scores
- Show strategy performance comparisons
- Zoom in on key metrics

## 🔧 Configuration

### Modify Cryptocurrencies
Edit the `CRYPTO_OPTIONS` dictionary in `app_v2.py`:
```python
CRYPTO_OPTIONS = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Solana (SOL)': 'SOL-USD',
    # Add more here
}
```

### Adjust Model Parameters
In `train_model_fixed.py`, modify:
```python
predictor = ImprovedCryptoPredictor(
    ticker=ticker,
    sequence_length=30,  # Days of historical data
    prediction_horizons=[1, 3, 7]  # Prediction days
)
```

### Change Analysis Period
Default periods can be modified in the sidebar options.

## 📊 Technical Indicators

The platform calculates and displays:
- **Price Metrics**: Open, High, Low, Close, Volume
- **Moving Averages**: SMA (10, 30, 50, 200), EMA (12, 26)
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume Ratio
- **Custom**: Support/Resistance Levels, Trend Direction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** - Real-time cryptocurrency data
- **Streamlit** - Amazing web app framework
- **TensorFlow** - Deep learning models
- **Plotly** - Interactive visualizations
- **TA-Lib** - Technical analysis indicators

## ⚡ Powered by Turtir-AI

This project is powered by **[Turtir-AI](https://github.com/turtir-ai)** - Advanced AI Solutions for Trading and Analysis.

## 📧 Contact

For questions, suggestions, or collaborations:
- Twitter: [@YourTwitter](https://x.com)
- GitHub: [Your GitHub Profile](https://github.com/yourusername)

---

<p align="center">
  Made with ❤️ for the crypto community
  <br>
  ⭐ Star this repo if you find it helpful!
</p>
