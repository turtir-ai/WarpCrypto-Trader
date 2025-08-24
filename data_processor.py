"""
WarpBTC-Trader: Veri İşleme Modülü
Bitcoin verilerini çeker ve teknik analiz indikatörlerini hesaplar
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(ticker='BTC-USD', period='5y', interval='1d'):
    """
    Kripto para verilerini çeker ve teknik indikatörleri hesaplar
    
    Args:
        ticker (str): Sembol (BTC-USD, ETH-USD, SOL-USD vb.)
        period (str): Veri periyodu (1y, 2y, 5y, max)
        interval (str): Veri aralığı (1d, 1h, 30m)
    
    Returns:
        pd.DataFrame: İşlenmiş veri seti
    """
    
    print(f"📊 {ticker} verisi çekiliyor... (Periyot: {period}, Aralık: {interval})")
    
    # Yahoo Finance'den veri çek
    try:
        crypto_data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if crypto_data.empty:
            raise Exception("Veri çekilemedi!")
            
        print(f"✅ {len(crypto_data)} satır veri başarıyla çekildi.")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None
    
    # Teknik indikatörleri hesapla
    print("🔧 Teknik indikatörler hesaplanıyor...")
    
    # Multi-level column'ları düzelt
    if isinstance(crypto_data.columns, pd.MultiIndex):
        crypto_data.columns = crypto_data.columns.get_level_values(0)
    
    # Temel fiyat hareketleri
    crypto_data['Returns'] = crypto_data['Close'].pct_change()
    crypto_data['Log_Returns'] = np.log(crypto_data['Close'] / crypto_data['Close'].shift(1))
    
    # Volatilite
    crypto_data['Volatility'] = crypto_data['Returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=crypto_data['Close'], window=14)
    crypto_data['RSI'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = MACD(close=crypto_data['Close'])
    crypto_data['MACD'] = macd_indicator.macd()
    crypto_data['MACD_Signal'] = macd_indicator.macd_signal()
    crypto_data['MACD_Diff'] = macd_indicator.macd_diff()
    
    # Bollinger Bantları
    bb_indicator = BollingerBands(close=crypto_data['Close'], window=20, window_dev=2)
    crypto_data['BB_High'] = bb_indicator.bollinger_hband()
    crypto_data['BB_Low'] = bb_indicator.bollinger_lband()
    crypto_data['BB_Middle'] = bb_indicator.bollinger_mavg()
    crypto_data['BB_Width'] = crypto_data['BB_High'] - crypto_data['BB_Low']
    crypto_data['BB_Position'] = (crypto_data['Close'] - crypto_data['BB_Low']) / crypto_data['BB_Width']
    
    # Hareketli Ortalamalar
    crypto_data['SMA_10'] = SMAIndicator(close=crypto_data['Close'], window=10).sma_indicator()
    crypto_data['SMA_30'] = SMAIndicator(close=crypto_data['Close'], window=30).sma_indicator()
    crypto_data['SMA_50'] = SMAIndicator(close=crypto_data['Close'], window=50).sma_indicator()
    # SMA 200'ü sadece yeterli veri varsa hesapla
    if len(crypto_data) >= 200:
        crypto_data['SMA_200'] = SMAIndicator(close=crypto_data['Close'], window=200).sma_indicator()
    else:
        crypto_data['SMA_200'] = crypto_data['SMA_50']  # 200 günlük veri yoksa 50 günlük SMA'yı kullan
    
    crypto_data['EMA_12'] = EMAIndicator(close=crypto_data['Close'], window=12).ema_indicator()
    crypto_data['EMA_26'] = EMAIndicator(close=crypto_data['Close'], window=26).ema_indicator()
    
    # Stokastik Osilatör
    stoch_indicator = StochasticOscillator(
        high=crypto_data['High'], 
        low=crypto_data['Low'], 
        close=crypto_data['Close']
    )
    crypto_data['Stoch_K'] = stoch_indicator.stoch()
    crypto_data['Stoch_D'] = stoch_indicator.stoch_signal()
    
    # Hacim İndikatörleri
    crypto_data['OBV'] = OnBalanceVolumeIndicator(
        close=crypto_data['Close'], 
        volume=crypto_data['Volume']
    ).on_balance_volume()
    
    crypto_data['Volume_SMA'] = crypto_data['Volume'].rolling(window=20).mean()
    # Volume_Ratio hesaplamasını güvenli yap
    with np.errstate(divide='ignore', invalid='ignore'):
        crypto_data['Volume_Ratio'] = np.where(
            crypto_data['Volume_SMA'] != 0,
            crypto_data['Volume'] / crypto_data['Volume_SMA'],
            1.0
        )
    
    # Fiyat Seviyeleri
    crypto_data['High_Low_Ratio'] = crypto_data['High'] / crypto_data['Low']
    crypto_data['Close_Open_Ratio'] = crypto_data['Close'] / crypto_data['Open']
    
    # Support ve Resistance Seviyeleri (basit yaklaşım)
    crypto_data['Resistance'] = crypto_data['High'].rolling(window=20).max()
    crypto_data['Support'] = crypto_data['Low'].rolling(window=20).min()
    # Price_Position hesaplamasını güvenli yap
    with np.errstate(divide='ignore', invalid='ignore'):
        crypto_data['Price_Position'] = np.where(
            (crypto_data['Resistance'] - crypto_data['Support']) != 0,
            (crypto_data['Close'] - crypto_data['Support']) / (crypto_data['Resistance'] - crypto_data['Support']),
            0.5
        )
    
    # Trend göstergeleri
    crypto_data['Trend'] = np.where(crypto_data['SMA_50'] > crypto_data['SMA_200'], 1, -1)
    
    # NaN değerleri temizle
    crypto_data = crypto_data.dropna()
    
    print(f"✅ {len(crypto_data.columns)} özellik başarıyla hesaplandı.")
    print(f"📈 İşlenmiş veri seti: {len(crypto_data)} satır x {len(crypto_data.columns)} sütun")
    
    return crypto_data

def generate_trading_signals(data):
    """
    Teknik indikatörlere dayalı alım-satım sinyalleri üretir
    
    Args:
        data (pd.DataFrame): İşlenmiş veri
    
    Returns:
        pd.DataFrame: Sinyal kolonları eklenmiş veri
    """
    
    # RSI Sinyalleri
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1,  # Aşırı satım - AL
                                  np.where(data['RSI'] > 70, -1,  # Aşırı alım - SAT
                                          0))  # TUT
    
    # MACD Sinyalleri
    data['MACD_Signal_Trade'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
    
    # Bollinger Bantları Sinyalleri
    data['BB_Signal'] = np.where(data['Close'] < data['BB_Low'], 1,  # Alt bantta - AL
                                 np.where(data['Close'] > data['BB_High'], -1,  # Üst bantta - SAT
                                         0))  # TUT
    
    # Hareketli Ortalama Kesişimleri
    data['MA_Cross_Signal'] = np.where(data['SMA_10'] > data['SMA_30'], 1, -1)
    
    # Kombinasyon Sinyal
    data['Combined_Signal'] = (
        data['RSI_Signal'] * 0.25 +
        data['MACD_Signal_Trade'] * 0.25 +
        data['BB_Signal'] * 0.25 +
        data['MA_Cross_Signal'] * 0.25
    )
    
    # Nihai Sinyal
    data['Final_Signal'] = np.where(data['Combined_Signal'] > 0.5, 'AL',
                                    np.where(data['Combined_Signal'] < -0.5, 'SAT', 'TUT'))
    
    return data

def get_latest_data(ticker='BTC-USD'):
    """
    En güncel kripto para verisini çeker
    
    Args:
        ticker (str): Sembol (BTC-USD, ETH-USD, SOL-USD vb.)
    
    Returns:
        dict: En güncel veri ve indikatörler
    """
    
    # Son 100 günlük veriyi çek
    data = load_and_process_data(ticker, period='3mo', interval='1d')
    
    if data is None or data.empty:
        return None
    
    # Sinyalleri ekle
    data = generate_trading_signals(data)
    
    # En son veriyi al
    latest = data.iloc[-1]
    
    return {
        'timestamp': data.index[-1],
        'price': latest['Close'],
        'volume': latest['Volume'],
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'signal': latest['Final_Signal'],
        'trend': 'Yükseliş' if latest['Trend'] == 1 else 'Düşüş',
        'volatility': latest['Volatility'] * 100,  # Yüzde olarak
        'data': data  # Tüm veri
    }

if __name__ == "__main__":
    # Test modülü
    print("=" * 60)
    print("WarpCrypto-Trader Veri İşleme Modülü Test")
    print("=" * 60)
    
    # Farklı kripto paralar için test
    test_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for ticker in test_tickers:
        print(f"\n🪙 {ticker} test ediliyor...")
        data = load_and_process_data(ticker, period='6mo')
        
        if data is not None:
            print(f"✅ {ticker} verisi başarıyla işlendi.")
            print(f"   Ortalama Fiyat: ${data['Close'].mean():,.2f}")
            print(f"   Son Fiyat: ${data['Close'].iloc[-1]:,.2f}")
            print(f"   RSI: {data['RSI'].iloc[-1]:.2f}")
