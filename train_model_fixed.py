"""
WarpCrypto-Trader: Çoklu Kripto Para LSTM Model Eğitim Modülü
Bitcoin, Ethereum, Solana ve diğer kripto paralar için optimize edilmiş model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import models as keras_models
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# TensorFlow uyarılarını kapat
tf.get_logger().setLevel('ERROR')

from data_processor import load_and_process_data

class ImprovedCryptoPredictor:
    """Geliştirilmiş Kripto Para tahmin modeli"""
    
    def __init__(self, ticker='BTC-USD', sequence_length=30, prediction_horizons=[1, 3, 7]):
        """
        Args:
            ticker (str): Kripto para sembolü (BTC-USD, ETH-USD, SOL-USD vb.)
            sequence_length (int): Tahmin için kullanılacak geçmiş gün sayısı
            prediction_horizons (list): Tahmin edilecek gelecek günler
        """
        self.ticker = ticker.lower().replace('-', '_')
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.model = None
        self.feature_scaler = RobustScaler()  # Outlier'lara karşı daha dayanıklı
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))  # Fiyat için ayrı scaler
        self.feature_columns = None
        self.history = None
        
    def prepare_features(self, data):
        """
        Model için özellikleri hazırlar
        
        Args:
            data (pd.DataFrame): İşlenmiş Bitcoin verisi
            
        Returns:
            pd.DataFrame: Seçilmiş özellikler
        """
        # Tahmin için kullanılacak özellikler
        feature_columns = [
            'Close', 'Volume', 'Returns', 'Volatility',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_Position', 'BB_Width',
            'SMA_10', 'SMA_30', 'SMA_50',
            'EMA_12', 'EMA_26',
            'Stoch_K', 'Stoch_D',
            'Volume_Ratio', 'Price_Position', 'Trend'
        ]
        
        # Mevcut sütunlardan seç
        self.feature_columns = [col for col in feature_columns if col in data.columns]
        
        return data[self.feature_columns].copy()
    
    def create_sequences_with_returns(self, features_df):
        """
        LSTM için sekansları oluşturur - getiri bazlı tahmin
        
        Args:
            features_df (pd.DataFrame): Özellikler
            
        Returns:
            tuple: X (özellikler), y (gelecek getiriler), prices (referans fiyatlar)
        """
        # Özellikleri normalleştir
        features_scaled = self.feature_scaler.fit_transform(features_df)
        
        # Fiyatları ayrıca sakla
        prices = features_df['Close'].values
        
        X, y, ref_prices = [], [], []
        
        for i in range(self.sequence_length, len(features_scaled) - max(self.prediction_horizons)):
            # Input sequence
            X.append(features_scaled[i-self.sequence_length:i])
            
            # Mevcut fiyat
            current_price = prices[i-1]
            ref_prices.append(current_price)
            
            # Her horizon için getiri oranı hesapla (log return)
            targets = []
            for horizon in self.prediction_horizons:
                future_price = prices[i + horizon - 1]
                # Log return kullan - daha stabil eğitim için
                log_return = np.log(future_price / current_price)
                targets.append(log_return)
            
            y.append(targets)
        
        return np.array(X), np.array(y), np.array(ref_prices)
    
    def build_improved_model(self, input_shape):
        """
        Geliştirilmiş LSTM/GRU hibrit model
        
        Args:
            input_shape (tuple): Giriş verisi şekli
            
        Returns:
            Sequential: Keras modeli
        """
        model = Sequential([
            # Birinci katman - LSTM
            LSTM(64, return_sequences=True, input_shape=input_shape, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            # İkinci katman - GRU (daha hızlı ve bazen daha iyi performans)
            GRU(32, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Üçüncü katman - LSTM
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            
            # Dense katmanlar
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Çıkış katmanı - log return tahminleri için
            Dense(len(self.prediction_horizons), activation='linear')
        ])
        
        # Özel loss fonksiyonu - direction accuracy'yi de önemse
        def custom_loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # Direction penalty - yön tahmini yanlışsa ekstra ceza
            direction_penalty = tf.where(
                tf.sign(y_true) == tf.sign(y_pred),
                0.0,
                tf.abs(y_true - y_pred) * 0.5
            )
            return mse + tf.reduce_mean(direction_penalty)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Outlier'lara karşı daha dayanıklı
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """
        Modeli eğitir
        
        Args:
            data (pd.DataFrame): Bitcoin verisi
            epochs (int): Epoch sayısı
            batch_size (int): Batch boyutu
            validation_split (float): Validasyon verisi oranı
        """
        print("🚀 Geliştirilmiş model eğitimi başlıyor...")
        
        # Özellikleri hazırla
        features = self.prepare_features(data)
        
        # Sekansları oluştur
        X, y, ref_prices = self.create_sequences_with_returns(features)
        
        print(f"📊 Veri şekli: X={X.shape}, y={y.shape}")
        print(f"📈 Getiri istatistikleri:")
        print(f"   Ortalama 1-gün getiri: {np.mean(y[:, 0]):.4f}")
        print(f"   Std 1-gün getiri: {np.std(y[:, 0]):.4f}")
        
        # Train-test split
        split_idx = int(len(X) * (1 - 0.2))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        self.test_ref_prices = ref_prices[split_idx:]
        
        # Modeli oluştur
        self.model = self.build_improved_model((X.shape[1], X.shape[2]))
        
        print("\n📐 Model Mimarisi:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.ticker}_predictor_improved.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Modeli eğit
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Test performansı
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Gerçek fiyat tahminlerini test et
        test_predictions = self.model.predict(X_test, verbose=0)
        
        # Log return'leri fiyatlara çevir
        predicted_prices = []
        for i, pred_returns in enumerate(test_predictions):
            current_price = self.test_ref_prices[i]
            pred_prices = current_price * np.exp(pred_returns)
            predicted_prices.append(pred_prices)
        
        predicted_prices = np.array(predicted_prices)
        
        # Gerçek fiyatları hesapla
        actual_prices = []
        for i, actual_returns in enumerate(y_test):
            current_price = self.test_ref_prices[i]
            act_prices = current_price * np.exp(actual_returns)
            actual_prices.append(act_prices)
        
        actual_prices = np.array(actual_prices)
        
        # MAPE hesapla
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        print(f"\n✅ Test Performansı:")
        print(f"   Loss: {test_loss:.6f}")
        print(f"   MAE (log return): {test_mae:.4f}")
        print(f"   MAPE (fiyat): {mape:.2f}%")
        
        # Direction accuracy
        direction_accuracy = np.mean(np.sign(test_predictions) == np.sign(y_test)) * 100
        print(f"   Yön Doğruluğu: {direction_accuracy:.2f}%")
        
        return self.history
    
    def predict_next_days(self, data):
        """
        Gelecek günlerin fiyat tahminini yapar
        
        Args:
            data (pd.DataFrame): Son veri
            
        Returns:
            dict: Tahminler
        """
        # Özellikleri hazırla
        features = self.prepare_features(data)
        
        # Yeterli veri kontrolü
        if len(features) < self.sequence_length:
            raise ValueError(f"Yetersiz veri: {len(features)} < {self.sequence_length}")
        
        # Son sequence_length günü al
        last_sequence = features.iloc[-self.sequence_length:].values
        
        # Normalleştir
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, self.sequence_length, -1)
        
        # Log return tahminleri yap
        log_returns = self.model.predict(last_sequence_scaled, verbose=0)[0]
        
        # Mevcut fiyat
        current_price = data['Close'].iloc[-1]
        
        # Log return'leri fiyatlara çevir
        predictions = current_price * np.exp(log_returns)
        
        # Mantıklı sınırlar koy (±%30 max değişim)
        max_change = 0.30
        predictions = np.clip(predictions, 
                            current_price * (1 - max_change),
                            current_price * (1 + max_change))
        
        # Sonuçları düzenle
        results = {
            'current_price': current_price,
            'predictions': {}
        }
        
        for i, horizon in enumerate(self.prediction_horizons):
            results['predictions'][f'{horizon}_day'] = {
                'price': predictions[i],
                'change': (predictions[i] - current_price) / current_price * 100
            }
        
        return results
    
    def save_model(self, model_path=None):
        """Model ve scaler'ları kaydet"""
        if self.model:
            if model_path is None:
                model_path = f'{self.ticker}_predictor_improved.keras'
            
            self.model.save(model_path)
            joblib.dump(self.feature_scaler, f'{self.ticker}_feature_scaler_improved.pkl')
            joblib.dump(self.feature_columns, f'{self.ticker}_feature_columns_improved.pkl')
            print(f"✅ Model kaydedildi: {model_path}")
            print(f"✅ Scaler'lar kaydedildi: {self.ticker}_*.pkl")

def main(ticker='BTC-USD'):
    """Ana eğitim fonksiyonu"""
    print("="*60)
    print(f"🚀 WarpCrypto-Trader {ticker} LSTM Model Eğitimi")
    print("="*60)
    
    # Veriyi yükle
    print(f"\n📊 {ticker} verisi yükleniyor...")
    data = load_and_process_data(ticker=ticker, period='2y')
    print(f"✅ {len(data)} satır veri yüklendi.")
    
    # Modeli oluştur ve eğit
    predictor = ImprovedCryptoPredictor(ticker=ticker, sequence_length=30, prediction_horizons=[1, 3, 7])
    
    # Eğit
    predictor.train(data, epochs=50, batch_size=32)
    
    # Model kaydet
    predictor.save_model()
    
    # Test tahmini
    print("\n🔮 Tahmin yapılıyor...")
    predictions = predictor.predict_next_days(data)
    
    print(f"\n📈 TAHMİN SONUÇLARI:")
    print(f"Mevcut Fiyat: ${predictions['current_price']:,.2f}")
    print("-" * 40)
    
    for period, pred in predictions['predictions'].items():
        print(f"  {period}: ${pred['price']:,.2f} ({pred['change']:+.2f}%)")
    
    print("\n✅ Eğitim tamamlandı!")

if __name__ == "__main__":
    # Komut satırı argümanlarını parse et
    parser = argparse.ArgumentParser(description='Kripto para tahmin modeli eğitimi')
    parser.add_argument('--ticker', type=str, default='BTC-USD',
                       help='Kripto para sembolü (örn: BTC-USD, ETH-USD, SOL-USD)')
    
    args = parser.parse_args()
    
    # Eğitimi başlat
    main(ticker=args.ticker)
