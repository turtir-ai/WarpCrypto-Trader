#!/bin/bash

echo "==========================================================="
echo "🚀 WarpCrypto-Trader - Tüm Modelleri Eğitme Başlıyor"
echo "==========================================================="
echo ""

# Eğitilecek kripto para listesi
TICKERS=("BTC-USD" "ETH-USD" "SOL-USD")

# Her kripto para için modeli eğit
for ticker in "${TICKERS[@]}"
do
    echo "-----------------------------------------------------------"
    echo "🪙 $ticker modeli eğitiliyor..."
    echo "-----------------------------------------------------------"
    
    python train_model_fixed.py --ticker "$ticker"
    
    if [ $? -eq 0 ]; then
        echo "✅ $ticker modeli başarıyla eğitildi!"
    else
        echo "❌ $ticker model eğitiminde hata oluştu!"
    fi
    
    echo ""
done

echo "==========================================================="
echo "🎉 Tüm modellerin eğitimi tamamlandı!"
echo "==========================================================="

# Oluşturulan model dosyalarını listele
echo ""
echo "📁 Oluşturulan model dosyaları:"
ls -lh *.keras 2>/dev/null
echo ""
echo "📁 Oluşturulan scaler dosyaları:"
ls -lh *.pkl 2>/dev/null
