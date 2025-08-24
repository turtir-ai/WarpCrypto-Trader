Write-Host "==========================================================="
Write-Host "🚀 WarpCrypto-Trader - Tüm Modelleri Eğitme Başlıyor" -ForegroundColor Cyan
Write-Host "==========================================================="
Write-Host ""

# Eğitilecek kripto para listesi
$tickers = @("BTC-USD", "ETH-USD", "SOL-USD")

# Her kripto para için modeli eğit
foreach ($ticker in $tickers) {
    Write-Host "-----------------------------------------------------------"
    Write-Host "🪙 $ticker modeli eğitiliyor..." -ForegroundColor Yellow
    Write-Host "-----------------------------------------------------------"
    
    python train_model_fixed.py --ticker $ticker
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $ticker modeli başarıyla eğitildi!" -ForegroundColor Green
    } else {
        Write-Host "❌ $ticker model eğitiminde hata oluştu!" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "==========================================================="
Write-Host "🎉 Tüm modellerin eğitimi tamamlandı!" -ForegroundColor Green
Write-Host "==========================================================="

# Oluşturulan model dosyalarını listele
Write-Host ""
Write-Host "📁 Oluşturulan model dosyaları:" -ForegroundColor Cyan
Get-ChildItem -Filter "*.keras" | Format-Table Name, Length, LastWriteTime -AutoSize

Write-Host "📁 Oluşturulan scaler dosyaları:" -ForegroundColor Cyan
Get-ChildItem -Filter "*.pkl" | Format-Table Name, Length, LastWriteTime -AutoSize
