# LSTM 與 GRU 股票收盤價比較

## 實驗設定

- 原 LSTM 實驗：`C:\Users\PGW\Github\LSTM_Stock_Trends_Prediction\results\20261120T134010119780Z_82eaad21`
- 模型：單向 LSTM(64) 與 GRU(64)，各接 Dense(1)；相同 units 不代表相同參數量，依各模型實際參數量列示。
- 資料：TWSE 2330.TW 原始 OHLCV；資料 SHA256 `c431da778661ac0c247a9d91842538d498a85d701d7d243b6d51dfe813853646`，交易日曆 SHA256 `4fbf02495853afce74c647a0673136b5c74ebd612fb46d7cd1fb73e32f994965`。
- 特徵組 A/B/C、lookback 60 個交易日、horizon 1；train 2015-01-01–2022-12-31、validation 2023-01-01–2023-12-31、test 2024-01-01–2025-12-31。
- 每組樣本數：train 1865、validation 239、test 485；測試目標日期 2024-01-02 至 2025-12-31。
- GRU 重用原 LSTM run 的 X_A/X_B/X_C/y scalers（只 transform，不 refit）；scaler SHA256：`{'scalers/description.json': 'd9ed1a164d986847b4beb4536831b5e9d7e34d9f3bd6e3a001951bec89e6b00f', 'scalers/X_A.pkl': 'cda6d5b254c7b039e783911ad8f8193f7e2186987001051b604c40ca43b5b29c', 'scalers/X_B.pkl': 'fc7786a7c4b87eaf440587b88af61f6c17edeefe579b1c9f7e5f3b6a6ace2b7f', 'scalers/X_C.pkl': '7225387b797150acabe128f1d7f955fc75f4c2e578f41d29942825e602d26821', 'scalers/y.pkl': '4cc59ad58010c878c821d86acfbadb5afd85ddb5131e4cc0918f52dcf85056fb'}`。
- 相同 optimizer=adam、learning rate=0.001、loss=mse、batch size=32、shuffle=False、max epochs=100、early stopping monitor=val_loss patience=10 restore_best_weights=True；seeds=[42, 43, 44]。
- 評估指標由共同測試資料上的逐日預測重算；MAE/RMSE 單位為 TWD，方向正確率是上漲/下跌/持平三分類比例。表格方向指標以百分比表示。
- GRU 執行環境：Python 3.11.18，TensorFlow CPU 2.21.0，Keras 3.15.1，device=cpu。

## 三組結果

### A（Close）

| Model | Parameters | train seconds mean | MAE mean ± sample SD (TWD) | RMSE mean ± sample SD (TWD) | Directional accuracy mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| LSTM | 16961 | 40.26 | 161.3534 ± 54.8886 | 185.0638 ± 60.3368 | 40.14% ± 0.12% |
| GRU | 12929 | 29.48 | 66.5081 ± 4.8650 | 80.7907 ± 5.6570 | 40.21% ± 0.00% |
### B（Open, High, Low, Close, Volume）

| Model | Parameters | train seconds mean | MAE mean ± sample SD (TWD) | RMSE mean ± sample SD (TWD) | Directional accuracy mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| LSTM | 17985 | 22.10 | 201.0378 ± 3.6275 | 241.9335 ± 4.2197 | 40.62% ± 0.00% |
| GRU | 13697 | 14.56 | 87.0576 ± 26.6635 | 105.1990 ± 28.3588 | 40.27% ± 0.48% |
### C（Open, High, Low, Close, Volume, MA5, MA20, RSI14, MACD, MACD_signal）

| Model | Parameters | train seconds mean | MAE mean ± sample SD (TWD) | RMSE mean ± sample SD (TWD) | Directional accuracy mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| LSTM | 19265 | 17.77 | 223.4121 ± 106.6117 | 265.0893 ± 112.6466 | 41.51% ± 1.37% |
| GRU | 14657 | 14.69 | 181.2857 ± 35.5932 | 211.9667 ± 37.9032 | 40.00% ± 0.00% |

每種子結果見 `evaluation/comparison_per_seed.csv`；彙總見 `evaluation/comparison_summary.csv`。標準差為三個固定種子的樣本標準差。

訓練秒數來自各 run 的訓練紀錄。若兩次實驗套件或硬體環境不同，時間只能作為各自紀錄，不能視為公平速度比較；環境版本見本 run 的 `environment.json` 及原 run 的 `environment.json`。

此比較固定資料、日期切分、縮放器、特徵組與訓練規則，未根據 test 調整超參數。結果不代表統計顯著差異或未來預測表現。
