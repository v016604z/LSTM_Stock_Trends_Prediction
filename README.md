
# LSTM Stock_Trend_Prediction

Analyze the stock trend prediction of 台積電(2330.TW) by constructing an LSTM model.

## Authors

- [@PoDa](https://github.com/v016604z)

## Code references

- [@AUSTIN2526](https://github.com/AUSTIN2526/learn-AI-in-30-days-book-version/tree/main)'s Ch.05 LSTM principle introduction and Keras implementation
- [@Docat0209]LSTM_Stock_Prediction_v1 co-producer

## LSTM 比較研究環境

新增的 `src/lstm_comparison/` 為獨立研究套件，使用 Python 3.11 與 CPU 版 TensorFlow。舊版 Notebook 與 `src/Trading/` 保留原用途；其舊相依套件清單不適用於本研究。研究設計與工作切分請見 [8 個 Commit 實作計畫](./docs/單變量與多變量%20LSTM%20股票價格預測績效比較｜實作計畫.md)。

研究相依套件由 `pyproject.toml` 宣告，`uv.lock` 鎖定直接及傳遞依賴。安裝基線為 Windows x64、Python 3.11、CPU；其他平台尚未驗收。

### 建立專案內環境

以下 PowerShell 指令在專案根目錄執行，使用 [uv 0.13.9 官方 standalone ZIP](https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-pc-windows-msvc.zip)，以官方 SHA256 驗證後解壓至 `.tools/uv/`。不修改全域 PATH 或 Windows registry。

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path .tools -Force | Out-Null
$uvUrl = 'https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-pc-windows-msvc.zip'
Invoke-WebRequest -Uri $uvUrl -OutFile .tools/uv.zip -UseBasicParsing
Invoke-WebRequest -Uri "$uvUrl.sha256" -OutFile .tools/uv.zip.sha256 -UseBasicParsing
$expectedHash = ((Get-Content .tools/uv.zip.sha256 -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash .tools/uv.zip -Algorithm SHA256).Hash -ne $expectedHash) { throw 'uv SHA256 mismatch' }
Expand-Archive -LiteralPath .tools/uv.zip -DestinationPath .tools/uv -Force
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.tools/python'
.\.tools\uv\uv.exe python install 3.11 --no-bin --no-registry
.\.tools\uv\uv.exe sync --locked
```

`.tools/`、`.uv-cache/`、`.venv/` 皆為本地環境，不納入版本控制。在新 PowerShell 視窗重新執行 uv 指令前，需再次設定上面兩個環境變數。也可直接使用 `.venv/Scripts/python.exe` 執行下列已安裝模組的指令，不需啟用虛擬環境。

### 本階段可用指令

```powershell
.\.venv\Scripts\python.exe -m lstm_comparison --help
.\.venv\Scripts\python.exe -m lstm_comparison validate-config --config configs/research.json
.\.venv\Scripts\python.exe -m lstm_comparison download --config configs/research.json
.\.venv\Scripts\python.exe -m lstm_comparison load-snapshot --config configs/research.json
.\.venv\Scripts\python.exe -m lstm_comparison run --config configs/research.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`configs/research.json` 保存正式研究設定：三組特徵、60 日視窗、年度切分，以及 seeds 42、43、44。資料快照與結果路徑一律相對於設定檔目錄解析，檢查成功時輸出解析後的絕對路徑；不要求行情已下載。`max_epochs` 與種子數等訓練設定可縮短供後續 smoke test 使用，正式實驗沿用核定設定。

`validate-config` 成功回傳 exit code 0，無效設定回傳 2。help 與設定檢查只使用 Python 標準庫，不載入 TensorFlow，也不連網。

`download` 會下載新行情快照；若快照已存在，會離線檢查並重用，不重新連線或覆寫。`load-snapshot` 僅離線檢查。`run` 只讀取既有快照並訓練，不自動下載；完成後另用 `evaluate --run-dir <實驗目錄>` 評估測試集，再用 `report --run-dir <實驗目錄>` 產生圖表與中文報告。來源、快照、訓練、評估或報告失敗回傳 1，`--config` 設定錯誤回傳 2。

### 三組研究特徵（Commit 3）

`build_feature_groups(frame, config)` 接受 `load_snapshot` 讀回的調整後 OHLCV，以及 `load_config` 驗證後的設定，回傳 `A`、`B`、`C` 三個 DataFrame：

```python
from lstm_comparison.config import load_config
from lstm_comparison.data import load_snapshot
from lstm_comparison.features import build_feature_groups

config = load_config("configs/research.json")
prices, metadata = load_snapshot(config)
groups = build_feature_groups(prices, config)
```

| 組別 | 欄位順序 | 欄數 |
| --- | --- | --- |
| A | Close | 1 |
| B | Open、High、Low、Close、Volume | 5 |
| C | Open、High、Low、Close、Volume、MA5、MA20、RSI14、MACD、MACD_signal | 10 |

所有指標由調整後 Close 計算，當日特徵只使用當日及以前資料。固定公式及初始化約定如下，暖機筆數指第一筆有效值之前保留的 NaN 筆數：

| 指標 | 公式與初始化 | 暖機筆數 |
| --- | --- | --- |
| MA5／MA20 | 包含當日在內的最近 5／20 筆 Close 算術平均；必須有完整視窗 | 4／19 |
| RSI14 | `gain=max(Close差分,0)`、`loss=max(-Close差分,0)`；以最初 14 個差分的平均漲幅與平均跌幅初始化，之後以 `(13×前值+當期值)/14` 作 Wilder 平滑；`RSI=100-100/(1+平均漲幅/平均跌幅)` | 14 |
| MACD | `EMA12(Close)-EMA26(Close)`；每個 EMA 以最初 n 筆輸入的 SMA 初始化，再用 `EMA_t=EMA_(t-1)+2/(n+1)×(輸入_t-EMA_(t-1))` 遞推 | 25 |
| MACD_signal | 對第一筆有效 MACD 起的序列計算 EMA9，同樣以 9 筆 SMA 初始化 | 33 |

RSI 的平均跌幅為零、平均漲幅大於零時取 100；平均漲幅為零、平均跌幅大於零時取 0；兩者皆為零時取 50。EMA 初始化與 pandas `ewm(adjust=False)` 預設的第一筆種子不同，本研究固定使用上述 SMA 種子。設定檔限制 MACD 為計畫指定的 12／26／9。

三組輸出保留原始交易日索引、OHLCV 數值與暖機 NaN，不更動輸入、不補值、不刪列、不重建交易日。C 的第 34 筆行情才有全部有效的當日特徵；資料不足時保留對應 NaN。特徵函式檢查日期與價量，但完整交易日及 hash 仍須先由 `load_snapshot` 驗證。

特徵測試使用離線合成行情，檢查獨立數值案例、暖機界線、漲跌與平盤，以及修改或附加未來行情不影響過去特徵。上述載入範例需要有效快照；正式行情的取得限制見下方「真實來源驗收狀態」。

### 時序資料集與縮放（Commit 4）

`build_dataset(prices, config)` 由原始行情建立三組特徵，選出共同有效樣本，依預測目標日期切分，再進行訓練集專用縮放：

```python
from lstm_comparison.dataset import build_dataset

dataset = build_dataset(prices, config)
group = dataset.groups["A"]
train = group.splits["train"]
validation = group.splits["validation"]
test = group.splits["test"]
original_prices = dataset.y_scaler.inverse_transform(test.y)
```

以上接續前節的 `load_config`／`load_snapshot` 範例，需要有效快照。`group.columns` 保存欄位順序，`group.x_scaler` 為該組的 X 縮放器；`dataset.y_scaler` 是三組共用的 y 縮放器。每個分區提供：

| 屬性 | 內容與形狀 |
| --- | --- |
| `X` | 縮放後 numpy 陣列 `(樣本數, lookback, 特徵數)`；正式 lookback 為 60 |
| `y` | 縮放後下一交易日 Close，numpy 陣列 `(樣本數, 1)` |
| `target_dates`／`input_end_dates` | 預測目標日期／最後輸入日期，均為 DatetimeIndex |
| `y_original`／`previous_close` | 原價口徑的目標 Close／前一交易日 Close，均為 numpy 陣列 `(樣本數, 1)` |

原行情位置 `j` 的目標只使用 `[j-lookback:j]` 輸入，標籤為 `Close[j]`，Naive 及方向評估的基準為 `Close[j-1]`。三組共用相同目標日期與標籤；每個原始連續視窗都必須在所有組具有完整有限值，不先刪除缺值列再拼接歷史。固定設定下 C 有 33 筆暖機，第一個共同樣本使用零起算位置 33–92，預測位置 93，也就是第 94 筆行情；最後一個目標為最後一筆已知行情。

年度分區只看目標日期：2015–2022 年為 train，2023 年為 validation，2024–2025 年為 test。先建立視窗再分區，保留跨年度已知歷史，亦不額外產生週末或休市日。每天使用當時可得的真實歷史行情，沒有將前一天預測值當作後續輸入。

各組 X 的 `MinMaxScaler(feature_range=(0,1), clip=False)` 僅以**共同訓練樣本實際輸入位置的聯集**擬合，每個行情位置只計一次，不包含暖機列或只作為最後訓練標籤的行情列。共用 y 縮放器只以共同訓練目標 Close 擬合。因此 X 與 y 的最值範圍可以不同；驗證與測試不重新擬合，超出訓練範圍時允許值小於 0 或大於 1，反正規化仍能還原原價。縮放器保留在回傳物件中，供後續訓練流程保存。

設定需先由 `load_config` 驗證，保留可配置正整數 lookback 供測試，horizon 固定為 1。無效行情、暖機後沒有共同樣本，或任一分區沒有共同樣本時，明確拋出 `DatasetError`。資料集測試使用離線合成行情驗證索引、跨年、共同日期、縮放 fit 範圍及反正規化；正式資料快照仍受下節記錄的來源問題限制。

### 共用模型與重複訓練（Commit 5）

`run` 先以 `load_snapshot` 離線檢查快照並建立資料集，通過後才在設定的 `output_dir` 下建立新的唯一實驗目錄；缺少或無效快照不建立實驗目錄。正式設定使用三組各 seeds 42、43、44，共 9 次獨立訓練，不自動下載或修復行情。

```powershell
# 將 matplotlib 的間接匯入快取放在專案內；不變更全域環境。
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tools/matplotlib'
.\.venv\Scripts\python.exe -m lstm_comparison run --config configs/research.json
```

三組都使用 `build_model((lookback, 特徵數), config, seed)`：單向 `LSTM(64) → Dense(1)`、Adam learning rate 0.001、MSE。每次清除 Keras 狀態、重設 Python／NumPy／TensorFlow seed，再建立新的模型與 optimizer，固定於 CPU 並啟用 deterministic ops；不同系統與套件版本仍可能有數值差異。輸入為 1／5／10 欄時，預設參數量分別為 16,961／17,985／19,265，會在訓練紀錄保存。

`train_one(group, config, seed, output_dir)` 使用 train 及 validation，每批 32 筆、最多 100 epochs，保持時間順序，`shuffle=False`；透過有限執行緒池的 `tf.data` 順序批次避免多次訓練累積大型執行緒池。早停只監看 `val_loss`，`patience=10`、`restore_best_weights=True`。無論提前停止或跑到最大 epochs，都在還原最佳驗證權重後保存 `best.weights.h5`。不把 test 傳入 fit、用於選權重或挑種子；只額外確認最佳權重的驗證預測為有限值。

每個實驗目錄保存以下產物，目錄名稱包含 UTC 時間與隨機識別碼；再次執行另建目錄，不覆寫既有結果：

```text
<output_dir>/<run_id>/
  config.json                 # 已解析的設定
  snapshot.metadata.json      # 原始快照 metadata，包含行情與交易日 hash
  environment.json           # Python、系統、套件版本、CPU／deterministic 設定
  run.json                   # stage=training、狀態、種子與各次訓練摘要
  scalers/
    X_A.pkl, X_B.pkl, X_C.pkl, y.pkl
    description.json         # 欄位順序與訓練最值
  A/seed_42/
    best.weights.h5
    history.csv              # epoch、loss、val_loss
    training.json            # seed、參數量、最佳 epoch、實際 epochs、時間與狀態
  ...                        # 其餘組別／種子
```

縮放器以標準庫 pickle 保存已擬合物件，須在相容環境載入自己的實驗產物。模型權重由既有忽略規則排除，設定、metadata 與精簡文字產物可納入版本控制。重新建立相同形狀的模型後，可使用 `model.load_weights(path, objects_to_skip=[model.optimizer])` 讀回推論權重，`dataset.y_scaler.inverse_transform(...)` 還原價格。載入時略過 optimizer 狀態；最佳模型權重的保存不代表提供中斷續訓。

`run.json` 的 `stage="training", status="completed"` 僅表示全部設定的訓練已完成。任何非有限 loss／驗證預測或保存失敗都回傳錯誤並保留 partial 產物及 `failed` 狀態，供查明原因；不提供自動續跑或覆寫選項。逐日預測、Naive 及績效由獨立評估指令產生，圖表與中文報告再由 `report` 整理保存產物。

測試用暫存的合成快照離線完成 1／5／10 維短訓練及產物驗證，另外以受控驗證 loss 確認早停與最大 epochs 結束都還原最佳權重，並檢查重新載入權重後預測一致。9 次排程亦有獨立測試。縮短測試 epochs／seeds 的設定與合成行情不代表正式研究完成；正式快照尚未取得，以下來源限制仍適用。

### 測試集預測與績效比較（Commit 6）

對 `run` 回傳的既有訓練目錄執行以下指令；角括號需換成實際路徑。`run` 維持只訓練，`evaluate` 不重新訓練、不下載，也不以測試績效選種子：

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tools/matplotlib'
.\.venv\Scripts\python.exe -m lstm_comparison evaluate --run-dir <實驗目錄>
```

亦可呼叫 `evaluate_experiment(run_dir)`，回傳 `<實驗目錄>/evaluation`。評估先檢查完整的 A／B／C × 設定種子清單、各子訓練完成狀態、欄序、參數量、權重及歷程檔；目前行情與交易日 hash 必須符合訓練保存的 metadata。載回 pickle 的 X／y 縮放器，檢查欄數、訓練最值、縮放設定與保存描述後，只做 transform，不重新 fit。

資料集 API `build_dataset(prices, config, x_scalers=..., y_scaler=...)` 支援這條載回路徑；X 字典必須包含三組且與 y 縮放器一起提供。省略兩者時維持訓練專用 fit 的原行為。包含 `copy=False` 的合法縮放器也使用獨立輸入副本，保留原行情及原價標籤。

只對 **test 分區**載回各種子的最佳權重，維持原順序分批推論，再以保存的 y 縮放器反正規化。三組所有種子與 Naive 共用完全相同的目標日期、原價真值與前一交易日 Close；這兩種真值直接取自原始行情，沒有以反縮放近似值代替。Naive 的逐日預測固定等於該目標之前一交易日 Close，只保存一份。

| 指標 | 公式及口徑 |
| --- | --- |
| MAE | `mean(abs(predicted_close-actual_close))`，主要比較指標，單位為調整後價格 |
| RMSE | `sqrt(mean((predicted_close-actual_close)^2))`，單位為調整後價格 |
| directional_accuracy | 比較 `sign(predicted_close-previous_close)` 與 `sign(actual_close-previous_close)`，保存 0–1 比例 |
| 誤差改善率 | `100×(Naive誤差-模型誤差)/Naive誤差`，分別計算 MAE／RMSE，單位為百分比；負值表示變差 |

方向明確分成 -1／0／1（下跌／持平／上漲），不加 epsilon 門檻；Naive 的預測變動雖固定為零，方向正確率仍依研究設計標示 N/A。每個種子保留績效，再按組別彙整算術平均與樣本標準差（`ddof=1`）。只有一個種子時標準差為 N/A；Naive 誤差為零時對應改善率為 N/A。Naive 彙總列的 `n_seeds=0`，不虛構重複實驗；沒有挑最佳種子或合成 ensemble。

```text
<實驗目錄>/evaluation/
  predictions_A_seed_42.csv   # 各組／種子分開保存
  ...
  predictions_naive.csv       # 共用一份
  metrics_per_seed.csv        # 各種子及 Naive 績效
  metrics_summary.csv         # 三組平均、樣本標準差、改善率及 Naive 列
  environment.json
  evaluation.json            # 獨立狀態、資料／設定／權重／縮放器 hash、樣本期間與績效
```

逐日 CSV 欄位為 `target_date,input_end_date,group,seed,actual_close,previous_close,predicted_close,absolute_error,squared_error,actual_direction,predicted_direction,direction_correct`，可直接重算績效。Naive 的 `seed` 與 `direction_correct` 為 N/A，`predicted_direction=0`。每種子表含 `mae,rmse,directional_accuracy` 及兩種誤差改善率；彙總欄位以 `_mean`／`_std` 標示。CSV 的未定義值寫成 `N/A`，JSON 寫成 `null`，不保存 NaN／Infinity。

輸入驗證通過後才建立固定 `evaluation/`，既有評估目錄一律拒絕覆寫。評估狀態獨立於訓練的 `run.json`；推論、輸出或有限數值檢查失敗會保留 partial 產物並標記 `failed`，不改寫原訓練狀態。逐日平方誤差超出浮點表示範圍也會明確失敗。成功僅代表指定訓練目錄的測試集評估完成，圖表與中文報告另由以下 `report` 指令產生。

離線測試包含手算績效、三分類與持平、9 次評估排程、CSV 績效獨立重算、來源／權重／縮放器錯誤及失敗狀態；另以合成快照完成 A／B／C seed42 短訓練後，重新載入真實權重及縮放器評估。這些測試不代表正式 9 次研究已完成；正式 Yahoo 快照仍受下節限制。

### 研究圖表與中文報告（Commit 7）

對同一個已完成訓練與評估的實驗目錄執行：

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tools/matplotlib'
.\.venv\Scripts\python.exe -m lstm_comparison report --run-dir <實驗目錄>
```

`generate_report(run_dir)` 只讀保存的設定、來源 metadata、訓練歷程與評估 CSV／JSON，回傳 `<實驗目錄>/report`。不載入 TensorFlow、不讀 pickle、不重訓、不重新預測或下載；即使原始快照、權重與縮放器已搬離原路徑，也可由保存結果產報。來源權重／縮放器的 hash 沿用評估紀錄作追溯，不假裝本步驟重新驗證了缺少的原檔。

產報前檢查訓練與評估完成狀態、確切組別／種子、設定／行情／交易日 hash 與來源參數，再從逐日價格重算誤差、方向、每種子及彙總指標，核對 CSV 與 evaluation manifest。所有模型與 Naive 的日期、原價真值與前收盤須完全對齊；已知相鄰交易日的前收盤也須等於上一筆真值。訓練歷程須為連續 epoch 1…n、有限非負 loss，實際 epochs、最佳 epoch／val_loss 必須與兩份訓練紀錄一致。缺檔、篡改或矛盾資料會在建立報告目錄前明確失敗。

```text
<實驗目錄>/report/
  report.md                  # 中文實驗報告，八個章節
  report.json                # 狀態、環境、所有報告輸入／輸出 hash、原預測來源 hash
  figures/
    prices.png
    errors.png
    direction.png
    loss.png
```

圖表採 matplotlib 的離線 Agg 輸出 PNG，不開視窗；圖中用清楚的日期、單位、種子與英文軸標，中文內容置於 Markdown 報告：

| 圖檔 | 呈現規則 |
| --- | --- |
| `prices.png` | 實際價格、三組 **seed 42** 預測及 Naive；固定42，不挑最佳、不合成ensemble |
| `errors.png` | 全部設定種子的 MAE／RMSE 平均與樣本std，Naive為單份基準；單seed不假造0 std |
| `direction.png` | 全部設定種子的方向正確率平均與樣本std，以百分比顯示；Naive只標N/A，不畫0%柱 |
| `loss.png` | A／B／C三個面板，全部設定seed的train／val loss，單位為縮放後Close的MSE |

若 `evaluation.plot_seed` 不是42或實驗沒有42，會拒絕產報，避免以其他種子替代。台積電 `2330.TW` 的價格標TWD；其他ticker使用來源metadata幣別，缺少幣別時用中性 `source price unit`，不套用TWD。

報告依序包含實驗概況、資料與方法、模型與實際訓練設定、逐seed／彙總績效、四圖、RQ1單變量對OHLCV／RQ2技術指標影響／RQ3與Naive比較、限制、重現及來源。結果文字由本次MAE及逐seed比較產生，同時列出RMSE與方向，不預設多變量勝出；平手、變差、負改善率與種子不一致都會保留。單seed樣本std、零Naive誤差的改善率與Naive方向仍顯示N/A。

限制包括單股／固定期間、未觀測因素與突發事件、來源資料品質與修訂、有限種子、模型參數量差異；不宣稱統計顯著、普遍因果或交易報酬。報告明示本次實際日期、epochs與seeds，短測試／非預設設定不能冒充正式研究；任何情況都不自動宣告Commit8已完成。

固定 `report/` 一律拒絕覆寫；輸出失敗保留partial及`failed`狀態，原訓練與評估紀錄不變。`report.json` 的`stage="report",status="completed"`只代表這次報告生成成功。離線測試含勝負／平手／種子不一致、單seed／零基準、來源矛盾與篡改、固定42、圖片／相對連結／hash，以及移除原行情與模型檔且禁止推論／fit／下載後仍可生成；另已實際檢視四張合成示範圖。正式資料與9次訓練仍須在來源問題解決後依Commit8執行。

### 行情快照與交易日完整性

預設快照由以下三個檔案組成，納入版本控制：

- `res/research/2330.TW_2015-2025.csv`：固定欄位 `Date,Open,High,Low,Close,Volume`；Date 為台北交易日的 ISO 日期。
- `res/research/2330.TW_2015-2025.sessions.csv`：研究期間預期的交易日。
- `res/research/2330.TW_2015-2025.metadata.json`：下載 UTC 時間、完整來源參數、Python／套件版本、筆數、日期邊界，以及行情與交易日 CSV 的 SHA256。

行情使用 yfinance 的 `auto_adjust=True`，保留來源成交量，不進行額外價格修復、補值或刪除缺值列。資料檢查拒絕空結果、重複日期或欄位、多股票混入、缺欄位、缺值、非有限數值、非正價格、不合理 High/Low，以及非正整數成交量。本研究要求有效交易日 `Volume > 0`；已知 Yahoo 存在 TWSE 實際有成交卻回傳零量的日期，因此不能將零量樣本視為正常行情。

預期交易日使用鎖定的 [exchange_calendars 4.13.3 XTAI](https://pypi.org/project/exchange_calendars/4.13.3/)，加上經 TWSE 官方資料核實的明列修正，研究區間共 **2,679 個預期交易日**：

- 額外休市：2022/02/04（[春節公告](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080817d22b9cb017e3336c4e203e3.pdf)）、2023/01/18（[當月市場成交資料](https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20230101&response=json)顯示 01/17 後至 01/30 才恢復交易）、2024/10/31（[康芮颱風公告](https://wwwc.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e30192dd523eb10329.pdf)）。
- 額外開市：2016/01/30、2016/06/04、2016/09/10、2017/02/18、2017/06/03；各日均由下表 TWSE 個股資料確認有正成交量及有效 OHLC，不能因落在週六而排除。

每筆修正的日期、原因與官方來源列於 `data.py`，成功凍結時一併寫入 metadata。行情的起訖及中間每個交易日都必須與預期日期完全相同；不使用單純工作日推測休市，也不根據 Yahoo 缺資料的日期自行刪除預期交易日。

交易日 CSV 與日曆來源版本一同凍結。後續離線讀取核對凍結日期及 hash，不因目前安裝的日曆版本不同而改寫歷史。來源資料仍受 Yahoo 的歷史資料修訂及日曆規則正確性限制，任何日期差異都應查明原因後再建立新快照。

快照先在同目錄暫存，完整寫入後以不覆寫的方式發布；metadata 最後發布，作為完整快照標記。一般寫入錯誤只清理本次建立的檔案，程序中斷可能留下部分檔案；讀取時會明確拒絕這類殘留，不將它當成成功快照，也不自動刪除或修復。遇到不完整或損壞快照，先保留檔案查明原因，或在設定中使用新的 `snapshot_path` 重新下載，沒有 `--force` 覆寫選項。

搬移快照時需一併搬移三個同名檔案，並更新設定中的 `snapshot_path`。離線檢查只比對資料來源參數，不綁定原始絕對路徑或模型訓練參數。CSV 固定使用 LF；`.gitattributes` 僅針對 `res/research/*.csv` 保留 LF，避免 Windows 換行轉換破壞 SHA256。搬移到其他版控目錄時，也需保留此換行規則。

### 真實來源驗收狀態

**Commit 2 的下載與離線快照程式已實作；正式行情快照尚未完成。** 使用者已選擇維持 Yahoo 與嚴格檢核，先交付程式；不以其他來源或補值繞過資料品質問題。

2026/11/19 實際向 Yahoo 取得 **2,682 筆**，起訖為 2015/01/05–2025/12/31，但無法通過獨立交易日與數值檢查。相對於 2,679 個預期交易日，來源缺少 2 日、多出 5 個非交易日；另有 6 個有效交易日 OHLC 全為 NaN，以及 1 個有效交易日為平價零量。缺少或價量異常的有效交易日合計 **9 日**，不含那 5 個非交易占位日。

| 日期 | Yahoo 狀況 | TWSE 官方個股證據 |
| --- | --- | --- |
| 2019/09/09 | 缺少該日 | [2019 年 9 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20190901&response=json&stockNo=2330)：17,317,833 股，OHLC 265.50／266.00／263.50／265.00 |
| 2021/04/06 | 缺少該日 | [2021 年 4 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20210401&response=json&stockNo=2330)：37,664,216 股，OHLC 615／616／608／610 |
| 2016/01/30 | OHLC 全 NaN、Volume 0 | [2016 年 1 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20160101&response=json&stockNo=2330)：11,865,065 股，OHLC 142.50／143.00／141.00／142.00 |
| 2016/06/04 | OHLC 全 NaN、Volume 0 | [2016 年 6 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20160601&response=json&stockNo=2330)：2,631,269 股，OHLC 160／160／159.50／160 |
| 2016/09/10 | OHLC 全 NaN、Volume 0 | [2016 年 9 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20160901&response=json&stockNo=2330)：15,470,223 股，OHLC 177／179／176／178 |
| 2017/02/18 | OHLC 全 NaN、Volume 0 | [2017 年 2 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20170201&response=json&stockNo=2330)：3,517,102 股，OHLC 190／190.50／189.50／189.50 |
| 2017/06/03 | OHLC 全 NaN、Volume 0 | [2017 年 6 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20170601&response=json&stockNo=2330)：1,535,230 股，OHLC 209／209／208／209 |
| 2025/08/01 | OHLC 全 NaN、Volume 0 | [2025 年 8 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20250801&response=json&stockNo=2330)：45,352,225 股，OHLC 1,145／1,150／1,130／1,140 |
| 2021/08/17 | OHLC 全相同、Volume 0 | [2021 年 8 月](https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20210801&response=json&stockNo=2330)：31,845,499 股，OHLC 580／582／578／580 |

Yahoo 多出的非交易占位日為 2015/07/10、2015/09/29、2016/07/08、2016/09/27、2016/09/28，皆為平價零量。已核對 TWSE 市場月成交資料：[2015 年 7 月](https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20150701&response=json)、[2015 年 9 月](https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20150901&response=json)、[2016 年 7 月](https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20160701&response=json)、[2016 年 9 月](https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20160901&response=json)。程式仍拒絕這些列，不自動刪除來使資料過關。

針對 2019/09/09、2021/04/06、2021/08/17、2025/08/01，再以有界短日期區間比較 Yahoo `download` 與 `Ticker.history`，結果均重現同一異常。上述 TWSE 未調整價格僅作查證證據，沒有拿來補入調整後行情。下載指令會明確非零拒絕並列出缺日、多日與無效日期，因此 `res/research/` 下沒有正式行情、sessions 或 metadata 三檔；診斷回應僅留在被忽略的 `.tools/diagnostics/`。
