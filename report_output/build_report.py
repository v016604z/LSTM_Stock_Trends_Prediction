#!/usr/bin/env python
"""Compose the evidence-based eight-page report from frozen repository outputs."""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
LSTM = ROOT / "results/20261120T134010119780Z_82eaad21/evaluation"
GRU = ROOT / "results/gru_20261120T143901617329Z_1fa00b5e/evaluation"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_figures():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    font = FontProperties(fname=str(font_path))
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = font.get_name()

    rows = read_csv(GRU / "comparison_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
    groups = ["A", "B", "C"]
    x = list(range(3))
    width = 0.34
    for offset, model, color in [(-width / 2, "LSTM", "#527B99"), (width / 2, "GRU", "#C17B52")]:
        selected = [next(r for r in rows if r["group"] == g and r["model"] == model) for g in groups]
        means = [float(r["mae_mean"]) for r in selected]
        stds = [float(r["mae_std"]) for r in selected]
        ax.bar([v + offset for v in x], means, width, yerr=stds, capsize=3,
               color=color, edgecolor="white", label=model, linewidth=0.5)
    naive = float(next(r["mae_mean"] for r in read_csv(LSTM / "metrics_summary.csv") if r["group"] == "Naive"))
    ax.axhline(naive, color="#333333", linewidth=1.2, linestyle="--", label=f"Naive {naive:.1f}")
    ax.set_xticks(x, ["A Close", "B OHLCV", "C OHLCV+指標"], fontproperties=font)
    ax.set_ylabel("MAE（新臺幣元）", fontproperties=font)
    ax.set_title("三種特徵組合的 MAE：平均值 ± 三個種子樣本標準差", fontproperties=font, fontsize=10)
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.6)
    ax.legend(frameon=False, ncol=3, prop=font, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "mae_models.png", dpi=200, facecolor="white")
    plt.close(fig)

    def map_rows(path):
        return {r["target_date"]: r for r in read_csv(path)}
    naive_rows = map_rows(LSTM / "predictions_naive.csv")
    dates = sorted(naive_rows)
    actual = [float(naive_rows[d]["actual_close"]) for d in dates]
    prev = [float(naive_rows[d]["previous_close"]) for d in dates]
    pred_lines = {}
    for model, base, pat in [
        ("LSTM-A 三種子平均", LSTM, "predictions_A_seed_{}.csv"),
        ("GRU-A 三種子平均", GRU, "predictions_A_seed_{}.csv"),
    ]:
        maps = [map_rows(base / pat.format(seed)) for seed in (42, 43, 44)]
        pred_lines[model] = [sum(float(m[d]["predicted_close"]) for m in maps) / 3 for d in dates]
    fig, ax = plt.subplots(figsize=(7.5, 3.3), constrained_layout=True)
    ax.plot(dates, actual, color="#202D38", linewidth=1.45, label="實際收盤價")
    ax.plot(dates, prev, color="#8B939B", linewidth=0.9, alpha=0.85, label="Naive 前日收盤")
    ax.plot(dates, pred_lines["LSTM-A 三種子平均"], color="#527B99", linewidth=1.0, label="LSTM-A 三種子平均")
    ax.plot(dates, pred_lines["GRU-A 三種子平均"], color="#C17B52", linewidth=1.0, label="GRU-A 三種子平均")
    ticks = list(range(0, len(dates), 60))
    ax.set_xticks(ticks, [dates[i][:7] for i in ticks], rotation=0)
    ax.set_ylabel("收盤價（新臺幣元）", fontproperties=font)
    ax.set_title("測試期 2024–2025 價格序列（線條作視覺對照）", fontproperties=font, fontsize=10)
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, prop=font, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "price_series.png", dpi=200, facecolor="white")
    plt.close(fig)


def build_spec():
    def h(text, level=2): return {"type": "heading", "text": text, "level": level}
    def p(text): return {"type": "paragraph", "text": text}
    def tab(rows): return {"type": "table", "rows": rows}
    def src(text, url): return {"type": "source", "text": text, "url": url}
    def img(path, height, width=6.25): return {"type": "image", "path": path, "height_in": height, "width_in": width}

    repo = "https://github.com/v016604z/LSTM_Stock_Trends_Prediction/tree/e92259e80fc93e94f46ca7121363e492d1016afd"
    twse = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20250101&stockNo=2330"
    results = "https://github.com/v016604z/LSTM_Stock_Trends_Prediction/blob/e92259e80fc93e94f46ca7121363e492d1016afd/docs/LSTM%20%E8%88%87%20GRU%20%E8%82%A1%E5%83%B9%E9%A0%90%E6%B8%AC%E6%AD%A3%E5%BC%8F%E5%AF%A6%E9%A9%97%E7%B5%90%E6%9E%9C.md"
    data_code = "https://github.com/v016604z/LSTM_Stock_Trends_Prediction/blob/e92259e80fc93e94f46ca7121363e492d1016afd/src/lstm_comparison/data.py"
    lstm_paper = "https://doi.org/10.1162/neco.1997.9.8.1735"
    gru_paper = "https://arxiv.org/abs/1406.1078"

    pages = [
      {"title":"臺灣積體電路製造股份有限公司\nLSTM 與 GRU 股價預測實驗報告", "blocks":[
        p("研究摘要｜本報告根據專案保存的正式實驗結果，比較以台積電（2330）日資料預測下一交易日收盤價的 LSTM 與 GRU，並檢驗 Close、OHLCV、OHLCV 加技術指標三組輸入。正式資料來自臺灣證券交易所（TWSE）月資料快照，共 2,682 列，涵蓋 2015–2025 年；模型在同一 2024–2025 測試集比較。研究問題不預設深度模型必然優於基準，並以前一交易日收盤價作為 Naive 基準。 [1–3]"),
        h("一、資料蒐集與前處理", 1),
        p("研究標的是台積電普通股（2330），以 TWSE STOCK_DAY 歷史月資料端點逐月取得 Open、High、Low、Close、Volume，並保留交易日期及交易日曆。資料凍結成 CSV 與來源 metadata；metadata 記錄 132 個月份請求與來源雜湊，載入時可離線核對列數、日期與 SHA-256。Volume 單位為股，價格維持未還原名目值。 [1,2]"),
        tab([["資料段","目標日期範圍","樣本數","用途"],["訓練","2015–2022","1,865","權重與縮放器擬合"],["驗證","2023","239","早停監控"],["測試","2024–2025","485","最終評估"]]),
        p("樣本數是共同有效的視窗目標，不等於原始日資料列數；訓練原始 Close 共 1,958 筆，序列前置期與特徵暖機後，才形成表列 1,865 個訓練目標。"),
        src("[1] 專案固定版本與研究資料","https://github.com/v016604z/LSTM_Stock_Trends_Prediction/tree/e92259e80fc93e94f46ca7121363e492d1016afd"),
        src("[2] TWSE STOCK_DAY 官方資料端點","https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=20250101&stockNo=2330"),
      ]},
      {"title":"資料前處理與研究演進", "blocks":[
        h("一、資料蒐集與前處理（續）", 1),
        p("預測目標為下一交易日 Close；輸入回看最近 60 個交易日，預測期距為 1 日。所有特徵只使用當日及更早的資訊；共同視窗規則讓 A/B/C 與 Naive 在同一目標日期上比較。技術指標暖機造成的缺值不以未來值補齊，而是排除無法形成完整共同樣本的初段目標。"),
        p("資料依目標日期切分，而非隨機打散：訓練 2015–2022、驗證 2023、測試 2024–2025。測試邊界的輸入視窗允許使用預測當時已知的前段歷史。X 特徵縮放器只用訓練視窗資料擬合；共用 y 縮放器只用訓練目標擬合。驗證與測試僅轉換，不重新擬合。MinMaxScaler 設為 [0,1] 且不 clip，因此超出訓練範圍的測試值可轉成範圍外數值，屬設計行為。"),
        h("資料與程式的演進", 2),
        p("早期版本包含 LSTM/BiLSTM notebook 與爬蟲試作，資料管線、視窗與模型設定不同，不能與本次正式結果公平合併。後續形成三組特徵共用同一 LSTM 架構；檢查 Yahoo 相容來源的缺日與異常問題後，正式研究改採 TWSE 凍結快照。其後完成 9 組 LSTM（A/B/C × 三種子），再在相同設定及測試樣本上完成 9 組 GRU 比較。歷史 notebook 圖不作為本研究正式結果圖。"),
        h("特徵組合", 2),
        tab([["組別","輸入欄位","說明"],["A","Close","單變量價格歷史"],["B","Open、High、Low、Close、Volume","原始 OHLCV"],["C","OHLCV、MA5、MA20、RSI14、MACD、MACD signal","含技術指標"]]),
        p("MACD 使用 12/26/9 設定；RSI 為 Wilder 14 日。所有組別都使用 60 日視窗，以固定資料規模下比較輸入資訊差異。"),
      ]},
      {"title":"二、AI 模型建構", "blocks":[
        h("二、AI 模型建構", 1),
        p("LSTM 透過輸入、遺忘與輸出閘門及記憶狀態傳遞序列資訊，目的是改善一般循環網路長序列訓練的梯度問題；GRU 以更新與重設閘門整合狀態，通常參數較精簡。兩者都是單向循環網路，按時間順序讀入 60 日窗口，再輸出一個下一日 Close 預測。這是架構背景，不構成模型必勝的理由。[4,5]"),
        tab([["項目","LSTM","GRU"],["網路","循環層 64 單元 → Dense(1)","循環層 64 單元 → Dense(1)"],["參數量 A / B / C","16,961 / 17,985 / 19,265","12,929 / 13,697 / 14,657"],["共用訓練設定","Adam 0.001；MSE；batch 32；最多 100 epochs","同左"],["停止與重現","監控 val_loss；patience 10；回復最佳權重","同左；seed 42、43、44"]]),
        h("基準與評估量", 2),
        p("Naive 以預測日的前一交易日收盤價直接當預測值。MAE 是絕對誤差平均，RMSE 對大誤差給予較大權重；方向準確率（DA）依儲存庫三分類定義比較上漲、下跌與持平方向。Naive 的 DA 依官方結果定義為 N/A，不填 0%。所有模型數字均以三個種子的平均與樣本標準差（ddof=1）表示；DA 標準差以百分點呈現。"),
        p("LSTM 為正式第一階段，GRU 使用同一凍結資料與既有 scaler 做轉換，並沿用同一測試目標日期。A/B/C 是同一演算法的特徵變化，不是三種演算法；真正跨演算法比較是 LSTM 對 GRU。"),
        src("[4] Hochreiter & Schmidhuber, Long Short-Term Memory (1997)", "https://doi.org/10.1162/neco.1997.9.8.1735"),
        src("[5] Cho et al., RNN Encoder–Decoder (2014)", "https://arxiv.org/abs/1406.1078"),
      ]},
      {"title":"三、Python 程式實作與重現", "blocks":[
        h("三、Python 程式實作", 1),
        p("專案以模組化 Python 套件分離設定、資料驗證、特徵、視窗與縮放、模型訓練、評估及報表。`data.py` 載入凍結快照並驗證交易日、OHLCV、metadata 與雜湊；`features.py` 計算 A/B/C；`dataset.py` 建立時間序列樣本並限縮 scaler 擬合範圍；`model.py` 建構 LSTM/GRU；`training.py` 與 `gru_experiment.py` 執行重複訓練；`evaluation.py` 匯出逐日預測與指標。"),
        h("重現入口", 2),
        p("研究套件依 README 使用 Python 3.11 與鎖定環境。以下命令從專案根目錄先完成 LSTM、再選擇性啟動 GRU。既有有效 TWSE 快照可離線驗證及重用；若缺失，標準下載入口才會連線取回。"),
        {"type":"code","text":(OUT / "reproduce_study.py").read_text(encoding="utf-8").rstrip()},
        p("本次已完成的可重現性檢查包括：來源 ZIP CRC 檢查通過；安全解壓後 `validate-config` 通過；離線 `load-snapshot` 驗證 2,682 列、日期範圍與資料/日曆雜湊；driver `--help` 與 `py_compile` 通過。這些檢查確認程式與資料包可讀，沒有重新訓練或重算實驗指標。報告引用已保存的正式實驗 CSV 與模型結果，不把檢查包裝成新實驗。"),
        p("附帶 ZIP 含完整 `src/lstm_comparison/`、測試原始碼、設定、鎖定依賴、凍結資料與 metadata、正式摘要 CSV 及 driver；不含權重或快取。完整可執行 driver 列於第八頁。"),
      ]},
      {"title":"四、實際實驗與比較", "blocks":[
        h("四、實際實驗與比較", 1),
        p("正式 LSTM 實驗已完成 9 次擬合（A/B/C 各 3 個 seed）；GRU 比較亦完成 9 次，兩者均使用 485 個相同測試目標。下表為測試期三種子的平均 ± 樣本標準差；價格誤差單位為新臺幣元，DA 以百分比表示，括號內是百分點標準差。Naive 無 seed 分布，DA 依原定義 N/A。"),
        tab([["組別","模型","MAE（元）","RMSE（元）","DA（%）"],
             ["A","Naive","15.37","20.97","N/A"],
             ["A","LSTM","161.35 ± 54.89","185.06 ± 60.34","40.14 ± 0.12"],
             ["A","GRU","66.51 ± 4.86","80.79 ± 5.66","40.21 ± 0.00"],
             ["B","LSTM","201.04 ± 3.63","241.93 ± 4.22","40.62 ± 0.00"],
             ["B","GRU","87.06 ± 26.66","105.20 ± 28.36","40.27 ± 0.48"],
             ["C","LSTM","223.41 ± 106.61","265.09 ± 112.65","41.51 ± 1.37"],
             ["C","GRU","181.29 ± 35.59","211.97 ± 37.90","40.00 ± 0.00"]]),
        p("GRU 在三組的 MAE / RMSE 都低於對應 LSTM；相對 LSTM 的 MAE 降幅 A/B/C 為 58.78% / 56.70% / 18.86%。但兩種深度模型在所有組別仍都明顯輸給 Naive。A 組在 LSTM 與 GRU 各自的平均誤差中最低；加入更多原始欄位與技術指標沒有改善本次 LSTM 平均 MAE。結果僅描述本測試窗口與三種子，不代表所有市場期間的普遍效應。"),
        src("完整正式結果與逐種子表：[3]", "https://github.com/v016604z/LSTM_Stock_Trends_Prediction/blob/e92259e80fc93e94f46ca7121363e492d1016afd/docs/LSTM%20%E8%88%87%20GRU%20%E8%82%A1%E5%83%B9%E9%A0%90%E6%B8%AC%E6%AD%A3%E5%BC%8F%E5%AF%A6%E9%A9%97%E7%B5%90%E6%9E%9C.md"),
      ]},
      {"title":"誤差指標與測試期價格序列", "blocks":[
        h("四、實際實驗與比較（續）", 1),
        p("圖 1 對照三組特徵的 MAE 平均及種子間樣本標準差；Naive 虛線為固定參考。誤差棒描述 seed 變異，不是信賴區間。由於 Naive 的 MAE 遠低於模型，圖中模型柱高反映其與簡單持平基準之差距。"),
        img("figures/mae_models.png", 2.55),
        {"type":"caption","text":"圖 1　正式測試結果 MAE（A/B/C；LSTM、GRU 與 Naive）。誤差棒為三個種子的樣本標準差。"},
        p("圖 2 以 A 組逐日預測序列作直觀比較。為可讀性，LSTM-A 與 GRU-A 各顯示三個 seed 的預測平均；這只是繪圖摘要，並非正式 ensemble，也不用它替代依 seed 計算的表列績效。Naive 線為前一日 Close。"),
        img("figures/price_series.png", 2.85),
        {"type":"caption","text":"圖 2　測試期實際收盤價、Naive 與 A 組三種子預測平均；平均線僅供視覺化。"},
      ]},
      {"title":"五、研究分析與改進", "blocks":[
        h("五、研究分析與改進", 1),
        p("本次結果支持的描述是：相同資料切分下，GRU 的價格誤差小於 LSTM；然而所有 LSTM/GRU 組合的 MAE、RMSE 均高於 Naive，方向準確率約 40% 左右。故「GRU 較 LSTM 好」只限本次各組 MAE/RMSE 比較，不能解讀成模型已預測成功或具有投資價值。三種子樣本太少，沒有進行統計顯著性檢定。"),
        h("可能機制與證據界線", 2),
        p("訓練期間 Close 範圍為 115–683 元，測試期間為 576–1,550 元；485 個測試收盤中，457 個（94.23%）落在訓練範圍之外。另由逐日預測計算，18 個模型/seed 組合的平均 signed error 都為負，低估比例介於 90.9%–100%。這些是儲存結果上的描述統計；價格尺度擴張與低估現象可能相關，但本研究沒有因果檢定，不能聲稱價格轉變已證明造成誤差。"),
        p("未還原價格保留企業行動造成的名目價格落差，因此 Close MAE/RMSE 不等於總報酬衡量。三分類 DA 也不等於交易獲利。測試預測採逐日真實已知歷史輸入，不是把模型輸出遞迴餵回的多步預測。"),
        h("後續改進", 2),
        p("一、增加跨期間與其他股票的滾動時間切分，檢查分布變化下的穩健性。二、預先設計價格差分、報酬率或調整後價格實驗，並與持平及其他簡單基準同時比較。三、擴充隨機種子並以配對重抽樣或區塊 bootstrap 評估誤差差異。四、分開報告單步與真正遞迴多步預測。五、若研究投資決策，再預先定義交易成本與風險指標，避免以 DA 直接推論獲利。"),
        p("結論：TWSE 凍結快照、共同時間切分與三種子重複實驗建立了可追溯比較基礎。本輪資料中，GRU 優於 LSTM 的 MAE/RMSE，但兩者都未超越 Naive；更豐富特徵未自動帶來改善。後續應優先處理資料尺度、基準與外部驗證，再討論模型複雜度。"),
      ]},
      {"title":"附錄　完整重現入口與參考資料", "blocks":[
        h("完整可執行 driver：report_output/reproduce_study.py", 1),
        {"type":"code","text":(OUT / "reproduce_study.py").read_text(encoding="utf-8").rstrip()},
        h("參考資料", 2),
        src("[1] v016604z，LSTM Stock Trends Prediction，凍結研究版本 ebe92c0d。", repo),
        src("[2] 臺灣證券交易所，個股日成交資訊（STOCK_DAY）。", twse),
        src("[3] 專案正式 LSTM/GRU 實驗結果、指標與預測 CSV。", results),
        src("[4] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.", lstm_paper),
        src("[5] Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder. arXiv:1406.1078.", gru_paper),
        p("附註：完整研究程式與凍結資料另附 LSTM_GRU_reproducible_source.zip。正式數據取自專案已完成實驗；本報告未重新訓練模型。"),
      ]},
    ]
    # Apply final editorial controls to the page-level content model.
    pages[0]["title"] = "台積電股價預測研究報告\nLSTM與GRU正式實驗比較"
    for page_index in (2, 4, 6):
        page = pages[page_index]
        if page["blocks"] and page["blocks"][0].get("type") == "heading" and page["blocks"][0].get("text") == page["title"]:
            page["blocks"].pop(0)
    pages[3]["blocks"][0]["text"] = "程式架構"
    pages[3]["blocks"][0]["level"] = 2
    p3_open = next(block for block in pages[2]["blocks"] if block.get("type") == "paragraph")
    p3_open["text"] += " " + "\u672c\u7814\u7a76\u4e26\u5217\u6bd4\u8f03\u5177\u6709\u9577\u671f\u8a18\u61b6\u80fd\u529b\u7684 LSTM \u8207\u53c3\u6578\u8f03\u7cbe\u7c21\u7684 GRU\uff0c\u5728\u76f8\u540c\u6642\u5e8f\u4efb\u52d9\u4e0b\u6aa2\u8996\u8aa4\u5dee\u8207\u7a2e\u5b50\u7a69\u5b9a\u6027\u3002"

    # Markdown emphasis markers are unnecessary in this native Word layout.
    def remove_markdown(value):
        if isinstance(value, str):
            return value.replace("`", "")
        if isinstance(value, list):
            return [remove_markdown(item) for item in value]
        if isinstance(value, dict):
            return {key: remove_markdown(item) for key, item in value.items()}
        return value
    pages = remove_markdown(pages)

    # Keep the naive baseline explicit as common to all three feature groups.
    results_table = next(block for block in pages[4]["blocks"] if block.get("type") == "table")
    for row in results_table["rows"]:
        if len(row) > 1 and row[1] == "Naive":
            row[0] = "共用"

    # Format the experiment table directly from authoritative summary CSVs.
    lookup = {}
    for row in read_csv(LSTM / "metrics_summary.csv"):
        if row["group"] != "Naive":
            lookup[(row["group"], "LSTM")] = row
    for row in read_csv(GRU / "comparison_summary.csv"):
        lookup[(row["group"], row["model"])] = row
    metrics_table = results_table["rows"]
    for row in metrics_table[1:]:
        if row[1] == "Naive":
            continue
        source = lookup[(row[0], row[1])]
        row[2] = f"{float(source['mae_mean']):.2f} ± {float(source['mae_std']):.2f}"
        row[3] = f"{float(source['rmse_mean']):.2f} ± {float(source['rmse_std']):.2f}"
        row[4] = f"{float(source['directional_accuracy_mean'])*100:.2f} ± {float(source['directional_accuracy_std'])*100:.2f}"

    p4_driver = next(block for block in pages[3]["blocks"] if block.get("type") == "code")
    p4_driver["text"] = "uv sync --locked\\n.venv\\Scripts\\python.exe report_output\\reproduce_study.py --config configs\\research.json --gru"
    # Appendix carries the complete functional driver, stripped only of prose and blank lines.
    driver = (OUT / "reproduce_study.py").read_text(encoding="utf-8").splitlines()
    compact = []
    in_docstring = False
    for line in driver:
        if line.strip().startswith('"""'):
            in_docstring = not in_docstring
            continue
        if in_docstring or not line.strip() or line.lstrip().startswith("#"):
            continue
        compact.append(line.rstrip())
    pages[7]["blocks"][1]["text"] = "\n".join(compact)

    # The cited content is pinned to the inspected local snapshot, not a claim of current upstream HEAD.
    pages[7]["blocks"][-1]["text"] += "報告依最新核對的 upstream main e92259e 撰寫；該次更新僅修改 README 與文件，實驗程式、設定、資料、結果與本地 ebe92c0 研究包相同。"
    # Final executable paths and provenance labels.
    pages[0]["blocks"].insert(0, {"type":"paragraph", "text":"\u6574\u7406\u65e5\u671f\uff1a2026\u5e7411\u670820\u65e5"})
    p4_intro = next(block for block in pages[3]["blocks"] if block.get("type") == "paragraph" and "README" in block.get("text", ""))
    p4_intro["text"] += " " + "\u8acb\u5148\u5c07\u9644\u4ef6 ZIP \u89e3\u58d3\u5230\u65b0\u76ee\u9304\uff0c\u5728\u9644\u4ef6\u6839\u76ee\u9304\u5b89\u88dd Python 3.11 \u8207 uv\uff0c\u518d\u57f7\u884c\u4e0b\u5217\u547d\u4ee4\u3002"
    p4_driver = next(block for block in pages[3]["blocks"] if block.get("type") == "code")
    p4_driver["text"] = chr(10).join(["uv sync --locked", r".venv\Scripts\python.exe reproduce_study.py --gru"])
    pages[7]["blocks"][0]["text"] = "\u5b8c\u6574\u53ef\u57f7\u884c driver\uff1areproduce_study.py"
    for block in pages[7]["blocks"]:
        if block.get("type") == "source" and block.get("text", "").startswith("[1]"):
            block["text"] = "[1] v016604z, upstream main e92259e; research source/result package matches local ebe92c0."
            break
    return {"pages": pages}


if __name__ == "__main__":
    build_figures()
    from generate_docx_native import build
    data = build_spec()
    spec_path = OUT / "report_content.json"
    spec_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target = OUT / "LSTM_GRU_研究報告.docx"
    build(data, target, OUT)
    print(f"DOCX created: {target}")
