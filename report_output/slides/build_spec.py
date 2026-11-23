"""Prepare images and flat editable COM slide spec. Run from repository root with .venv Python."""
from pathlib import Path
import csv,json,math,shutil
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent; A=O/'assets'; A.mkdir(exist_ok=True)
L=R/'results/20261120T134010119780Z_82eaad21/evaluation'; G=R/'results/gru_20261120T143901617329Z_1fa00b5e/evaluation'
shutil.copy2(R/'report_output/figures/price_series.png',A/'price_series.png')
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
lm={x['group']:x for x in rows(L/'metrics_summary.csv')}; gm={x['group']:x for x in rows(G/'comparison_summary.csv') if x['model']=='GRU'}
naive_rows=rows(L/'predictions_naive.csv')
naive_mae=sum(float(r['absolute_error']) for r in naive_rows)/len(naive_rows)
naive_rmse=math.sqrt(sum(float(r['squared_error']) for r in naive_rows)/len(naive_rows))

NAVY='#142B4A'; BLUE='#2878B5'; ORANGE='#E98235'; TEAL='#15968C'; GRAY='#738092'; PALE='#F3F6F9'; INK='#172A3A'; MUTED='#526579'; WHITE='#FFFFFF'
def txt(x,y,w,h,text,size=20,color=INK,bold=False,align='left'):
 return {'type':'text','x':x,'y':y,'w':w,'h':h,'text':text,'font_size':size,'bold':bold,'color':color,'align':align,'font_name':'Microsoft JhengHei'}
def rect(x,y,w,h,fill=PALE,line='#E2E8EF'):
 return {'type':'rect','x':x,'y':y,'w':w,'h':h,'fill':fill,'line_color':line,'line_width':1}
def line(x1,y1,x2,y2,color=GRAY,width=1):return {'type':'line','x1':x1,'y1':y1,'x2':x2,'y2':y2,'color':color,'width':width}
def img(path,x,y,w,h):return {'type':'image','path':str((O/path).resolve()),'x':x,'y':y,'w':w,'h':h}
def box(x,y,w,h,title,body,accent=BLUE,body_size=18):return [rect(x,y,w,h),rect(x,y,8,h,accent,accent),txt(x+20,y+10,w-35,30,title,19,accent,True),txt(x+20,y+45,w-36,h-50,body,body_size)]
def slide(title,source,notes,elements):return {'title':title,'source':source,'notes':notes,'elements':[rect(0,0,960,10,NAVY,NAVY),txt(38,22,880,45,title,28,NAVY,True),*elements,line(40,511,920,511,'#DCE3EA',1),txt(42,514,810,17,source,10,MUTED),txt(875,514,45,17,'',10,MUTED,False,'right')]}
S=[]
# 1
S.append({'title':'','source':'資料：臺灣證券交易所 2330.TW；正式實驗結果見專案文件與逐頁來源說明','notes':'開場：本研究比較 LSTM 與 GRU 在三種特徵下預測台積電次一交易日 Close。問題是更多特徵和更複雜記憶結構是否一定降低誤差。課程與報告資訊均為可編輯佔位。','elements':[rect(0,0,960,540,'#F5F7FA','#F5F7FA'),rect(0,0,960,12,NAVY,NAVY),rect(48,65,10,145,BLUE,BLUE),txt(82,70,825,83,'LSTM 與 GRU 於股票價格預測之績效比較',32,NAVY,True),txt(84,160,815,45,'比較兩種循環網路、三組特徵，預測台積電次一交易日收盤價',21,MUTED),*box(82,238,390,95,'研究問句','特徵越多、記憶結構越複雜，預測誤差就越低嗎？',TEAL,18),*box(500,238,375,95,'研究目的','在相同時間切分與訓練設定下，比較 LSTM、GRU 與前日收盤基準。',ORANGE,17),txt(84,370,760,70,'課程：[請填寫]   報告者：[請填寫]\n指導教師：[請填寫]   報告日期：[請填寫]',18,INK)]})
# 2
e=[]
for x,w,t,b,c in [(50,250,'資料量','2,682 列\nOHLCV 未調整價',BLUE),(340,250,'期間','2015-01 至 2025-12\n交易日序列',TEAL),(630,280,'完整性','日期、交易日曆與 SHA-256\n依正式 snapshot metadata 核對',ORANGE)]:e+=box(x,92,w,86,t,b,c,16)
e += [txt(55,195,820,30,'正式 raw snapshot 抽樣（可編輯資料表）',18,NAVY,True)]
raw=rows(R/'res/research/2330.TW_2015-2025_twse.csv'); heads=['Date','Open','High','Low','Close','Volume']
for ri,vals in enumerate([heads]+[[r.get(h,'') for h in heads] for r in raw[:4]]):
 for ci,val in enumerate(vals):
  x=55+ci*142; y=230+ri*42; e += [rect(x,y,142,42,NAVY if ri==0 else ('#F3F6F9' if ri%2 else WHITE),'#D9E1E8'),txt(x+4,y+5,134,31,str(val),16,WHITE if ri==0 else INK,ri==0,'center')]
S.append(slide('資料快照：2,682 個交易日，涵蓋 2015–2025','來源：TWSE STOCK_DAY（2330）；run snapshot 與資料快照完整性紀錄','資料來源為 TWSE STOCK_DAY，代號 2330。抽樣表來自 res/research/2330.TW_2015-2025_twse.csv 前四筆。snapshot metadata 含來源 URL、月資料 cache SHA-256、資料與日曆 SHA-256。',e))
# 3
e=[]
for x,t,b,c in [(48,'A｜Close（1）','60 日 Close → 下一交易日 Close',BLUE),(355,'B｜OHLCV（5）','Open、High、Low、Close、Volume',TEAL),(662,'C｜OHLCV + 指標（10）','MA5、MA20、RSI14、MACD、Signal',ORANGE)]:e+=box(x,92,255,112,t,b,c,16)
e += [txt(50,225,850,30,'依時間切分（未隨機打散）',18,NAVY,True)]
for x,t,b,c in [(50,'Train｜2015–2022','1,865 筆',BLUE),(345,'Validation｜2023','239 筆',TEAL),(640,'Test｜2024–2025','485 筆',ORANGE)]:e += [rect(x,270,260,68,c,c),txt(x+5,278,250,52,t+'\n'+b,17,WHITE,True,'center')]
e += [txt(54,365,850,60,'MinMaxScaler 僅 fit train；validation 用於 early stopping；test 僅作最終評估。\n指標：MA5/20、Wilder RSI14、MACD(12,26,9)。',17,INK)]
S.append(slide('輸入特徵與時間切分：只用過去預測下一交易日','來源：兩個正式 run config；切分依日期，scaler 僅 fit train','三組特徵為同一任務的輸入設計。所有視窗均為連續 60 個交易日映射到下一日。樣本數和指標計算依正式 run config/evaluation。',e))
# 4
e=[];e+=box(50,95,395,145,'LSTM｜三個門','輸入門：寫入資訊\n遺忘門：保留或移除 cell state\n輸出門：產生隱狀態',BLUE,17);e+=box(515,95,395,145,'GRU｜兩個門','更新門：控制舊狀態保留\n重設門：控制候選狀態使用過去資訊\n沒有獨立 cell state',ORANGE,17)
e += [txt(60,265,840,50,'比較理由：同為序列模型，檢視記憶機制與參數量差異是否帶來誤差差異。',18,NAVY,True),rect(70,350,225,68,'#EAF2F8','#C8D8E5'),txt(76,360,213,45,'輸入序列\n60 日 × 特徵數',16,INK,True,'center'),rect(365,350,225,68,'#EAF2F8','#C8D8E5'),txt(371,360,213,45,'LSTM(64)\n或 GRU(64)',17,INK,True,'center'),rect(660,350,225,68,'#EEF7F5','#C8E5DF'),txt(666,360,213,45,'Dense(1)\n預測次日 Close',17,INK,True,'center'),line(295,384,365,384,BLUE,2),line(590,384,660,384,TEAL,2),txt(76,435,780,35,'兩個模型各自採：輸入 → 循環層 → Dense(1)；A=1、B=5、C=10 個特徵',16,NAVY,True,'center')]
S.append(slide('兩種門控記憶：LSTM 保留 cell state，GRU 合併更新','來源：專案模型實作；論文完整引用列於講者備註','LSTM 使用 cell state 與輸入/遺忘/輸出門；GRU 使用更新/重設門。模型皆為 64 單元循環層接 Dense(1)。參考：Hochreiter & Schmidhuber (1997), Neural Computation 9(8); Cho et al. (2014), EMNLP。',e))
# 5
e=[]; steps=[('資料','snapshot'),('特徵','A / B / C'),('視窗','60 → 1'),('模型','LSTM / GRU'),('訓練','3 seeds'),('評估','485 筆'),('報告','MAE / RMSE')]
for i,(a,b) in enumerate(steps):
 x=42+i*126;e += [rect(x,125,108,80,PALE,'#D8E0E8'),txt(x+4,137,100,25,a,16,NAVY,True,'center'),txt(x+4,164,100,24,b,12,MUTED,False,'center')]
 if i<6:e += [line(x+108,165,x+126,165,GRAY,1)]
e+=box(48,236,410,112,'訓練控制','Adam 0.001｜batch 32｜最多 100 epochs\nEarlyStopping：val_loss、patience 10、restore best',BLUE,15);e+=box(500,236,410,112,'執行設定','shuffle=False｜CPU｜seed 42/43/44\nGRU 為獨立正式實驗分支 gru_experiment',ORANGE,15)
e += [txt(55,355,850,49,'程式流程：data.py load_snapshot() → features.py → dataset.py → model.py → training.py → evaluation.py → metrics.py → report.py\n另有 cli.py、gru_experiment.py；離線驗證已通過：--help／validate-config／load-snapshot',13,INK),rect(55,408,850,95,'#F1F5F8','#DDE5EC'),txt(66,410,828,91,'training.py:45–49｜EarlyStopping\nsettings = config["training"]["early_stopping"]\nreturn [FiniteLoss(), tf.keras.callbacks.EarlyStopping(\n    monitor=settings["monitor"], patience=settings["patience"],\n    restore_best_weights=settings["restore_best_weights"], mode="min",\n)]',11,NAVY,True)]
S.append(slide('同一評估流程：從資料快照到測試報告','來源：專案流程與訓練設定；CLI 為前階段離線驗證','本簡報製作沒有重新訓練或推論。CLI 三項曾於前階段通過：命令列 help、validate-config、load-snapshot。前階段環境記錄 Python 3.11.18。EarlyStopping 監控 val_loss，patience 10 並恢復最佳權重。',e))
# 6
e=[];e+=box(55,95,390,80,'LSTM','A、B、C 各以 seed 42、43、44 訓練',BLUE,15);e+=box(510,95,390,80,'GRU','A、B、C 各以 seed 42、43、44 訓練',ORANGE,15)
for x,t,b,c in [(60,'MAE','平均絕對誤差（元）',BLUE),(350,'RMSE','均方根誤差（元）',TEAL),(640,'DA','三分類方向準確率',ORANGE)]:e+=box(x,215,250,90,t,b,c,16)
e += [rect(55,345,850,78,'#F3F6F9','#DFE6ED'),txt(70,353,820,60,f'基準：前一交易日 Close（Naive）\nMAE {naive_mae:.2f} 元｜RMSE {naive_rmse:.2f} 元｜DA 不適用',18,INK,True,'center'),txt(60,448,840,35,'平均 ± 樣本標準差（ddof=1），描述三個種子的變異；不是信賴區間。',15,MUTED,False,'center')]
S.append(slide('實驗設計：2 個模型 × 3 組特徵 × 3 個種子','來源：正式 LSTM / GRU runs；test n=485','共 18 次既有正式訓練。Naive 是前一交易日收盤價；依專案評估規則，DA 標示 N/A。三種子摘要使用樣本標準差 ddof=1；模型依 validation 選擇，未以 test 選 seed。',e))
# 7
e=[]
def pair(g,m):
 r=(lm if m=='LSTM' else gm)[g]
 return (f"{float(r['mae_mean']):.2f} ± {float(r['mae_std']):.2f}",f"{float(r['rmse_mean']):.2f} ± {float(r['rmse_std']):.2f}")
data=[('Naive',f'{naive_mae:.2f}',f'{naive_rmse:.2f}')]+[(m+' '+g,*pair(g,m)) for g in 'ABC' for m in ('LSTM','GRU')]
heads=['模型組別','MAE 均值 ± SD','RMSE 均值 ± SD']; widths=[135,170,178]; xpos=[40]
for w in widths[:-1]:xpos.append(xpos[-1]+w)
for r,vals in enumerate([heads]+[list(a) for a in data]):
 for c,v in enumerate(vals):
  x=xpos[c]; y=90+r*40; w=widths[c]; fill=NAVY if r==0 else ('#E8F3FA' if r>0 and 'LSTM' in str(vals[0]) else ('#FFF1E6' if r>0 and 'GRU' in str(vals[0]) else '#F1F4F7'))
  e += [rect(x,y,w,40,fill,'#D9E1E8'),txt(x+3,y+3,w-6,34,str(v),16,WHITE if r==0 else INK,r==0 or c==0,'center')]
e += [txt(570,88,335,30,'MAE（元）｜± 樣本 SD',16,NAVY,True,'center')]
barv=[('Naive',naive_mae,0,GRAY)]+[(m+' '+g,float((lm if m=='LSTM' else gm)[g]['mae_mean']),float((lm if m=='LSTM' else gm)[g]['mae_std']),BLUE if m=='LSTM' else ORANGE) for g in 'ABC' for m in ('LSTM','GRU')]
chart_top,chart_bottom=135,375;scale=400
for tick in (0,100,200,300,400):
 yy=chart_bottom-tick/scale*(chart_bottom-chart_top);e += [line(562,yy,907,yy,'#DCE3EA',1),txt(535,yy-10,24,18,str(tick),12,MUTED,False,'right')]
for i,(lab,val,sd,col) in enumerate(barv):
 x=568+i*48; h=val/scale*(chart_bottom-chart_top); err=sd/scale*(chart_bottom-chart_top); top=chart_bottom-h
 short=['Naive','A','A','B','B','C','C'][i]; model=['','LSTM','GRU','LSTM','GRU','LSTM','GRU'][i]
 e += [rect(x,top,28,h,col,col),line(x+14,top-err,x+14,top+err,'#263849',2),line(x+8,top-err,x+20,top-err,'#263849',1),line(x+8,top+err,x+20,top+err,'#263849',1),txt(x-10,378,48,32,(short if not model else model+'\n'+short),12,MUTED,False,'center')]
e += [rect(575,116,14,14,BLUE,BLUE),txt(594,112,64,22,'LSTM',13,INK),rect(676,116,14,14,ORANGE,ORANGE),txt(695,112,64,22,'GRU',13,INK),txt(50,435,850,38,'A 組最低；特徵變多未改善結果；六組模型 MAE 與 RMSE 均高於 Naive。',16,NAVY,True,'center')]
S.append(slide('測試集結果：GRU 三組 MAE 都低於 LSTM，但皆輸給 Naive','來源：正式 evaluation summary CSV；MAE ± sample SD，元','數值由完整精度 CSV 四捨五入至兩位。RMSE ± SD：LSTM A 185.06±60.34、B 241.93±4.22、C 265.09±112.65；GRU A 80.79±5.66、B 105.20±28.36、C 211.97±37.90。DA：LSTM 40.14/40.62/41.51%；GRU 40.21/40.27/40.00%。Naive 是單一基準，不列 seed SD。所有模型 MAE/RMSE 高於 Naive。',e))
# 8
e=[img('assets/price_series.png',48,78,864,380),txt(70,462,820,30,'A 組三種子預測均值；僅供視覺化，非正式 Ensemble 績效。',15,NAVY,True,'center')]
S.append(slide('預測曲線：A 組模型未能跟上測試期快速上行','來源：正式 prediction CSV；2024–2025；A 組三 seed 預測均值','曲線為實際 Close、前日 Close Naive，以及 LSTM A / GRU A 三 seed 預測逐日均值。只供視覺觀察，沒有用平均線重算正式績效，也不稱 ensemble 指標。',e))
# 9
e=[]
for x,t,b,c in [(45,'可能解釋 1','GRU 參數較少，可能降低估計負擔；因果未驗證。',ORANGE),(350,'可能解釋 2','更多特徵或有冗餘/噪音，且參數量亦改變。',TEAL),(655,'可能解釋 3','測試期價位分布高於訓練期，外推較難。',BLUE)]:e+=box(x,95,260,112,t,b,c,15)
e += [rect(48,238,860,68,'#FFF5EC','#F4D9C5'),txt(58,247,840,50,'457 / 485 = 94.23% 測試 Close 高於 train raw Close 上界 683 元；test 範圍 576–1,550 元。',15,INK,True,'center'),txt(55,338,820,30,'未來工作',18,NAVY,True),txt(60,372,825,62,'報酬率 / 差分目標｜滾動驗證｜其他簡單基準\n只用 validation 調參，不以 test 決策',17,INK,False,'center')]
S.append(slide('可能解釋與後續：把分布變化當線索，不作因果結論','來源：正式資料統計與實驗結果；解釋屬待驗證假說','457/485 是比較正式 test actual_close 和 train raw Close 上界所得。Train 範圍 115–683 元，test 範圍 576–1,550 元。兩模型參數量：LSTM A/B/C 16,961/17,985/19,265；GRU A/B/C 12,929/13,697/14,657。這些差異僅為可能解釋線索，不代表因果。未來工作是建議，並未已完成。',e))
# 10
e=[]
for i,(t,b,c) in enumerate([('GRU 誤差較低','A、B、C 三組 MAE 均低於對應 LSTM。',ORANGE),('多特徵未必更好','A 組單一 Close 在兩模型中表現最佳。',TEAL),('Naive 最低誤差','六個模型 MAE 與 RMSE 都高於前日 Close。',BLUE),('結論有範圍','單股、固定期間與設定；不代表可交易報酬。',GRAY)]):e+=box(48,82+i*84,540,72,t,b,c,15)
e+=box(625,95,280,160,'限制','單一股票、固定期間、3 seeds；未評估交易成本與投資報酬。',GRAY,16)
e += [rect(625,285,280,132,'#F4F7FA','#E0E6EC'),txt(640,296,250,105,'主要參考\n正式實驗結果文件\nhttps://github.com/v016604z/\nLSTM_Stock_Trends_Prediction',13,NAVY,False,'center'),txt(50,450,860,30,'重現細節與完整來源：簡報素材來源說明.md',14,MUTED,False,'center')]
S.append(slide('結論：本設定下 GRU 較準，但簡單基準仍勝出','來源：正式結果文件、GitHub 專案；原始論文引用見講者備註','主要來源：「LSTM 與 GRU 股價預測正式實驗結果.md」與 https://github.com/v016604z/LSTM_Stock_Trends_Prediction。原始論文：Hochreiter & Schmidhuber (1997), Long Short-Term Memory, Neural Computation 9(8); Cho et al. (2014), Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, EMNLP。既有權重缺失僅作結果分析，未在本簡報分析權重。',e))
for i,s in enumerate(S):
 s['elements'] += [txt(875,514,45,17,f'{i+1:02d} / 10',10,MUTED,False,'right')]
spec={'canvas_width':960,'canvas_height':540,'font_name':'Microsoft JhengHei','slides':S}
with open(O/'slides_spec.json','w',encoding='utf-8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
with open(O/'content_manifest.json','w',encoding='utf-8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
with open(O/'逐頁講稿.md','w',encoding='utf-8') as f:
 for i,s in enumerate(S,1):f.write(f"## 第 {i} 頁\n\n{s['notes']}\n\n")
with open(O/'簡報素材來源說明.md','w',encoding='utf-8') as f:
 f.write('''# 簡報素材來源說明\n\n## 資料與結果\n\n- 原始市場來源：TWSE STOCK_DAY，2330，非還原 OHLCV。課堂資料表取自 `res/research/2330.TW_2015-2025_twse.csv`；快照來源參數、月快取 SHA-256、資料與交易日曆摘要見兩個正式 run 的 `snapshot.metadata.json` 與 `evaluation/evaluation.json`。\n- 正式結果：LSTM `results/20261120T134010119780Z_82eaad21/evaluation/metrics_summary.csv`；GRU `results/gru_20261120T143901617329Z_1fa00b5e/evaluation/comparison_summary.csv`。表格由上述 CSV 的完整精度均值、樣本 SD 四捨五入至小數 2 位。Naive MAE/RMSE 由 LSTM run 的 `evaluation/predictions_naive.csv` 之 `absolute_error`、`squared_error` 彙整（平均絕對誤差與平方誤差平均後開根號）。\n- 特徵與設定：兩個 run 的 `config.json`；測試集樣本、日期範圍與資料/日曆雜湊見各 run 的 `evaluation/evaluation.json`。DA 為正式 summary CSV 的 directional_accuracy 欄，逐頁講稿中列出均值。± 為三個種子的樣本標準差（ddof=1），非信賴區間。\n- 分布統計：slide evidence 的描述性統計：train raw Close 115–683 元；test 576–1,550 元；457/485（94.2268%）test Close 高於 train raw Close 上界。此結果僅描述分布，不是因果證明。\n\n## 圖像與編輯性\n\n- 第 7 頁 MAE 圖以原生 PowerPoint 矩形、線段、文字刻度與誤差棒建構，所有元素可編輯；均值及樣本 SD 取自正式 summary CSV。\n- 第 8 頁使用 `report_output/figures/price_series.png` 的副本 `assets/price_series.png`。此既有圖包含 actual、Naive、LSTM A 三種子均值與 GRU A 三種子均值；明確標示僅作視覺化，不以均值線重算正式 Ensemble 指標。\n- 時間軸、表格與流程圖均以原生 PowerPoint 文字、形狀、線條或表格建立；只有第 8 頁曲線圖是點陣圖。\n\n## 專案與方法來源\n\n- 專案說明及版本來源：[GitHub](https://github.com/v016604z/LSTM_Stock_Trends_Prediction)。\n- 專案正式結果說明：`docs/LSTM 與 GRU 股價預測正式實驗結果.md`。\n- 原始架構論文：Hochreiter & Schmidhuber (1997), “Long Short-Term Memory,” *Neural Computation*, 9(8); Cho et al. (2014), “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP. 僅作架構脈絡參考。\n\n## 生成與重建\n\n- `build_spec.py` 從正式資料/結果 CSV 產生 `slides_spec.json`、`content_manifest.json`、`逐頁講稿.md`、來源說明與曲線圖副本。\n- `render_native.ps1` 以 PowerPoint COM 建立可編輯簡報、內嵌講者備註並匯出 PDF。\n- 執行：`.venv\\Scripts\\python.exe -B report_output/slides/build_spec.py`，再執行 `powershell -NoProfile -File report_output/slides/render_native.ps1`。\n''')
print('wrote spec, manifest, notes, sources and copied image')
