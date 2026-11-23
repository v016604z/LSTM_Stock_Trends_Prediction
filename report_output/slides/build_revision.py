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
# Revision-only local refinements. Keep slides 1, 6 and 8 visually identical to v1.
def set_content(page_no, new_elements):
    old=S[page_no-1]['elements']
    S[page_no-1]['elements']=old[:2]+new_elements+old[-3:]

# Page 2: more explicit snapshot validation and units, with three raw rows.
e=[]
for x,w,t,b,c in [(50,250,'資料量','2,682 個交易日\n2015–2025',BLUE),(340,250,'欄位與單位','OHLC 價格：元\nVolume：股',TEAL),(630,280,'快照檢核','TWSE 原始資料\n資料與日曆雜湊核對',ORANGE)]: e+=box(x,88,w,82,t,b,c,16)
e += [txt(55,178,820,28,'正式 raw snapshot 抽樣（可編輯資料表）',18,NAVY,True)]
raw=rows(R/'res/research/2330.TW_2015-2025_twse.csv'); heads=['Date','Open','High','Low','Close','Volume']
for ri,vals in enumerate([heads]+[[r.get(h,'') for h in heads] for r in raw[:3]]):
 for ci,val in enumerate(vals):
  x=55+ci*142; y=210+ri*42; e += [rect(x,y,142,42,NAVY if ri==0 else ('#F3F6F9' if ri%2 else WHITE),'#D9E1E8'),txt(x+4,y+5,134,31,str(val),16,WHITE if ri==0 else INK,ri==0,'center')]
for x,title,body,col in [(55,'日期/欄位','Date 有效、唯一、排序；OHLCV 五欄完整。',BLUE),(350,'數值','缺值、NaN/Inf、非正價格或 Volume 會拒絕。',TEAL),(645,'交易日曆','與 XTAI 逐日比對；缺少或多出交易日會拒絕。',ORANGE)]:e+=box(x,390,260,96,title,body,col,16)
set_content(2,e)
S[1]['notes']='資料取自 res/research/2330.TW_2015-2025_twse.csv，這是 2015 至 2025 年台積電未還原行情，共 2,682 個交易日。表格抽出正式快照前三筆，價格欄以元表示，成交量 Volume 是股數。專案驗證會確認必要欄位 Open、High、Low、Close、Volume 完整，日期有效、唯一且嚴格排序；數值必須為有限數，缺值、NaN 或 Inf 會被拒絕，非正價格、非正成交量與不合理 OHLC 關係也會拒絕。日期再與獨立 XTAI 交易日曆逐日比對，缺日或多日都會使載入失敗。正式 metadata 保存原始來源參數、月快取雜湊，以及資料與日曆雜湊。來源：src/lstm_comparison/data.py:188–219、290–323；snapshot metadata；res/research/2330.TW_2015-2025_twse.csv。'

# Page 3: separate C heading and explain window-target counts.
e=[]
for x,title,body,col in [(48,'A｜Close（1）','60 日 Close → 下一交易日 Close',BLUE),(355,'B｜OHLCV（5）','Open、High、Low、Close、Volume',TEAL)]:e+=box(x,92,255,112,title,body,col,16)
e += [rect(662,92,255,112,PALE,'#E2E8EF'),rect(662,92,8,112,ORANGE,ORANGE),txt(682,99,220,42,'C｜OHLCV + 指標\n（10 個特徵）',17,ORANGE,True),txt(682,146,220,50,'MA5、MA20、RSI14\nMACD、signal',16,INK)]
e += [txt(50,225,850,30,'依時間切分（未隨機打散）',18,NAVY,True)]
for x,t,b,c in [(50,'Train｜2015–2022','1,865 筆',BLUE),(345,'Validation｜2023','239 筆',TEAL),(640,'Test｜2024–2025','485 筆',ORANGE)]:e += [rect(x,270,260,68,c,c),txt(x+5,278,250,52,t+'\n'+b,17,WHITE,True,'center')]
e += [txt(54,352,850,42,'1,865 / 239 / 485 是三組共用有效視窗的 target 數，不是原始行情列數。',17,NAVY,True),txt(54,401,850,62,'MinMaxScaler 只 fit train；validation 用於 early stopping；test 僅作最後評估。\n指標：MA5/20、Wilder RSI14、MACD(12,26,9)。',16,INK)]
set_content(3,e)
S[2]['notes']='特徵設計分三組：A 只有 Close，B 使用 Open、High、Low、Close、Volume，C 在 OHLCV 上加入 MA5、MA20、Wilder RSI14、MACD 與 signal，總共十個特徵。C 標題分行顯示，避免和指標清單打架。每個樣本用過去 60 個交易日預測下一交易日 Close；各特徵組共用有效視窗與 target 日期，所以 1,865、239、485 是訓練、驗證、測試的視窗目標數，不是行情檔原始列數。時間切分依年份，不打散；MinMaxScaler 只以訓練段擬合，validation 供 early stopping 使用，test 保留到最後評估。這樣才能避免未來資料洩漏到縮放或模型選擇。來源：兩個正式 run 的 config.json、evaluation/evaluation.json；src/lstm_comparison/dataset.py:85–114。'

# Page 4: add a compact editable parameter-count table in the existing gap.
param_l={g:int(json.loads((R/f'results/20261120T134010119780Z_82eaad21/{g}/seed_42/training.json').read_text(encoding='utf-8'))['parameter_count']) for g in 'ABC'}
param_g={g:int(gm[g]['parameter_count']) for g in 'ABC'}
old=S[3]['elements']; content=old[2:-3]
e=content[:8]
for r,vals in enumerate([('組別','LSTM 參數量','GRU 參數量'),('A｜Close',f'{param_l["A"]:,}',f'{param_g["A"]:,}'),('B｜OHLCV',f'{param_l["B"]:,}',f'{param_g["B"]:,}'),('C｜+ 指標',f'{param_l["C"]:,}',f'{param_g["C"]:,}')]):
 for c,v in enumerate(vals):
  x=[60,320,615][c]; w=[260,295,295][c]; y=246+r*21; fill=NAVY if r==0 else ('#E8F3FA' if c==1 else ('#FFF1E6' if c==2 else PALE))
  e += [rect(x,y,w,21,fill,'#D9E1E8'),txt(x+3,y+1,w-6,19,v,16,WHITE if r==0 else INK,r==0 or c==0,'center')]
e+=content[9:]
set_content(4,e)
S[3]['notes']='LSTM 以 cell state 搭配輸入、遺忘、輸出三個門，GRU 則用更新與重設兩個門，沒有獨立 cell state。這裡兩者都是 64 單元循環層接一個 Dense(1)，輸出次日 Close；A、B、C 是輸入欄位設計，不是三種模型。小表呈現參數量隨輸入維度改變：LSTM A/B/C 為 16,961、17,985、19,265；GRU 為 12,929、13,697、14,657。這使比較包含架構記憶方式，也包含實際模型容量差異，不能把結果只歸因於門控形式。參數量來自正式 training.json 與 GRU comparison_summary.csv。概念來源：Hochreiter & Schmidhuber (1997), “Long Short-Term Memory,” Neural Computation 9(8)；Cho et al. (2014), “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP。'

# Page 5: show a larger, exact four-line excerpt (training.py 46–49).
e=S[4]['elements'][2:-3]
for el in e:
 if el.get('type')=='rect' and el.get('x')==55 and el.get('y')==408: el['y']=408;el['h']=95
 if el.get('type')=='text' and el.get('x')==66 and el.get('y')==410:
  el['text']='training.py:46–49｜EarlyStopping 節錄\nreturn [FiniteLoss(), tf.keras.callbacks.EarlyStopping(\n    monitor=settings["monitor"], patience=settings["patience"],\n    restore_best_weights=settings["restore_best_weights"], mode="min",\n)]';el['y']=410;el['h']=91;el['font_size']=14
set_content(5,e)
S[4]['notes']='整個流程由 data.py 的 load_snapshot() 載入並驗證快照，接著 features.py 建特徵、dataset.py 產生視窗、model.py 建立網路、training.py 訓練、evaluation.py 與 metrics.py 計分，最後 report.py 彙整報告；cli.py 提供操作入口，GRU 實驗由 gru_experiment.py 分支執行。投影片程式碼是 training.py 第 46–49 行的逐字節錄，省略前一行的 settings 設定與尾端以外的上下文，非自行拼造。EarlyStopping 監控 val_loss，patience=10，恢復最佳權重；Adam 0.001、batch 32、最多 100 epochs、shuffle=False、CPU、seed 42/43/44。CLI 的 --help、validate-config、load-snapshot 是前階段離線驗證，並非本次製作簡報時重跑，也不代表重新訓練或推論。來源：src/lstm_comparison/training.py:45–49；兩個正式 run config/environment；前階段CLI驗證紀錄。'

# Page 7: six model bars plus a distinct dashed baseline, keeping formal table unchanged.
e=[]
data=[('Naive',f'{naive_mae:.2f}',f'{naive_rmse:.2f}')]+[(m+' '+g,*pair(g,m)) for g in 'ABC' for m in ('LSTM','GRU')]
heads=['模型組別','MAE 均值 ± SD','RMSE 均值 ± SD']; widths=[135,170,178]; xpos=[40]
for w in widths[:-1]:xpos.append(xpos[-1]+w)
for r,vals in enumerate([heads]+[list(a) for a in data]):
 for c,v in enumerate(vals):
  x=xpos[c]; y=90+r*40; w=widths[c]; fill=NAVY if r==0 else ('#E8F3FA' if r>0 and 'LSTM' in str(vals[0]) else ('#FFF1E6' if r>0 and 'GRU' in str(vals[0]) else '#F1F4F7'))
  e += [rect(x,y,w,40,fill,'#D9E1E8'),txt(x+3,y+3,w-6,34,str(v),16,WHITE if r==0 else INK,r==0 or c==0,'center')]
e += [txt(565,82,345,28,'測試 MAE（元）',16,NAVY,True,'center'),rect(580,116,14,14,BLUE,BLUE),txt(599,112,60,22,'LSTM',13,INK),rect(663,116,14,14,ORANGE,ORANGE),txt(682,112,55,22,'GRU',13,INK),line(740,123,770,123,GRAY,1),txt(775,112,130,22,f'Naive {naive_mae:.2f}',13,INK)]
chart_top,chart_bottom,scale=146,370,400
for tick in (0,100,200,300,400):
 yy=chart_bottom-tick/scale*(chart_bottom-chart_top);e += [line(562,yy,907,yy,'#DCE3EA',1),txt(535,yy-10,24,18,str(tick),12,MUTED,False,'right')]
barv=[(m+' '+g,float((lm if m=='LSTM' else gm)[g]['mae_mean']),float((lm if m=='LSTM' else gm)[g]['mae_std']),BLUE if m=='LSTM' else ORANGE) for g in 'ABC' for m in ('LSTM','GRU')]
for i,(lab,val,sd,col) in enumerate(barv):
 x=584+i*51; h=val/scale*(chart_bottom-chart_top); err=sd/scale*(chart_bottom-chart_top); top=chart_bottom-h
 e += [rect(x,top,30,h,col,col),line(x+15,top-err,x+15,top+err,'#263849',2),line(x+8,top-err,x+22,top-err,'#263849',1),line(x+8,top+err,x+22,top+err,'#263849',1),txt(x-9,374,48,25,('L' if i%2==0 else 'G')+'-'+('ABC'[i//2]),12,MUTED,False,'center')]
baseline=chart_bottom-naive_mae/scale*(chart_bottom-chart_top)
for x in range(563,908,11):e.append(line(x,baseline,min(x+5,908),baseline,GRAY,2))
e += [txt(50,435,850,38,'A 組最低；特徵變多未改善結果；六組模型 MAE 與 RMSE 均高於 Naive。',16,NAVY,True,'center')]
set_content(7,e)
S[6]['notes']='左表是測試集正式結果，數值取自兩份 summary CSV 原始精度後四捨五入到小數第二位。LSTM 的 A/B/C MAE 為 161.35±54.89、201.04±3.63、223.41±106.61 元；GRU 為 66.51±4.86、87.06±26.66、181.29±35.59 元。RMSE 也呈相同排序方向，六組模型都高於前日 Close 的 Naive 基準 15.37 元 MAE、20.97 元 RMSE。三seed的±是樣本標準差 ddof=1，表示種子間變動，不能當成信賴區間；Naive 是單一基準，不計seed SD。右圖只呈現 MAE，虛線標示Naive值，柱上的線是±SD。重點是本設定內GRU對應三組均低於LSTM，且A組最小；這不是因果或普遍模型優劣結論。來源：兩個run的evaluation/metrics_summary.csv、comparison_summary.csv與LSTM evaluation/predictions_naive.csv。'

# Page 9: align each testable hypothesis with a future validation step.
e=[]
cards=[(45,'參數較少？','假說：估計負擔可能較低；因果未驗證。\n未來：只用 validation 比容量與正則化。',ORANGE),(350,'特徵冗餘？','假說：指標可能冗餘或含噪音。\n未來：特徵消融/選擇。',TEAL),(655,'分布改變？','假說：價位上移使外推困難。\n未來：報酬率/差分目標與滾動驗證。',BLUE)]
for x,t,b,c in cards:e+=box(x,95,260,125,t,b,c,15)
e += [rect(48,238,860,68,'#FFF5EC','#F4D9C5'),txt(58,247,840,50,'457 / 485 = 94.23% 測試 Close 高於 train raw Close 上界 683 元；test 範圍 576–1,550 元。',15,INK,True,'center'),txt(55,338,820,30,'其他可檢驗方向',18,NAVY,True),txt(60,372,825,62,'三項延伸都只用 validation 選擇；不得以 test 調參或選模型。',17,INK,False,'center')]
set_content(9,e)
S[8]['notes']='三個解釋都只是待驗證假說。第一，GRU在相同單元數下參數較少，可能降低估計負擔；下一步可只用validation比較不同容量與正規化設定。第二，C組的技術指標可能冗餘或帶入噪音；可做特徵消融與特徵選擇，並以validation決定。第三，測試期股價分布高於訓練期上界，固定價位尺度可能不易外推；457/485筆（94.23%）test Close高於train raw Close上界683元，train範圍115–683元，test範圍576–1,550元。後續可改用報酬率或差分目標並採滾動驗證。這些只是因果候選，不表示資料已證明原因；也不是已完成研究。所有超參數、特徵與模型選擇均應限於validation，test只作一次最終評估。來源：slide_evidence.json、正式run資料與prediction CSV。'

# Page 10: a readable, three-line GitHub address; full citations stay in notes.
e=S[9]['elements'][2:-3]
for el in e:
 if el.get('type')=='text' and el.get('x')==640 and el.get('y')==296:
  el['text']='主要參考\n正式結果文件與 CSV\nGitHub 專案';el['font_size']=17;el['h']=118;el['hyperlink']='https://github.com/v016604z/LSTM_Stock_Trends_Prediction'
set_content(10,e)
S[9]['notes']='最後總結四點：這六個既有組合中GRU的MAE都低於相同特徵的LSTM；單一Close的A組在兩模型內均是最低誤差；所有神經網路結果仍高於前日Close Naive；結論只適用於單一股票、此期間、此資料處理和這組訓練設定。這不是交易策略，也未計算成本、持有期報酬或風險調整績效。主要證據是docs/LSTM 與 GRU 股價預測正式實驗結果.md與兩份evaluation summary CSV；可重現專案位於https://github.com/v016604z/LSTM_Stock_Trends_Prediction。架構參考僅列於本頁講者備註：Hochreiter & Schmidhuber (1997), Long Short-Term Memory, Neural Computation 9(8)；Cho et al. (2014), Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation, EMNLP。研究限制包括單股、固定期間、三種子，且沒有交易報酬評估。'

revision_notes=[
'本次報告以台積電為例，探討循環神經網路能否從過去價格預測下一交易日收盤價。研究動機在於股票序列同時包含短期波動與長期趨勢，LSTM與GRU都使用門控狀態處理序列，但計算結構不同；另外，輸入欄位是否更豐富也可能改變預測結果。我們因此將兩種模型放在三組輸入條件下比較，並加入前日收盤價作為簡單基準。核心問題不是建立買賣訊號，而是檢查固定資料期間與訓練設定中的誤差差異。封面保留課程、報告者、教師與日期欄位供使用者自行填寫。後面先說明資料和切分，再介紹模型與實驗，最後回到結果邊界。來源：正式實驗結果文件、專案GitHub與各頁列示的CSV。',
'本研究使用TWSE的2330日資料，時間由2015年初至2025年底，共2,682個交易日。上方列出的OHLC價格以元記錄，Volume以股數記錄；下方表格取自正式研究快照前三列，不是示意數字。資料入口不只檢查欄位存在，還要求日期有效、唯一、遞增，OHLCV數值可轉為有限數字；缺值、NaN、Inf與不合理價格或成交量會拒絕。專案將日期集合與獨立XTAI日曆比對，因此漏掉或多出交易日都不會被靜默補齊。快照metadata另存來源網址、請求參數與雜湊，讓日後可以辨認檔案是否改變。這些檢查支持資料可追溯性，但不代表資料沒有市場結構偏誤。來源：res/research/2330.TW_2015-2025_twse.csv、snapshot metadata、src/lstm_comparison/data.py。',
'A組只看Close，B組使用OHLCV，C組再加入兩條移動平均線、RSI及MACD相關指標。三組都用60個連續交易日作輸入，輸出下一日Close。樣本按時間分成2015–2022訓練、2023驗證、2024–2025測試；圖中1,865、239、485是跨三組共用的有效視窗目標筆數，並非原始行情列數。這個差異很重要，因為技術指標需要暖機資料，且60日視窗需要完整連續序列。MinMaxScaler只在訓練資料上fit，validation提供early stopping訊號，而test保留作最後評估。使用時間切分而非隨機抽樣，是為了讓評估較接近先看歷史再預測未來的情境。來源：兩份正式run config、evaluation metadata與dataset.py。',
'LSTM與GRU都以門控控制狀態更新。LSTM有獨立cell state，輸入、遺忘、輸出門分別控制寫入、保留與輸出；GRU以更新門和重設門合併部分狀態路徑。兩模型均為64單元循環層接Dense(1)，預測單一數值。下方參數量小表提醒我們，模型容量不只由單元數決定，也受輸入維度影響：三組特徵下LSTM參數數分別為16,961、17,985、19,265，GRU為12,929、13,697、14,657。因而結果比較同時反映結構與參數量的變化，不能將差異簡化成「某種門一定較好」。架構概念參考Hochreiter與Schmidhuber的LSTM論文及Cho等人的GRU相關工作；參數數據則取自既有訓練紀錄。',
'這頁把程式工作流濃縮成八個主要模組：data載入快照、features建立指標、dataset形成時間窗、model建立網路、training執行學習、evaluation和metrics計算指標，report輸出結果；cli提供命令入口，gru_experiment負責GRU正式實驗分支。畫面中的EarlyStopping程式是training.py第46至49行原文節錄，省略前置設定變數一行，並非另寫的示意碼。訓練監控validation loss，連續十個epoch沒有改善就停止並恢復最佳權重。這些命令列驗證在前一階段離線完成；簡報製作沒有再次訓練或下載資料。Python環境記錄為3.11.18。讀者若要重現，應以來源說明中的正式設定與資料快照為準。來源：src/lstm_comparison各模組、run config與環境紀錄。',
'正式設計包含LSTM、GRU兩模型，每個模型跑A、B、C三組輸入，各以42、43、44三個seed訓練，合計18次既有實驗。評估使用MAE、RMSE及方向準確率DA；DA依專案規則將漲、平、跌視為方向類別。Naive預測值是前一交易日Close，正式結果中DA標為N/A，因此不能拿它與模型方向準確率直接比較。模型以validation表現控制early stopping，test不拿來選seed或調整超參數。每組均值旁的±是三seed樣本標準差，使用ddof=1，呈現不同seed的結果變動；它不是母體估計的信賴區間，也不是顯著性檢定。測試集共485個目標日。來源：兩run的evaluation JSON、metrics CSV與設定檔。',
'第七頁把兩種摘要放在同一頁。表格完整列出Naive及六個模型組合的MAE與RMSE，模型數值先由正式CSV完整精度讀入，再四捨五入至小數第二位；Naive則由正式Naive prediction CSV的absolute_error與squared_error彙整。圖表只畫MAE均值，模型柱上的誤差線是±樣本SD，灰色虛線標出Naive MAE 15.37元。可見GRU在A、B、C三組都比相對應LSTM低，A組在兩個模型中也都是最低；然而六個模型的MAE與RMSE仍高於Naive。這個結果說明本次較複雜模型沒有勝過簡單前值基準。SD差異也提醒我們，三個seed的穩定度並不一致；本研究沒有做統計顯著性推論。來源：LSTM metrics_summary.csv、GRU comparison_summary.csv、predictions_naive.csv。',
'第八頁沿用專案已產生的價格序列圖，顯示2024至2025測試期間實際Close、前日Close，以及A組LSTM與GRU各三個seed預測的逐日均值。模型線大方向跟著價格走，但價格上升較快時落在實際線下方，視覺上反映出外推落後。這張圖只用來觀察曲線，不用三seed均值線重新計算MAE/RMSE，也不將其稱為正式ensemble績效；正式表格仍以各seed正式指標彙總。保留這頁可讓聽眾理解數值誤差的時間形狀，但不能只挑視覺效果下結論。實際Close快速抬升是資料分布的一部分，下一頁會用訓練與測試價位範圍描述這個現象。來源：report_output/figures/price_series.png，該檔副本置於slides/assets。',
'我們提出三個可被後續研究檢驗的假說，而不是已證實的成因。第一，GRU參數量較少，可能降低估計負擔；可在validation上比較容量與正規化設定。第二，增加技術指標可能帶入冗餘或噪音；可以做特徵消融與選擇。第三，價格測試期高於訓練期上界，價位尺度外推可能困難；可改用報酬率或差分目標並進行滾動驗證。資料證據顯示457/485個測試Close高於訓練Close最大值683元，比例94.23%，但這只說明分布有移動，不能證明它造成模型誤差。所有改良與超參數比較都應僅由validation決定，test留作最後一次評估。來源：正式snapshot、test prediction CSV與slide evidence統計。',
'結論限於單一股票、固定年份範圍、特定特徵工程與三個seed：本次GRU誤差低於對應LSTM，單一Close輸入A組最低，且兩種神經網路都未勝過前日Close基準。這些結果不能直接推導交易獲利，因為沒有估計交易成本、持有期報酬或風險，也沒有跨股票、跨市場期間驗證。簡報的主要研究來源是專案正式結果文件與其LSTM/GRU evaluation CSV；GitHub連結提供程式與文件入口。LSTM與GRU原始論文只用於說明架構背景，不用來替本次實驗結果背書。實際數字仍以正式run產物為準。報告結束後可討論：若改採報酬率、擴大股票樣本或做滾動驗證，結論是否改變？來源：docs/LSTM 與 GRU 股價預測正式實驗結果.md、兩份summary CSV、GitHub專案。'
]
note_extensions=[
'在解讀後面結果前，先記得本研究把預測誤差當成比較工具，而非直接衡量投資價值；因此除了觀察兩種網路，也要先問簡單基準能否提供更有用的參考。',
'日期唯一與交易日完整是兩件不同的事：唯一檢查避免重複觀測，日曆比對則避免漏掉應有交易日或納入非交易日。遇到異常時系統採拒絕載入，不會自行填補，避免修補方式改變後續指標。',
'技術指標會在開頭產生暖機區段，時間視窗亦需要完整歷史，因此三組模型共用目標日期集合，才可公平比較。測試年完全晚於訓練與驗證期，這能避免隨機混合未來樣本，但也使模型面對新的價格區間。',
'小表中的參數量是正式訓練紀錄讀出的數值，不是估算值。由於C組輸入維度較多，兩個模型的參數量也上升；未來若要分離架構效果與容量效果，應在validation設計容量匹配或正規化比較。',
'這個流程圖描述已完成的研究管線，而不是本次簡報的執行清單。程式碼摘錄保留來源行號，呈現callbacks如何接到Keras訓練程序；其餘超參數與執行條件以正式config及run metadata為準。',
'DA衡量預測方向是否正確，價格誤差則衡量預測數值距離，兩者回答不同問題。Naive是把前一日價格延續到下一日，專案沒有替它定義漲跌方向分數，因此這格應保留N/A，而不是自行比較。',
'MAE可直觀解讀為平均偏差金額，RMSE會對較大錯誤給予更重權重。表格同時列出兩者，是避免只挑一個有利指標；均值描述中心，SD描述三個隨機種子的分散，不足以單獨作顯著性判斷。',
'圖中Naive線與實際價格較貼近，模型線在上漲期間有落後現象；這是該期間的視覺描述，不能推論所有行情都會如此。數值評估仍應回到逐seed測試結果表。',
'驗證未來改良時，應事先固定測試流程，避免反覆依test表現修改模型後又把同一段資料當作未見資料。這樣才能保留最後測試結果的解釋力。',
'若聽眾想延伸，可討論不同目標尺度和跨期間驗證是否改變排序；回答仍須依新的正式實驗資料，而非從本次一段測試曲線外推。'
]
revision_notes=[a+' '+b for a,b in zip(revision_notes,note_extensions)]
revision_notes[3]+='這也提示模型參數需和資料量一併考量。'
revision_notes[5]+='方向準確率不等同報酬高低，也不包含交易成本。'
revision_notes[0]=revision_notes[0].replace('封面保留課程、報告者、教師與日期欄位供使用者自行填寫。','')
revision_notes[3]=revision_notes[3].replace('正規化比較','正則化比較')
transitions={1:'接著先看資料快照的檢核結果，再進入特徵設計與視窗切分。',2:'資料切分清楚後，下一步比較兩種循環網路各自如何保留序列資訊。',3:'理解兩種模型的狀態結構後，再看程式如何把它們放入同一實驗流程。',4:'流程相同之後，接著說明模型、特徵與隨機種子的實驗組合。',5:'設計交代清楚後，下一頁直接比較各組在測試集的正式指標。',6:'數值整理出差異後，接著用測試期曲線觀察誤差隨時間的樣貌。',8:'確認這些假說都尚待驗證後，最後回到本次研究可以支持的結論。'}
for i,t in transitions.items():revision_notes[i]+=' '+t
revision_notes[0]='講者提示（不口述）：課程、報告者、指導教師與日期保留為可編輯欄位。\n\n'+revision_notes[0]
revision_sources=[]
for n in revision_notes:
 if '來源：' in n:
  speech,src=n.rsplit('來源：',1); revision_sources.append(speech.strip()+'\n\n備註｜來源：'+src.strip())
 else: revision_sources.append(n)
revision_notes=revision_sources
for i,n in enumerate(revision_notes): S[i]['notes']=n

for i,s in enumerate(S):
 s['elements'] += [txt(875,514,45,17,f'{i+1:02d} / 10',10,MUTED,False,'right')]
spec={'canvas_width':960,'canvas_height':540,'font_name':'Microsoft JhengHei','slides':S}
with open(O/'slides_spec_revision.json','w',encoding='utf-8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
with open(O/'content_manifest_revision.json','w',encoding='utf-8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
with open(O/'逐頁講稿.md','w',encoding='utf-8') as f:
 for i,s in enumerate(S,1):f.write(f"## 第 {i} 頁\n\n{s['notes']}\n\n")
with open(O/'簡報素材來源說明_修訂版.md','w',encoding='utf-8') as f:
 f.write('''# 簡報素材來源說明\n\n## 資料與結果\n\n- 原始市場來源：TWSE STOCK_DAY，2330，非還原 OHLCV。課堂資料表取自 `res/research/2330.TW_2015-2025_twse.csv`；快照來源參數、月快取 SHA-256、資料與交易日曆摘要見兩個正式 run 的 `snapshot.metadata.json` 與 `evaluation/evaluation.json`。\n- 正式結果：LSTM `results/20261120T134010119780Z_82eaad21/evaluation/metrics_summary.csv`；GRU `results/gru_20261120T143901617329Z_1fa00b5e/evaluation/comparison_summary.csv`。表格由上述 CSV 的完整精度均值、樣本 SD 四捨五入至小數 2 位。Naive MAE/RMSE 由 LSTM run 的 `evaluation/predictions_naive.csv` 之 `absolute_error`、`squared_error` 彙整（平均絕對誤差與平方誤差平均後開根號）。\n- 特徵與設定：兩個 run 的 `config.json`；測試集樣本、日期範圍與資料/日曆雜湊見各 run 的 `evaluation/evaluation.json`。DA 為正式 summary CSV 的 directional_accuracy 欄，逐頁講稿中列出均值。± 為三個種子的樣本標準差（ddof=1），非信賴區間。\n- 分布統計：slide evidence 的描述性統計：train raw Close 115–683 元；test 576–1,550 元；457/485（94.2268%）test Close 高於 train raw Close 上界。此結果僅描述分布，不是因果證明。\n\n## 圖像與編輯性\n\n- 第 7 頁 MAE 圖以原生 PowerPoint 矩形、線段、文字刻度與誤差棒建構，所有元素可編輯；均值及樣本 SD 取自正式 summary CSV。\n- 第 8 頁使用 `report_output/figures/price_series.png` 的副本 `assets/price_series.png`。此既有圖包含 actual、Naive、LSTM A 三種子均值與 GRU A 三種子均值；明確標示僅作視覺化，不以均值線重算正式 Ensemble 指標。\n- 時間軸、表格與流程圖均以原生 PowerPoint 文字、形狀、線條或表格建立；只有第 8 頁曲線圖是點陣圖。\n\n## 專案與方法來源\n\n- 專案說明及版本來源：[GitHub](https://github.com/v016604z/LSTM_Stock_Trends_Prediction)。\n- 專案正式結果說明：`docs/LSTM 與 GRU 股價預測正式實驗結果.md`。\n- 原始架構論文：Hochreiter & Schmidhuber (1997), “Long Short-Term Memory,” *Neural Computation*, 9(8); Cho et al. (2014), “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP. 僅作架構脈絡參考。\n\n## 生成與重建\n\n- `build_revision.py` 從正式資料/結果 CSV 產生 `slides_spec_revision.json`、`content_manifest_revision.json`、`逐頁講稿.md`、來源說明與曲線圖副本。\n- `render_native.ps1` 以 PowerPoint COM 建立可編輯簡報、內嵌講者備註並匯出 PDF。\n- 執行：`.venv\\Scripts\\python.exe -B report_output/slides/build_revision.py`，再執行 `powershell -NoProfile -File report_output/slides/render_native.ps1 -Spec report_output/slides/slides_spec_revision.json -Pptx report_output/slides/LSTM_GRU_課堂報告_修訂版.pptx -Pdf report_output/slides/LSTM_GRU_課堂報告_修訂版.pdf -Preview report_output/slides/preview_revision`。\n''')
with open(O/'簡報素材來源說明_修訂版.md','a',encoding='utf-8') as f:
 f.write('''\n## 本修訂版補充\n\n- 第 2 頁驗證敘述對應 `src/lstm_comparison/data.py`：日期唯一排序、必要 OHLCV 欄位、有限數值檢查、異常值拒絕與 XTAI 日曆缺漏/多日比對。快照目前2,682筆日期唯一、欄位齊全、無缺值/非有限值，且日曆與資料hash一致；價格單位元、Volume單位股。\n- 第 3 頁 1,865/239/485 為 A/B/C 共用有效時間窗 target 數，不是原始行情列數。\n- 第 4 頁參數量取自正式 LSTM `training.json` 與 GRU `comparison_summary.csv`；不是重新訓練所得。\n- 第 5 頁程式碼是 `training.py:46–49` 精確節錄，完整行文保留在內嵌講者備註與講稿。\n- 第 7 頁 Naive 水平虛線為正式 baseline MAE 15.371134 元；正式表格數字保持不變。\n- 第 9 頁三項改善路徑均為未來工作，調參與特徵選擇限定 validation。\n- 修訂版建置器 `build_revision.py` 輸出 `slides_spec_revision.json` 與 `content_manifest_revision.json`；原第一版簡報/PDF保留作對照。\n''')
print('wrote revision spec, manifest, speaker notes and source notes')
