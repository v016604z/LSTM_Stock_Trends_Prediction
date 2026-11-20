import yfinance as yf
import ta

ticker_symbol = '2330.TW'

stock = yf.Ticker(ticker_symbol)

# 獲取指定期間的歷史數據
df = stock.history(start="2017-01-01", end="2021-02-02")

# 獲取主要持股人、財務數據、資產負債表和現金流量數據
major_holders = stock.major_holders
fin_data = stock.financials
balance_data = stock.balance_sheet
cf_data = stock.cashflow

# 將所有技術分析特徵添加到數據框中
data = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)

# 打印包含技術分析特徵的數據框
print(data)
