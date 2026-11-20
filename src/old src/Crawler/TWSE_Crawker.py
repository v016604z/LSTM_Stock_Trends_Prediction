import requests
import json
import pandas as pd

def twse_data(r_date: str):
    try:
        # 發送 GET 請求到 TWSE API，帶入指定日期
        response = requests.get(f'https://www.twse.com.tw/rwd/zh/fund/T86?date={r_date}&selectType=ALLBUT0999&response=json&_=1721655494966')
        response.raise_for_status()  # 檢查請求是否成功
    except requests.exceptions.RequestException as e:
        print(f"HTTP 請求錯誤: {e}")
        return None

    try:
        # 將回應內容解析為 JSON 格式
        data_json = response.json()
    except json.JSONDecodeError as e:
        print(f"JSON 解析錯誤: {e}")
        print("回應內容:", response.text)
        return None

    try:
        # 使用 JSON 內容建立 DataFrame
        data_store = pd.DataFrame(data_json['data'], columns=data_json['fields'])
    except KeyError as e:
        print(f"資料處理錯誤: 找不到鍵 {e}")
        return None

    return data_store

# 取得並打印指定日期的資料
data = twse_data('20210225')
if data is not None:
    print(data)
