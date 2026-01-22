from pathlib import Path
from hmmlearn.hmm import GaussianHMM,GMMHMM
import pandas as pd
import numpy as np
import sys
BASE = Path(__file__).resolve().parents[2]
print(f"현재 잡힌 BASE 경로: {BASE}")
sys.path.insert(0, str(BASE))
import DB.db_conn as db
import joblib
# 20일 MA,  20일의 volatility, 거래량, 종가의 위치((종가-최저가)/(최고가-최저가))
conn,cur=db.open_db()
sql="""
select Open,High,Low,Close,Volume,`Change`,Date from kKospi where Year(Date)>2000 and Year(Date)<2015 order by Date asc"""
df = pd.read_sql(sql, conn)
def features(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA20_slope']=df['MA20'].pct_change()
    df['Disparity'] = (df['Close'] - df['MA20']) / df['MA20']
    df['Volatility'] = df['Change'].rolling(window=20).std()
    vol_v1 = df['Volume'].shift(2)
    vol_v2 = df['Volume'].shift(1)
    df['Vol_Change'] = np.where(vol_v1==0,0,(vol_v1 - vol_v2) / vol_v1)
    denominator=df["High"]-df["Low"]
    df['Position']=np.where(denominator==0,0, (df["Close"]-df["Low"])/(denominator))
    return df
df_processed=features(df)
df_processed.dropna(inplace=True)
feature_cols = ['MA20_slope', 'Disparity', 'Volatility', 'Vol_Change', 'Position']
X = df_processed[feature_cols]
model = GaussianHMM(n_components=3, verbose=True, n_iter=100, random_state= 108).fit(X)
joblib.dump(model, 'hmm_model.pkl')
print("모델이 hmm_model.pkl로 저장되었습니다.")