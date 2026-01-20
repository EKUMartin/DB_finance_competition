from pathlib import Path
from hmmlearn.hmm import GaussianHMM,GMMHMM
import pandas as pd
import numpy as np
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import data.process_data as pp
import DB.db_conn as db



# 20일 MA,  20일의 volatility, 거래량, 종가의 위치((종가-최저가)/(최고가-최저가))
conn,cur=db.open_db()
def _20_ma():
    
    return ma20

model = GaussianHMM(n_components=4, verbose=True, n_iter=100, random_state= 108).fit(X)