import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

class HMM_Model:
    def __init__(self, model_path='hmm_model.pkl'):
        self.model = joblib.load(model_path)
    def preprocess(self, df):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Change'])          
        df = df.copy()
        if len(df) < 20:
            return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA20_slope'] = df['MA20'].pct_change()
        df['Disparity'] = (df['Close'] - df['MA20']) / df['MA20']
        df['Volatility'] = df['Change'].rolling(window=20).std()      
        vol_v1 = df['Volume'].shift(1)
        vol_v2 = df['Volume']
        # 0으로 나누기 방지
        df['Vol_Change'] = np.where(vol_v1 == 0, 0, (vol_v2 - vol_v1) / vol_v1)  
        denominator = df["High"] - df["Low"]
        df['Position'] = np.where(denominator == 0, 0, (df["Close"] - df["Low"]) / denominator)
        # 마지막 행(오늘)의 Feature만 추출
        feature_cols = ['MA20_slope', 'Disparity', 'Volatility', 'Vol_Change', 'Position']
        last_row = df.iloc[[-1]][feature_cols].fillna(0) 
        return last_row.values

    def predict(self, kospi_data):
        X = self.preprocess(kospi_data)     
        if X is None:
            return 0 # 데이터 부족 시 기본값            
        # HMM 예측
        try:
            regime = self.model.predict(X)
            return int(regime[0])
        except:
            return 0
    def __call__(self, kospi_data):
        return self.predict(kospi_data)
class PCA_Model:
    def __init__(self, pca_path='pca_model.pkl', scaler_path='scaler_model.pkl'):
        self.pca = joblib.load(pca_path)
        self.scaler = joblib.load(scaler_path)        
        self.feature_names = [
            "at", "cl", "cr", "dr", "em", "et", "lr", "nc", "npm", 
            "of", "opm", "re", "wc", "roa", "tb"
        ]
    def calculate_ratios(self, row_data):
        (Netincome, Totalequity, Totalassets, Operatingincome, Revenue, 
         Totalliabilities, Currentassets, Currentliabilities, Pretaxincome, 
         Retainedearnings, Noncurrentliabilities, Noncurrentassets) = row_data
        def safe_div(a, b):
            return a / b if b != 0 else 0.0

        ratios = []
        ratios.append(safe_div(Revenue, Totalassets))          # at
        ratios.append(safe_div(Currentliabilities, Totalliabilities)) # cl
        ratios.append(safe_div(Currentassets, Currentliabilities))    # cr
        ratios.append(safe_div(Totalliabilities, Totalequity))        # dr
        ratios.append(safe_div(Totalassets, Totalequity))             # em
        ratios.append(safe_div(Revenue, Totalequity))                 # et
        ratios.append(safe_div(Totalliabilities, Totalassets))        # lr
        ratios.append(safe_div(Noncurrentassets, Totalequity + Noncurrentliabilities)) # nc
        ratios.append(safe_div(Netincome, Revenue))                   # npm
        ratios.append(safe_div(Operatingincome, Pretaxincome))        # of
        ratios.append(safe_div(Operatingincome, Revenue))             # opm
        ratios.append(safe_div(Retainedearnings, Totalassets))        # re
        ratios.append(safe_div(Currentassets - Currentliabilities, Totalassets)) # wc
        ratios.append(safe_div(Netincome, Totalassets))               # roa
        ratios.append(safe_div(Netincome, Pretaxincome))              # tb
        
        return np.array(ratios)

    def __call__(self, kor_bf_list):
        processed_data = []
        for raw_data in kor_bf_list:
            if np.all(raw_data == 0):
                processed_data.append(np.zeros(15))
                continue
            ratios = self.calculate_ratios(raw_data)
            processed_data.append(ratios)
            
        if not processed_data:
            return np.zeros(self.pca.n_components_)

        X = np.array(processed_data) # (N_stocks, 15)

        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled) 

        pcs = self.pca.transform(X_scaled) # (N_stocks, 5)

        avg_pcs = np.mean(pcs, axis=0)
        
        return avg_pcs[:4]
class Cov_Model:
    def __init__(self):
        pass
        
    def calculate_changes(self, price_series, vol_series):
        price_diff = price_series[1:] - price_series[:-1]# 오늘-어제
        price_returns = price_diff / (price_series[:-1] + 1e-8)#변화율/어제
        
        vol_diff = vol_series[1:] - vol_series[:-1]
        vol_returns = vol_diff / (vol_series[:-1] + 1e-8)

        combined_features = np.concatenate([price_returns, vol_returns])
        return combined_features

    def __call__(self, us_tick, kor_tick):
        all_assets_vectors = []
        
        for tick in us_tick:
            if len(tick) < 2:
                all_assets_vectors.append(np.zeros(38))
                continue
            prices = tick[:, 0]
            volumes = tick[:, 1]
            vec = self.calculate_changes(prices, volumes)
            all_assets_vectors.append(vec)

        for tick in kor_tick:
            if len(tick) < 2:
                all_assets_vectors.append(np.zeros(38))
                continue
            prices = tick[:, 0]
            volumes = tick[:, 1]
            vec = self.calculate_changes(prices, volumes)
            all_assets_vectors.append(vec)
            
        if not all_assets_vectors:
            return np.zeros((1, 1))

        data_matrix = np.array(all_assets_vectors)

        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(data_matrix)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        return corr_matrix