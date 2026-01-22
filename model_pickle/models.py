import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

class HMM_Model:
    def __init__(self, model_path='hmm_model.pkl'):
        # 1. 모델 로드
        self.model = joblib.load(model_path)
        
    def preprocess(self, df):
        """KOSPI 데이터프레임을 받아서 HMM 입력용 Feature로 변환"""
        # Env에서 넘어오는 df는 컬럼명이 없을 수 있으므로 컬럼명 지정 필요
        # Env에서 .values로 넘기지 말고 DataFrame으로 넘기거나, 여기서 생성
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Change'])
            
        df = df.copy()
        # 데이터가 너무 적으면 Rolling 계산 불가 (최소 20개 필요)
        if len(df) < 20:
            return None

        # Feature Engineering (사용자님 로직 그대로)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA20_slope'] = df['MA20'].pct_change()
        df['Disparity'] = (df['Close'] - df['MA20']) / df['MA20']
        df['Volatility'] = df['Change'].rolling(window=20).std()
        
        vol_v1 = df['Volume'].shift(2)
        vol_v2 = df['Volume'].shift(1)
        # 0으로 나누기 방지
        df['Vol_Change'] = np.where(vol_v1 == 0, 0, (vol_v1 - vol_v2) / vol_v1)
        
        denominator = df["High"] - df["Low"]
        df['Position'] = np.where(denominator == 0, 0, (df["Close"] - df["Low"]) / denominator)
        
        # 마지막 행(오늘)의 Feature만 추출
        feature_cols = ['MA20_slope', 'Disparity', 'Volatility', 'Vol_Change', 'Position']
        last_row = df.iloc[[-1]][feature_cols].fillna(0) # NaNs 처리
        
        return last_row.values

    def predict(self, kospi_data):
        """
        Args:
            kospi_data: Env에서 받은 (Window, 6) 크기의 데이터 (Open, High, Low, Close, Volume, Change)
        Returns:
            regime (int): 0, 1, 2 중 하나
        """
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
        # 1. 모델 및 스케일러 로드
        self.pca = joblib.load(pca_path)
        self.scaler = joblib.load(scaler_path)
        
        self.feature_names = [
            "at", "cl", "cr", "dr", "em", "et", "lr", "nc", "npm", 
            "of", "opm", "re", "wc", "roa", "tb"
        ]

    def calculate_ratios(self, row_data):
        """Raw 재무 데이터를 받아 15개 비율 계산"""
        # 입력: [Netincome, Totalequity, ..., Noncurrentassets] 순서의 1차원 배열
        # 편의상 딕셔너리 매핑 (인덱스 주의! Env에서 넘겨주는 순서와 일치해야 함)
        # Env에서는 kor_bf에 12개 값을 넣어서 보냄
        
        (Netincome, Totalequity, Totalassets, Operatingincome, Revenue, 
         Totalliabilities, Currentassets, Currentliabilities, Pretaxincome, 
         Retainedearnings, Noncurrentliabilities, Noncurrentassets) = row_data

        # 0으로 나누기 방지 (Safe Division)
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
        """
        Args:
            kor_bf_list: Env에서 받은 리스트 [stock1_raw, stock2_raw, ...]
        Returns:
            pca_vals (4,): 포트폴리오 평균 PC값
        """
        processed_data = []
        
        for raw_data in kor_bf_list:
            # 데이터가 0으로 패딩된 경우(데이터 없음) 스킵 혹은 0 처리
            if np.all(raw_data == 0):
                processed_data.append(np.zeros(15))
                continue
                
            # 1. 비율 계산 (12개 Raw -> 15개 Ratio)
            ratios = self.calculate_ratios(raw_data)
            processed_data.append(ratios)
            
        if not processed_data:
            return np.zeros(self.pca.n_components_)

        X = np.array(processed_data) # (N_stocks, 15)
        
        # 2. Scaling (저장된 Scaler 사용)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled) # NaN 안전장치
        
        # 3. PCA 변환
        pcs = self.pca.transform(X_scaled) # (N_stocks, 5)
        
        # 4. 포트폴리오 전체 평균 PC값 리턴 (Env 요구사항)
        # 혹은 각 주식별 PC가 필요하면 pcs 그대로 리턴
        avg_pcs = np.mean(pcs, axis=0)
        
        return avg_pcs[:4] # 4개만 쓴다면 슬라이싱
class Cov_Model:
    def __init__(self):
        pass
        
    def calculate_changes(self, price_series, vol_series):
        """
        20일치 시계열을 받아 변화율 벡터 생성
        Output: (38,) 크기의 1차원 벡터 (가격변화율 19개 + 거래량변화율 19개)
        """
        # 1. 가격 변화율 (Returns)
        # (P_t - P_t-1) / P_t-1
        price_diff = price_series[1:] - price_series[:-1]
        price_returns = price_diff / (price_series[:-1] + 1e-8)
        
        # 2. 거래량 변화율 (Volume Change)
        # (V_t - V_t-1) / V_t-1
        # 거래량은 0인 경우가 많으므로 안전장치 필수
        vol_diff = vol_series[1:] - vol_series[:-1]
        vol_returns = vol_diff / (vol_series[:-1] + 1e-8)
        
        # 3. 두 특징을 하나로 합침 (Concatenate)
        # 예: [r1, r2, ..., r19, v1, v2, ..., v19]
        # 이렇게 하면 가격 움직임과 거래량 움직임 패턴을 동시에 고려하여 상관계수를 계산함
        combined_features = np.concatenate([price_returns, vol_returns])
        
        return combined_features

    def __call__(self, us_tick, kor_tick):
        """
        Args:
            us_tick: List of arrays (N_us, Window, 2) -> [Adj Close, Volume]
            kor_tick: List of arrays (N_kr, Window, 5) -> [..., Close, Volume]
        Returns:
            corr_matrix: (N_total, N_total) 상관계수 행렬
        """
        all_assets_vectors = []
        
        # 1. 미국 주식 처리
        for tick in us_tick:
            # 데이터가 충분하지 않으면(20일 미만) 0으로 채움
            if len(tick) < 2:
                # 변화율 계산시 길이가 (Window-1)*2가 됨. Window=20이면 38
                # 임시로 0 벡터 삽입 (길이 38)
                all_assets_vectors.append(np.zeros(38))
                continue
                
            prices = tick[:, 0] # Adj Close
            volumes = tick[:, 1] # Volume
            
            vec = self.calculate_changes(prices, volumes)
            all_assets_vectors.append(vec)

        # 2. 한국 주식 처리
        for tick in kor_tick:
            if len(tick) < 2:
                all_assets_vectors.append(np.zeros(38))
                continue
            
            prices = tick[:, 3] # Close (4번째 컬럼)
            volumes = tick[:, 4] # Volume (5번째 컬럼)
            
            vec = self.calculate_changes(prices, volumes)
            all_assets_vectors.append(vec)
            
        # 3. 상관계수 행렬 계산
        if not all_assets_vectors:
            return np.zeros((1, 1))

        # (N_assets, 38) 형태의 행렬
        data_matrix = np.array(all_assets_vectors)
        
        # numpy corrcoef는 행(Row)을 변수(Asset)로 인식하므로 그대로 넣으면 됨
        # 결과: (N_assets, N_assets)
        corr_matrix = np.corrcoef(data_matrix)
        
        # 4. NaN 처리 (변동성이 0인 경우 상관계수가 NaN이 나옴 -> 0으로 대체)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        return corr_matrix