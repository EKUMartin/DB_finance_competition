from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
# environment 정의
import numpy as np
import pandas as pd
import torch

#reset/step/_is_done/record
#record 저장 구조
class Environment:
    def __init__(self,time_window,budget,kor,us,kfb,kospi,pca_model,hmm_model,cov):
        self._initial_budget=budget
        self.budget=budget
        self.pca_model=pca_model
        self.hmm_model=hmm_model
        self.cov_model=cov
        self.kospi=kospi
        self.us=us
        self.kor=kor
        self.kfb=kfb
        self.risk_lambdas = {0: 0.001, 1: 0.05, 2: 0.1}#안정,불안정, 폭락
        self.current_regime = 0
        unique_tickers=kor['Tick_id'].unique()
        self.ticker_list=sorted(unique_tickers)
        self.portfolio_size = len(self.ticker_list)
        us_tickers=us['Tick_id'].unique()
        self.us_ticker_list=sorted(us_tickers)
        self.us_ticker_size=len(self.us_ticker_list)
        self.dates = sorted(kor['Date'].unique())
        self.time_window=time_window
        self._time_index=time_window
        self.kor_dict = dict(list(self.kor.groupby('Tick_id')))
        self.us_dict = dict(list(self.us.groupby('Tick_id')))
        self.kfb_dict=dict(list(self.kfb.groupby('Tick_id')))
        self.bool_init=True
        self._done=False
        self.reset()
    
    
    def reset(self):#포트폴리오,리워드,observation,자산 초기화
        self.portfolio_value=self._initial_budget
        self.portfolio=np.zeros(self.portfolio_size,dtype=np.float32)
        self._time_index = self.time_window
        self.budget=self._initial_budget
        self.observation = self._get_state()
        self._done=False
        self.reward=[0]
        self.total_reward=[0]
        return self.observation

    
    def step(self, action):
            t = self._time_index
            t_next = t + 1
            if t_next+1 >= len(self.dates):
                return None, 0.0, True, {"reason": "end_of_data", "t": t}
            
            cost = self.cal_cost(action)
            
            earnings = self.cal_earnings(action)
            
            port_earnings = self.port_earnings(earnings)
            
            reward = self.get_reward(port_earnings, action)

            self.update_state(action, cost)
            
            self._time_index = t_next
            next_state = self._get_state()
            done = self._is_done()
            info = {}
            return next_state, float(reward), bool(done), info
    
    def _get_state(self):#입력된 시점의 상태 리턴
        time_index=self._time_index
        time_window=self.time_window
        kor_features = []
        kor_target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        kospi_target_cols=['Open','High','Low','Close','Volume','Change' ]
        us_target_cols=['Adj_close', 'Volume']
        fin_cols = [
            'Netincome', 'Totalequity', 'Totalassets', 'Operatingincome', 'Revenue',
            'Totalliabilities', 'Currentassets', 'Currentliabilities', 'PretaxIncome',
            'Retainedearnings', 'Noncurrentliabilities', 'Noncurrentassets'
        ]
        # 기준 날짜
        ko_now = self.dates[time_index]  

        #미국 마지막 장 날짜    
        matched_rows = self.us[self.us['Date'] <= ko_now]
        
        #미국 슬라이싱 범위
        us_index = matched_rows.index[-1]
        us_index_start=us_index-time_window
        us_index_end=us_index
        ko_year=ko_now.year-1
        #HMM
        kospi=self.kospi[time_index-time_window+1:time_index+1][kospi_target_cols].values    #HMM:Open,High,Low,Close,Volume,`Change` 데이터 준비
        
        #PCA 데이터 준비
        kor_bf=[]
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            bf_data=self.kfb_dict[ticker]
            matching_data = bf_data[bf_data['Date'].dt.year == ko_year]
            fin_values = matching_data[fin_cols].values[0]
            kor_bf.append(fin_values)

        
        #cov,corr 데이터 준비
        #미국 데이터
        us_tick=[]
        us_features=[]
        for i in range(self.us_ticker_size):
            ticker = self.us_ticker_list[i]
            us_data=self.us_dict[ticker]
            us_input=us_data.iloc[us_index_start:us_index_end][us_target_cols].values
            us_tick.append(us_input)
            if len(us_input) >= 2:
                price_t = us_input[-1, 0]
                price_prev = us_input[-2, 0]
                price_chg = (price_t - price_prev) / (price_prev + 1e-8)
                vol_t = us_input[-1, 1]
                vol_prev = us_input[-2, 1]
                vol_chg = (vol_t - vol_prev) / (vol_prev + 1e-8)
                us_features.append([price_chg, vol_chg])
            else:
                us_features.append([0.0, 0.0])
        #한국 데이터
        kor_tick=[]
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            kor_data=self.kor_dict[ticker]
            kor_input=kor_data.iloc[time_index-time_window+1:time_index+1][kor_target_cols].values
            kor_tick.append(kor_input)
            kor_features.append(kor_input[-1])
        #모델 실행
        regime=self.hmm_model(kospi)
        self.current_regime = int(regime)
        pca=self.pca_model(kor_bf)
        cov=self.cov(us_tick,kor_tick)
        state = {
            'regime': self.current_regime,   # 스칼라 (0 or 1 or 2)
            'pca': pca,                 # (4,) : PC1~4 값
            'cov': cov,                      # (N_us+N_kr, N_us+N_kr)
            'kor_feat': kor_features,        # (N_kr, 5)
            'us_feat': us_features
        }
        return state
    
    def _is_done(self): #파산했는지 확인)
        if self.portfolio_value < self._initial_budget * 0.5:
            return True
        return False
    
    # def record(self):#action, reward DB에 저장
        
    def get_reward(self,port_earnings,action):# 포트폴리오 수익을 바탕으로 reward 계산
        performance = port_earnings
        current_lambda = self.risk_lambdas.get(self.current_regime, 0.01)
        cost = 0.002 * self.turnover
        t = self._time_index+1
        w = self.time_window
        weights = action
        price_history = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            prices = self.kor_dict[ticker].iloc[t-w : t]['Close'].values
            price_history.append(prices)
        price_history = np.array(price_history).T
        returns = (price_history[1:] - price_history[:-1]) / (price_history[:-1] + 1e-8)
        cov_matrix = np.cov(returns, rowvar=False)
        risk = current_lambda * np.dot(weights.T, np.dot(cov_matrix, weights))
        reward = performance - cost - risk    
        return reward
    
    def cal_earnings(self,action):#주식별 수익 계산
        t = self._time_index+1
        current_prices = []
        next_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            current_prices.append(self.kor_dict[ticker].iloc[t]['Close'])
            next_prices.append(self.kor_dict[ticker].iloc[t+1]['Close'])
        current_prices = np.array(current_prices)
        next_prices = np.array(next_prices)
        current_total_val = self.portfolio_value
        target_amount = current_total_val * action
        target_stocks = np.floor(target_amount / (current_prices + 1e-8))
        stock_earnings = np.sum(target_stocks * next_prices)
        return stock_earnings
    
    def cal_cost(self,action):#주식별 cost 계산
        t = self._time_index+1
        current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[t]['Close']
            current_prices.append(price)
        total_val = self.budget + np.sum(self.portfolio * current_prices)
        if total_val > 0:
            w_old = (self.portfolio * current_prices) / total_val
        else:
            w_old = np.zeros(self.portfolio_size)
        diff = w_old - action
        self.turnover = np.sum(np.maximum(diff, 0))
        current_prices = np.array(current_prices)    
        cost = 0.002 * self.turnover
        return cost
    # def trans_actions(action):#action 포트폴리오 비중으로 변환
    #     trans_actions=0
    #     return trans_actions
    
    def port_earnings(self,earnings):#portfolio 수익 계산
        v_old = self.portfolio_value
        v_new = earnings + self.budget
        if v_old <= 0 or v_new <= 0:
            performance = 0
        else:
            performance = np.log(v_new / v_old)
        return performance
    
    def update_state(self,target_weights, cost_ratio):#다음 에피소드에서 사용 할 포트폴리오,시점,포트폴리오 가치 업데이트
        t = self._time_index+1
        current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[t]['Close']
            current_prices.append(price)
        current_prices = np.array(current_prices)
        transaction_cost_amount = self.portfolio_value * cost_ratio
        available_value = self.portfolio_value - transaction_cost_amount
        target_amounts = available_value * target_weights
        new_portfolio_shares = np.floor(target_amounts / (current_prices + 1e-8))
        stock_value = np.sum(new_portfolio_shares * current_prices)
        self.budget = available_value - stock_value
        self.portfolio = new_portfolio_shares
        self.portfolio_value = self.budget + stock_value