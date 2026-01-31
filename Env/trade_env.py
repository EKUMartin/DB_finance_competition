from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
# environment 정의
import numpy as np
import pandas as pd
import torch
import random
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
        self.risk_lambdas = {0: 0, 1: 0.00001, 2: 0.005}#안정,불안정, 폭락
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
        min_play_steps=2500
        last_possible_start = len(self.dates) - min_play_steps - 1
        self._time_index = random.randint(self.time_window,last_possible_start)
        self.budget=self._initial_budget
        self.observation = self._get_state()
        self._done=False
        self.reward=[0]
        self.total_reward=[0]
        return self.observation

    
    def step(self, action):
        t = self._time_index
        t_next = t + 1
        if t_next + 1 >= len(self.dates):
            total_return = (self.portfolio_value - self._initial_budget) / self._initial_budget
            terminal_bonus =total_return * 100.0 
            print(f"@@@완주!End of Data!@@@ Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")
            info = {"reason": "end_of_data", "return": total_return}
            return self.observation, terminal_bonus, True, info
        # -------------------------------------------------------
        cost = self.cal_cost(action)
        
        earnings = self.cal_earnings(action)
        
        port_earnings = self.port_earnings(earnings)
        
        reward = self.get_reward(port_earnings, action)



        self.update_state(action, cost)
        
        self._time_index = t_next
        next_state = self._get_state()

        done = self._is_done()
        
        
        if done:
            total_return = (self.portfolio_value - self._initial_budget) / self._initial_budget
            terminal_bonus = total_return * 10000
            reward += terminal_bonus 
            print(f"###파산!Bankrupt!### Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")
        
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
            bf_years = pd.to_datetime(bf_data['Date']).dt.year.values
            matching_indices = np.where(bf_years == int(ko_year))[0]
            fin_values = bf_data.iloc[matching_indices[0]][fin_cols].values
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
        cov=self.cov_model(us_tick,kor_tick)
        temp_current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[time_index]['Close']
            temp_current_prices.append(price)
        temp_current_prices = np.array(temp_current_prices)
        
        # 현재 평가액 계산
        stock_val = np.sum(self.portfolio * temp_current_prices)
        total_val = self.budget + stock_val
        
        # 비중 계산 (이걸 넘겨줘야 모델이 이해함)
        if total_val > 0:
            w_stock = (self.portfolio * temp_current_prices) / total_val
            w_cash = self.budget / total_val
            current_weights = np.concatenate(([w_cash], w_stock))
        else:
            # 파산 시 현금 100% 처리
            current_weights = np.zeros(self.portfolio_size + 1)
            current_weights[0] = 1.0

        state = {
            'regime': self.current_regime,
            'pca': pca,
            'cov': cov,
            'kor_feat': kor_features,
            'us_feat': us_features,
            'weights': current_weights, # [중요] 주식 수가 아니라 비중을 넘김
        }
        return state
    
    def _is_done(self): #파산했는지 확인)
        if self.portfolio_value < self._initial_budget * 0.5:
            return True
        return False
    
    # def record(self):#action, reward DB에 저장
        
    

    # def trans_actions(action):#action 포트폴리오 비중으로 변환
    #     trans_actions=0
    #     return trans_actions
    
    def port_earnings(self,earnings):#portfolio 수익 계산
        v_old = self.portfolio_value
        v_new = v_old + earnings
        if v_old <= 0 or v_new <= 0:
            performance = 0
        else:
            performance = np.log(v_new / v_old)
        return performance
    def cal_cost(self, action): 
        t = self._time_index + 1
        current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[t]['Close']
            current_prices.append(price)
        current_prices = np.array(current_prices)
        
        # 현재 포트폴리오 가치 계산
        stock_val = np.sum(self.portfolio * current_prices)
        total_val = self.budget + stock_val
        
        # 1. 현재 비중(w_old) 계산
        if total_val > 0:
            w_stock = (self.portfolio * current_prices) / total_val 
            w_cash = self.budget / total_val                    
            w_old = np.concatenate(([w_cash], w_stock)) 
        else:
            w_old = np.zeros(self.portfolio_size + 1)
     
        diff = w_old - action 
        stock_diff = diff[1:] 
        
        # 회전율 계산
        raw_turnover = np.sum(np.abs(stock_diff)) / 2
        
        # 노이즈 필터링
        if raw_turnover < 0.03: 
            self.turnover = 0.0
        else:
            self.turnover = raw_turnover
    
        # 비용 계산
        cost = total_val * 0.002 * self.turnover
        return cost

    def get_reward(self, port_earnings, action):
     
            if port_earnings<0:
                performance = port_earnings*10
            else:
                performance = port_earnings*10
    
            current_lambda = self.risk_lambdas.get(self.current_regime, 0.01)
            
            t = self._time_index + 1
            w = self.time_window
            stock_weights = action[1:] # 주식 비중만 추출
            
            # 과거 w 기간 동안의 수익률 데이터 준비
            price_history = []
            for i in range(self.portfolio_size):
                ticker = self.ticker_list[i]
                
                start_idx = max(0, t - w)
                prices = self.kor_dict[ticker].iloc[start_idx : t]['Close'].values
                price_history.append(prices)
                
            price_history = np.array(price_history).T
            
            returns_hist = (price_history[1:] - price_history[:-1]) / (price_history[:-1] + 1e-8)
            
            if returns_hist.shape[0] > 1:
                cov_matrix = np.cov(returns_hist, rowvar=False)
            
                port_variance = np.dot(stock_weights.T, np.dot(cov_matrix, stock_weights))
                
                risk_penalty = current_lambda * port_variance
            else:
                risk_penalty = 0.0
                            
            turnover = getattr(self, 'turnover', 0.0)

            turnover_penalty = 0.001 * turnover             
            
            reward = performance - risk_penalty - turnover_penalty

            # if t % 100 == 0:
            # print(f"R: {reward:.5f} | Perf: {performance:.5f} | Risk: {risk_penalty:.5f} | Turn: {turnover_penalty:.5f}")
                
            return reward

    def update_state(self, target_weights, cost_val):

        t = self._time_index + 1
        current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[t]['Close']
            current_prices.append(price)
        current_prices = np.array(current_prices)
        

        if cost_val == 0.0:
            stock_val = np.sum(self.portfolio * current_prices)
            self.portfolio_value = self.budget + stock_val
            return 

        total_val = self.budget + np.sum(self.portfolio * current_prices)
        available_value = total_val - cost_val

        target_stock_w = target_weights[1:]
        

        target_stock_amounts = available_value * target_stock_w
        
    
        new_portfolio_shares = np.floor(target_stock_amounts / (current_prices + 1e-8))
        
       
        stock_buy_value = np.sum(new_portfolio_shares * current_prices)
        self.budget = available_value - stock_buy_value
        self.portfolio = new_portfolio_shares
        
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.budget + stock_buy_value

    def cal_earnings(self, action):

        t = self._time_index + 1
        current_prices = []
        next_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            current_prices.append(self.kor_dict[ticker].iloc[t]['Close'])
            next_prices.append(self.kor_dict[ticker].iloc[t+1]['Close'])
            
        current_prices = np.array(current_prices)
        next_prices = np.array(next_prices)
        
    
        stock_weights = action[1:]
        cash_weight = action[0]
        
        current_total_val = self.portfolio_value
        
        # 주식 수익
        target_stock_amt = current_total_val * stock_weights
        target_stocks = np.floor(target_stock_amt / (current_prices + 1e-8))
        stock_future_val = np.sum(target_stocks * next_prices)
        
        # 현금 수익 (이자율 0 가정, 그대로 유지)
        cash_future_val = current_total_val * cash_weight 
        
        total_future_val = stock_future_val + cash_future_val
        

        
        return total_future_val - current_total_val