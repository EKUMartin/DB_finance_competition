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
        
        # [수정 1] 데이터 끝(만기) 도달 여부 확인
        # 여기서 바로 리턴하지 않고, 플래그만 세워둡니다.
        is_end_of_data = (t_next + 1 >= len(self.dates))
        
        # 1. 비용 계산
        cost = self.cal_cost(action)
        
        # 2. 시장 수익 계산
        gross_earnings = self.cal_earnings(action)
        
        # 3. 순수익
        net_earnings = gross_earnings - cost 
        
        # 4. 수익률
        port_earnings = self.port_earnings(net_earnings) 
        
        # 5. 기본 보상 (Daily Reward)
        reward = self.get_reward(port_earnings, action)
        
        # 6. 상태 업데이트
        self.update_state(action, cost)
        
        # 7. 종료 조건 체크 (파산 여부)
        is_bankrupt = self._is_done()
        
        # [🔥 핵심 수정] 종료(Done)는 '파산'하거나 '만기'일 때 모두 True
        done = is_bankrupt or is_end_of_data
        
        # 8. [🔥 핵심 수정] 졸업 선물 (Terminal Reward) 주는 로직
        if done:
            # 최종 수익률 계산: (최종자산 - 원금) / 원금
            # 예: 1억 -> 1.2억 (+0.2), 1억 -> 5천만 (-0.5)
            total_return = (self.portfolio_value - self._initial_budget) / self._initial_budget
            
            # 보너스 계산 (가중치 100배)
            # - 파산 시: -0.5 * 100 = -50점 (강력한 처벌)
            # - 생존 및 수익 시: +0.2 * 100 = +20점 (달콤한 보상)
            terminal_bonus = total_return * 100.0
            
            reward += terminal_bonus
            
            # 로그 출력 (확인용)
            if is_bankrupt:
                print(f"💀 Bankrupt! Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")
            else:
                print(f"🎉 Survival! Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")

        # 9. 다음 상태 준비
        self._time_index = t_next
        
        # 만약 데이터가 끝났으면 next_state를 구할 수 없으므로 현재 state를 반환하거나 None 처리
        if is_end_of_data:
             # 마지막 스텝에서는 next_state가 중요하지 않음 (어차피 done=True라 학습 종료)
             # 형식상 현재 상태를 리턴해줌
            next_state = self.observation 
        else:
            next_state = self._get_state()
            
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
        # action: [Cash_Weight, Stock_1_W, Stock_2_W, ...] (크기: N+1)
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
            
        # [🔥 수정됨] 들여쓰기를 밖으로 꺼냈습니다. (이제 정상 실행됩니다)
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
            # action: [Cash, Stocks...]
            
            # 1. 수익률 (Performance)
            # 이미 port_earnings 계산 시 (v_new / v_old)에 거래비용(cost)이 반영되어 있습니다.
            # 따라서 보상 식에서 cost를 또 뺄 필요가 없습니다.
            performance = port_earnings*100
            
            # 2. 리스크 (Volatility Penalty)
            # HMM Regime에 따라 Lambda를 가져옵니다.
            current_lambda = self.risk_lambdas.get(self.current_regime, 0.01)
            
            t = self._time_index + 1
            w = self.time_window
            stock_weights = action[1:] # 주식 비중만 추출
            
            # 과거 w 기간 동안의 수익률 데이터 준비
            price_history = []
            for i in range(self.portfolio_size):
                ticker = self.ticker_list[i]
                # [수정] iloc 슬라이싱 범위 안전하게 처리
                start_idx = max(0, t - w)
                prices = self.kor_dict[ticker].iloc[start_idx : t]['Close'].values
                price_history.append(prices)
                
            price_history = np.array(price_history).T
            
            # 수익률 변환 (Shape: [w-1, N_stocks])
            returns_hist = (price_history[1:] - price_history[:-1]) / (price_history[:-1] + 1e-8)
            
            if returns_hist.shape[0] > 1:
                cov_matrix = np.cov(returns_hist, rowvar=False)
                # 포트폴리오 분산 = w.T * Cov * w
                port_variance = np.dot(stock_weights.T, np.dot(cov_matrix, stock_weights))
                
                # 리스크 = lambda * 분산 (일반적인 Mean-Variance Optimization 식)
                # 리스크 값이 너무 커지지 않도록 스케일 조정 필요할 수 있음
                risk_penalty = current_lambda * port_variance
            else:
                risk_penalty = 0.0
                
            # 3. 회전율 페널티 (Turnover Penalty)
            # 학습 초반에 너무 세게 잡으면 아무것도 안 함. 
            # performance가 대략 0.01(1%) 내외이므로, 페널티는 그보다 작아야 함 (예: 0.0005)
            # 기존 0.05 * lambda * turnover는 너무 컸을 수 있음.
            
            turnover = getattr(self, 'turnover', 0.0)
            
            # [수정] 고정된 작은 상수를 곱하거나, lambda와 무관하게 아주 작게 설정
            turnover_penalty = 0.001 * turnover 
            
            # 4. 최종 보상 계산
            # 2*performance는 수익 추구를 강조하기 위함 (선택 사항)
            # cost 항 제거함 (performance에 이미 반영됨)
            
            reward = performance - risk_penalty - turnover_penalty
            
            # [디버깅용 출력] - 학습이 안될 때 이 주석을 풀어서 값들의 크기(Scale)를 비교해보세요.
            # if t % 100 == 0:
            # print(f"R: {reward:.5f} | Perf: {performance:.5f} | Risk: {risk_penalty:.5f} | Turn: {turnover_penalty:.5f}")
                
            return reward

    def update_state(self, target_weights, cost_val):
        # target_weights: [Cash, Stocks...]
        t = self._time_index + 1
        current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[t]['Close']
            current_prices.append(price)
        current_prices = np.array(current_prices)
        
        # [🔥 여기부터 추가된 부분입니다] 
        # cost_val이 0이면 (cal_cost에서 Hold로 판정), 
        # 포트폴리오(주식 수)를 변경하지 않고 가치만 갱신하고 끝냅니다.
        if cost_val == 0.0:
            stock_val = np.sum(self.portfolio * current_prices)
            self.portfolio_value = self.budget + stock_val
            return # 여기서 함수 종료! (밑에 재분배 로직 실행 안 함)
        # [🔥 여기까지 추가]

        # -----------------------------------------------------------
        # 아래는 기존 로직 그대로 실행 (cost_val > 0 일 때만)
        # -----------------------------------------------------------
        
        # 거래비용 차감 후 가용 자산
        # cost_val은 금액(won) 단위여야 함
        total_val = self.budget + np.sum(self.portfolio * current_prices)
        available_value = total_val - cost_val
        
        # 현금 비중과 주식 비중 분리
        target_stock_w = target_weights[1:]
        
        # 주식 매수 목표 금액
        target_stock_amounts = available_value * target_stock_w
        
        # 주식 수 계산 (내림)
        new_portfolio_shares = np.floor(target_stock_amounts / (current_prices + 1e-8))
        
        # 실제 주식 매수 후 남은 돈은 현금(Budget)으로
        stock_buy_value = np.sum(new_portfolio_shares * current_prices)
        self.budget = available_value - stock_buy_value
        self.portfolio = new_portfolio_shares
        
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.budget + stock_buy_value

    def cal_earnings(self, action):
        # action: [Cash, Stocks...]
        # 수익률 계산은 '실제 보유 주식' 기준이므로 action(목표 비중)보다는
        # update_state가 끝난 후의 self.portfolio로 계산하는 게 맞지만,
        # 여기 로직상으로는 '다음 스텝의 예상 수익'을 계산하는 구조로 보임.
        
        t = self._time_index + 1
        current_prices = []
        next_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            current_prices.append(self.kor_dict[ticker].iloc[t]['Close'])
            next_prices.append(self.kor_dict[ticker].iloc[t+1]['Close'])
            
        current_prices = np.array(current_prices)
        next_prices = np.array(next_prices)
        
        # [🔥 수정] 주식 비중만 발라내서 계산
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
        
        # 원래 함수가 'Earnings(차액)'이 아니라 '미래 가치'를 리턴하는 구조였으면 이것 사용
        # 사용자님 코드는 stock_earnings만 리턴했었음.
        # 포트폴리오 전체 미래 가치를 리턴해야 port_earnings 함수에서 log(new/old) 가능
        
        return total_future_val - current_total_val # 순수익 금액 리턴