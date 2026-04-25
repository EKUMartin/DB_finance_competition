from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

# environment 정의
import numpy as np
import os
import pandas as pd
import torch
import random

class Environment:
    def __init__(self, time_window, budget, kor, us, kfb, kospi, pca_model, hmm_model, cov):
        self._initial_budget = budget
        self.budget = budget
        self.pca_model = pca_model
        self.hmm_model = hmm_model
        self.cov_model = cov
        self.kospi = kospi
        self.us = us
        self.kor = kor
        self.kfb = kfb
        self.risk_lambdas = {0: 0, 1: 0.00001, 2: 0.005} # 안정, 불안정, 폭락
        self.current_regime = 0
        
        unique_tickers = kor['Tick_id'].unique()
        self.ticker_list = sorted(unique_tickers)
        self.portfolio_size = len(self.ticker_list)
        
        us_tickers = us['Tick_id'].unique()
        self.us_ticker_list = sorted(us_tickers)
        self.us_ticker_size = len(self.us_ticker_list)
        
        self.dates = sorted(kor['Date'].unique())
        self.time_window = time_window
        self._time_index = 2*time_window
        
        self.kor_dict = dict(list(self.kor.groupby('Tick_id')))
        self.us_dict = dict(list(self.us.groupby('Tick_id')))
        self.kfb_dict = dict(list(self.kfb.groupby('Tick_id')))
        
        self.bool_init = True
        self._done = False
        self.current_episode = 1
        self.reset()
    
    def reset(self):
        self.portfolio_value = self._initial_budget
        self.portfolio = np.zeros(self.portfolio_size, dtype=np.float32)
        min_play_steps=1024
        last_possible_start = len(self.dates) - min_play_steps - 1
        self._time_index = random.randint(self.time_window,last_possible_start)
        self.budget = self._initial_budget
        self.observation = self._get_state()
        self._done = False
        self.reward = [0]
        self.total_reward = [0]
        self.trade_log = []
        return self.observation

    def step(self, action):
        t = self._time_index
        t_next = t + 1
        
        if t_next + 1 >= len(self.dates):
            total_return = (self.portfolio_value - self._initial_budget) / self._initial_budget
            terminal_bonus = total_return * 100 if total_return>0 else total_return*10
            print(f"@@@완주!End of Data!@@@ Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")
            info = {"reason": "end_of_data", "return": total_return}
            self.save_log_to_csv()
            return self.observation, terminal_bonus, True, info
            
        # 1. Action 처리
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) > 0:
            norm_action = action / np.sum(action)
        else:
            norm_action = np.zeros_like(action)
            norm_action[0] = 1.0
            
        # 2. Hold 판단
        t_kor = self._time_index
        current_prices = np.array([self.kor_dict[tk].iloc[t_kor]['Close'] for tk in self.ticker_list])
        total_val = self.budget + np.sum(self.portfolio * current_prices)
        
        if total_val > 0:
            w_stock = (self.portfolio * current_prices) / total_val
            w_cash = self.budget / total_val
            w_old = np.concatenate(([w_cash], w_stock))
        else:
            w_old = np.zeros(self.portfolio_size + 1)
            w_old[0] = 1.0
            
        turnover = np.sum(np.abs(norm_action - w_old)) / 2
        is_hold = False
        
        # 3. 매매 실행
        cost = self.cal_cost(norm_action)
        earnings = self.cal_earnings(norm_action, is_hold)
        port_earnings = self.port_earnings(earnings, total_val)
        reward = self.get_reward(port_earnings, norm_action)

        self.update_state(norm_action, cost, is_hold)
        
        self._time_index = t_next
        next_state = self._get_state()
        done = self._is_done()
        
        if done:
            total_return = (self.portfolio_value - self._initial_budget) / self._initial_budget
            terminal_bonus = abs(total_return)*20
            reward -= terminal_bonus 
            self.save_log_to_csv()
            print(f"###파산!Bankrupt!### Return: {total_return*100:.2f}% | Bonus: {terminal_bonus:.2f}")
        
        return next_state, float(reward), bool(done), {}

    def _get_state(self):
        time_index = self._time_index
        time_window = self.time_window
        
        # Target Columns
        kor_target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        kospi_target_cols = ['Open','High','Low','Close','Volume','Change']
        us_target_cols = ['Adj_close', 'Volume']
        fin_cols = [
            'Netincome', 'Totalequity', 'Totalassets', 'Operatingincome', 'Revenue',
            'Totalliabilities', 'Currentassets', 'Currentliabilities', 'PretaxIncome',
            'Retainedearnings', 'Noncurrentliabilities', 'Noncurrentassets'
        ]
        
        # 기준 날짜 (한국 주식 거래일 기준)
        ko_now = self.dates[time_index]  
        ko_year = ko_now.year - 1
        
        kospi_mask = self.kospi['Date'] <= ko_now
        kospi_end_pos = kospi_mask.sum() # True 개수가 곧 마지막 인덱스+1
        kospi_start_pos = max(0, kospi_end_pos - time_window)
        
        # HMM 모델용 데이터
        kospi = self.kospi.iloc[kospi_start_pos:kospi_end_pos][kospi_target_cols].values

        kor_bf = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            bf_data = self.kfb_dict[ticker]
            bf_years = pd.to_datetime(bf_data['Date']).dt.year.values
            matching_indices = np.where(bf_years == int(ko_year))[0]
            if len(matching_indices) > 0:
                fin_values = bf_data.iloc[matching_indices[0]][fin_cols].values
            else:
                fin_values = np.zeros(len(fin_cols))
            kor_bf.append(fin_values)

        us_tick = []
        us_features = []
        
        for i in range(self.us_ticker_size):
            ticker = self.us_ticker_list[i]
            us_data = self.us_dict[ticker]
            
            # 미국 데이터도 날짜 기준으로 슬라이싱
            mask = us_data['Date'] < ko_now
            end_pos = mask.sum()
            start_pos = max(0, end_pos - time_window)
            
            us_input = us_data.iloc[start_pos:end_pos][us_target_cols].values
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


        
        kor_tick = []
        kor_features = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            kor_data = self.kor_dict[ticker]
            
            kor_input = kor_data.iloc[time_index-time_window+1:time_index+1][kor_target_cols].values
            
            kor_tick.append(kor_input)
            
            if len(kor_input) >= 2:
                close_t = kor_input[-1, 3]   
                close_prev = kor_input[-2, 3]
                price_chg = (close_t - close_prev) / (close_prev + 1e-8)

                vol_t = kor_input[-1, 4] 
                vol_prev = kor_input[-2, 4]
                vol_chg = (vol_t - vol_prev) / (vol_prev + 1e-8)
    
                kor_features.append([price_chg, vol_chg]) 
            else:
                kor_features.append([0.0, 0.0])
                
  
        regime = self.hmm_model(kospi)
        self.current_regime = int(regime)
        pca_list = []
        for bf in kor_bf:
            single_pca = self.pca_model([bf])
            if isinstance(single_pca, torch.Tensor):
                pca_list.append(single_pca.detach().cpu().numpy().flatten())
            else:
                pca_list.append(np.array(single_pca).flatten())
        pca = np.array(pca_list)
        kor_tick_aligned = [k[:, [3, 4]] for k in kor_tick]

        cov = self.cov_model(us_tick, kor_tick_aligned)

        temp_current_prices = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            price = self.kor_dict[ticker].iloc[time_index]['Close']
            temp_current_prices.append(price)
        temp_current_prices = np.array(temp_current_prices)

        stock_val = np.sum(self.portfolio * temp_current_prices)
        total_val = self.budget + stock_val
        
        if total_val > 0:
            w_stock = (self.portfolio * temp_current_prices) / total_val
            w_cash = self.budget / total_val
            current_weights = np.concatenate(([w_cash], w_stock))
        else:
            current_weights = np.zeros(self.portfolio_size + 1)
            current_weights[0] = 1.0

        state = {
            'regime': self.current_regime,
            'pca': pca,
            'cov': cov,
            'kor_feat': kor_features,
            'us_feat': us_features,
            'weights': current_weights,
        }
        return state
    
    def _is_done(self):
        if self.portfolio_value < self._initial_budget * 0.3:
            return True
        return False
    
    def port_earnings(self, earnings, current_total_val):
        v_old = current_total_val
        v_new = v_old + earnings
        if v_old <= 0 or v_new <= 0:
            performance = 0
        else:
            performance = np.log(v_new / v_old)
        return performance

    def cal_cost(self, norm_action): 
        t_kor = self._time_index+1
        current_prices = np.array([self.kor_dict[tk].iloc[t_kor]['Open'] for tk in self.ticker_list])
        
        stock_val = np.sum(self.portfolio * current_prices)
        total_val = self.budget + stock_val
        
        if total_val > 0:
            w_stock = (self.portfolio * current_prices) / total_val 
            w_cash = self.budget / total_val                    
            w_old = np.concatenate(([w_cash], w_stock)) 
        else:
            w_old = np.zeros(self.portfolio_size + 1)
     
        diff = w_old - norm_action 
        log_row = {
            'Episode': self.current_episode,
            'Date': self.dates[self._time_index], 
            'Total_Turnover': np.sum(np.abs(diff))/2,
            'Portfolio_Value': self.budget + np.sum(self.portfolio * current_prices)
        }
        
        log_row['Cash_Diff'] = diff[0]
        log_row['Cash_Weight'] = norm_action[0]
        
        for i, ticker in enumerate(self.ticker_list):
            log_row[f'{ticker}_Diff'] = diff[i+1]
            log_row[f'{ticker}_Weight'] = norm_action[i+1]
            
        self.trade_log.append(log_row)
        stock_diff = diff[1:] 
        
        raw_turnover = np.sum(np.max(stock_diff,0)) 
        self.turnover = raw_turnover
        cost = total_val * 0.00215 * self.turnover 
        return cost

    def get_reward(self, port_earnings, action):
        performance = port_earnings * 100
        
        current_lambda = self.risk_lambdas.get(self.current_regime, 0)
        
        t = self._time_index + 1
        w = self.time_window
        stock_weights = action[1:] 

        price_history = []
        for i in range(self.portfolio_size):
            ticker = self.ticker_list[i]
            start_idx = max(0, t - w)
            prices = self.kor_dict[ticker].iloc[start_idx : t]['Close'].values
            price_history.append(prices)
        
        price_history = np.array(price_history).T
        if len(price_history) > 1:
            returns_hist = (price_history[1:] - price_history[:-1]) / (price_history[:-1] + 1e-8)
        else:
            returns_hist = np.zeros_like(price_history)

        if returns_hist.shape[0] > 1:
            cov_matrix = np.cov(returns_hist, rowvar=False)
            port_variance = np.dot(stock_weights.T, np.dot(cov_matrix, stock_weights))
            risk_penalty = current_lambda * port_variance*100.0
        else:
            risk_penalty = 0.0
                        
        turnover = getattr(self, 'turnover', 0.0)
        turnover_penalty = turnover * 0.00215 *100
        
        reward = performance - turnover_penalty# -risk_penalty
        return reward

    def update_state(self, target_weights, cost_val, is_hold=False):
        t_next = self._time_index+1
        prices_next_open = np.array([self.kor_dict[tk].iloc[t_next]['Open'] for tk in self.ticker_list])
        prices_next_close = np.array([self.kor_dict[tk].iloc[t_next]['Close'] for tk in self.ticker_list])
        
        if is_hold:
            self.portfolio_value = self.budget + np.sum(self.portfolio * prices_next_close)
            return

        # 내일 시가 기준으로 총 자산 평가
        total_val_open = self.budget + np.sum(self.portfolio * prices_next_open)
        available_value = total_val_open - cost_val 

        target_stock_w = target_weights[1:]
        target_stock_amounts = available_value * target_stock_w
        
        # 내일 시가로 주식 매수
        new_portfolio_shares = np.floor(target_stock_amounts / (prices_next_open + 1e-8))
        stock_buy_value = np.sum(new_portfolio_shares * prices_next_open)
        
        # 잔여 현금 및 포트폴리오 상태 업데이트
        self.budget = available_value - stock_buy_value
        self.portfolio = new_portfolio_shares
        
        # 다음 스텝으로 넘어가기 전, 최종 포트폴리오 가치는 내일 '종가' 기준으로 세팅
        self.portfolio_value = self.budget + np.sum(self.portfolio * prices_next_close)

    def cal_earnings(self, norm_action, is_hold=False):#내일 종가를 보고 내일 시가 행동에 대한 수익을 얻음
        t_kor = self._time_index 
        t_kor_next = t_kor + 1
        
        if t_kor_next >= len(self.dates):
            return 0
            
        prices_t = np.array([self.kor_dict[tk].iloc[t_kor]['Close'] for tk in self.ticker_list])
        prices_t_next = np.array([self.kor_dict[tk].iloc[t_kor_next]['Close'] for tk in self.ticker_list])
        prices_next_open=prices_next_open = np.array([self.kor_dict[tk].iloc[t_kor_next]['Open'] for tk in self.ticker_list])
        current_total_val = self.budget + np.sum(self.portfolio * prices_t)
        val_at_open=self.budget + np.sum(self.portfolio * prices_next_open)
        if is_hold:
            stock_future_val = np.sum(self.portfolio * prices_t_next)
            cash_future_val = self.budget
            return (stock_future_val + cash_future_val) - current_total_val

        cash_weight = norm_action[0]
        stock_weights = norm_action[1:]
        
        target_stock_amt = val_at_open * stock_weights
        target_stocks = np.floor(target_stock_amt / (prices_next_open + 1e-8))
        actual_stock_cost = target_stocks * prices_next_open
        remainder = target_stock_amt - actual_stock_cost
        
        stock_future_val = np.sum(target_stocks * prices_t_next)
        cash_future_val = (val_at_open * cash_weight) + np.sum(remainder)
        
        return (stock_future_val + cash_future_val) - current_total_val

    def save_log_to_csv(self):
        if len(self.trade_log) == 0:
            return
            
        filename = "all_trade_log.csv"
        df = pd.DataFrame(self.trade_log)
        
        if not os.path.exists(filename):
            df.to_csv(filename, index=False, mode='w', encoding='utf-8-sig')
        else:
            df.to_csv(filename, index=False, mode='a', header=False, encoding='utf-8-sig')
            
        print(f" >> 💾 매매 기록 저장 완료 (-> {filename})")