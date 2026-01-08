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
    def __init__(self,time_window,budget,tick_list):
        self._time_window=time_window
        self._time_index=time_window-1
        self._initial_budget=budget
        self.budget=budget
        self.tick_list=tick_list
        self.portfolio_size=len(self.tick_list)
        self._done=False
        self.portfoli=[0]*self.porfolio_size

    def reset(self):#포트폴리오,리워드,observation,자산 초기화
        self.portfolio_value=[self._initial_budget]
        self.portfolio=np.zeros(self.portfolio_size,dtype=np.float32)
        self.budget=self._initial_budget
        self.observation=self.market[:self._time_window]
        self._done=False
        self.reward=[0]
        self.total_reward=[0]
        return self.observation

    def step(self, action):
        t = self._time_index
        t_next = t + 1
        # 시간 넘어 가면 종료
        if t_next >= len(self.market):
            return None, 0.0, True, {"reason": "end_of_data", "t": t}
        trans_action=self.trans_actions(action)
        cost=self.cal_turnover(trans_action)
        earnings=self.cal_earnings(trans_action)
        
        #포트폴리오 수익률 계산
        port_earnings=self.port_earnings(earnings,cost)
        reward=self.get_reward(port_earnings)
        #상태 업데이트
        self.update_state()
        # 다음 상태
        self._time_index = t_next
        next_state=self._get_state()
        done=self._is_done()
        info={}
        return next_state, float(reward),bool(done),info 
    def _get_state(self):#입력된 시점의 상태 리턴
       
        return state
    
    def _is_done(self):#파산했는지 확인
        earnings=self.cal_earnings(self.portfolio)
        if earnings<0:
            return True
    
    def record(self):#action, reward DB에 저장
        
    def get_reward(self,port_earnings):# 포트폴리오 수익을 바탕으로 reward 계산
        
        return 
    def cal_earnings(self):#주식별 수익 계산
        earnings=0
        return earnings
    def cal_cost(self):#주식별 cost 계산
        cost=0
        return cost
    def trans_actions(action):#action 포트폴리오 비중으로 변환
        trans_actions=0
        return trans_actions
    
    def port_earnings(self):#portfolio 수익 계산
        port_earnings=0
        return port_earnings
    
    def update_state(self):#다음 에피소드에서 사용 할 포트폴리오,시점,포트폴리오 가치 업데이트
