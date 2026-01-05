# environment 정의
import numpy as np
import pandas as pd
import torch
class Environment:
    def __init__(self, df_close_us: pd.DataFrame, df_close_kr:pd.DataFrame, predictor, device,
                 seq_window=30, bol_window=20, cost_rate=0.0005,
                 normalize_action=True):
        self.df_us = df_close_us.dropna().copy()
        self.df_kr = df_close_kr.dropna().copy()
        self.predictor = predictor.to(device)
        self.device = device
        self.seq_window = seq_window
        self.bol_window = bol_window
        self.cost_rate = cost_rate
        self.normalize_action = normalize_action
        self.tickers = list(self.df_kr.columns)
        self.N = len(self.tickers)
        self.t = None
        self.prev_w = None

    def reset(self,start_idx=None):
        self.t = (self.seq_window - 1 if start_idx is None else max(start_idx, self.seq_window - 1))
        self.prev_w = np.ones(self.N, dtype=np.float32) / self.N
        return self._get_state()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            w = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            w = np.asarray(action, dtype=np.float32).reshape(-1)
        #softmax로 변환
        
        return next_state, reward, done, info
    def _get_state(self):
       
        return state