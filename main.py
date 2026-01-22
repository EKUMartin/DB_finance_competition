from Data.process_data import process_data
from model_pickle.models import HMM_Model, PCA_Model, Cov_Model
from Env.trade_env import Environment
from Policy.opt_policy import ActorCritic
from Agent.ppo_agent import PPOAgent, Memory
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
def observation_to_graph(observation, num_us_nodes, device='cpu'):
    """
    Args:
        observation: Env 리턴값
        num_us_nodes: 미국 주식 개수
    """
    regime = observation['regime']
    pca_vals = observation['pca']          # (4,)
    cov_matrix = observation['cov']        # (N_us+N_kr, N_us+N_kr)
    kor_feat = np.array(observation['kor_feat']) # (N_kr, 5)
    
    # [수정] Env에서 넘겨준 미국 주식 특징 가져오기
    # (N_us, 2) 형태 -> [Price_Chg, Vol_Chg]
    us_feat = np.array(observation['us_feat'])   
    
    num_kr = len(kor_feat)
    num_pca = len(pca_vals) # 4
    num_regime = 3          # 3
    
    # -------------------------------------------------------
    # 1. Node Features 구성 (Max Dim = 5)
    # -------------------------------------------------------
    
    # (1) [수정] 미국 주식 (N_us, 5): [Price_Chg, Vol_Chg, 0, 0, 0]
    # 실제 데이터 2개 + 0 패딩 3개 = 총 5개
    us_tensor = torch.tensor(us_feat, dtype=torch.float, device=device)
    us_pad = torch.zeros((num_us_nodes, 3), device=device) # 뒤에 3개는 0으로 채움
    x_us = torch.cat([us_tensor, us_pad], dim=1)
    
    # (2) 한국 주식 (N_kr, 5): [Open, High, Low, Close, Volume]
    x_kr = torch.tensor(kor_feat, dtype=torch.float, device=device)
    
    # (3) PC 노드 (4, 5): [PC값, 0, 0, 0, 0]
    pca_tensor = torch.tensor(pca_vals, dtype=torch.float, device=device).view(-1, 1)
    pca_pad = torch.zeros((num_pca, 4), device=device)
    x_pca = torch.cat([pca_tensor, pca_pad], dim=1)
    
    # (4) Regime 노드 (3, 5): One-hot [1, 0, 0, 0, 0]
    regime_tensor = torch.zeros((num_regime, 5), device=device)
    regime_tensor[regime, 0] = 1.0 
    x_regime = regime_tensor
    
    # [최종 Node Feature Matrix]
    # 순서: [US(Feat) | KR(Data) | PC | Regime]
    x = torch.cat([x_us, x_kr, x_pca, x_regime], dim=0)
    
    total_nodes = num_us_nodes + num_kr + num_pca + num_regime
    
    # -------------------------------------------------------
    # 2. Edge 구성 (동일)
    # -------------------------------------------------------
    edge_indices = []
    edge_attrs = []
    
    # (1) Stock-Stock (US & KR) Covariance Edge
    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float, device=device)
    mask = torch.abs(cov_tensor) > 0.0
    stock_edge_idx = mask.nonzero().t()
    stock_edge_attr = cov_tensor[mask]
    
    edge_indices.append(stock_edge_idx)
    edge_attrs.append(stock_edge_attr)
    
    # (2) Stock <-> PC Edge (모든 주식과 연결)
    stock_indices = torch.arange(num_us_nodes + num_kr, device=device)
    pca_indices = torch.arange(num_us_nodes + num_kr, num_us_nodes + num_kr + num_pca, device=device)
    
    s_grid, p_grid = torch.meshgrid(stock_indices, pca_indices, indexing='ij')
    sp_edges = torch.stack([s_grid.flatten(), p_grid.flatten()], dim=0)
    sp_attr = torch.ones(sp_edges.shape[1], device=device)
    
    edge_indices.append(sp_edges)
    edge_attrs.append(sp_attr)
    
    # (3) Stock <-> Regime Edge (모든 주식과 연결)
    regime_indices = torch.arange(total_nodes - num_regime, total_nodes, device=device)
    
    s_grid, r_grid = torch.meshgrid(stock_indices, regime_indices, indexing='ij')
    sr_edges = torch.stack([s_grid.flatten(), r_grid.flatten()], dim=0)
    sr_attr = torch.ones(sr_edges.shape[1], device=device)
    
    edge_indices.append(sr_edges)
    edge_attrs.append(sr_attr)
    
    # -------------------------------------------------------
    # 3. 데이터 합치기
    # -------------------------------------------------------
    final_edge_index = torch.cat(edge_indices, dim=1)
    final_edge_attr = torch.cat(edge_attrs, dim=0).view(-1, 1)
    
    # Mask 생성
    stock_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    stock_mask[num_us_nodes : num_us_nodes + num_kr] = True 
    
    data = Data(x=x, edge_index=final_edge_index, edge_attr=final_edge_attr)
    data.stock_mask = stock_mask
    
    return data
def train():
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Start on: {device}")
    
    # 1. 데이터 로드 (process_data 클래스 사용)
    print("Loading Data...")
    loader = process_data() # DB 연결 및 클래스 인스턴스화
    
    us_df = loader.fetch_us()
    kospi_df = loader.fetch_kospi()
    kor_df = loader.fetch_kor()
    bs_df = loader.fetch_bs() # kfb (재무 데이터)
    bs_df = bs_df.loc[:, ~bs_df.columns.duplicated()]
    us_df['Date'] = pd.to_datetime(us_df['Date'])
    kospi_df['Date'] = pd.to_datetime(kospi_df['Date'])
    kor_df['Date'] = pd.to_datetime(kor_df['Date'])
    # bs_df(재무)도 필요하다면 변환
    if 'Date' in bs_df.columns:
        bs_df['Date'] = pd.to_datetime(bs_df['Date'])

    print("Data Loaded Successfully.")
    print(f"KR Stocks: {len(kor_df['Tick_id'].unique())} tickers")
    print(f"US Stocks: {len(us_df['Tick_id'].unique())} tickers")

    # 2. 모델 로드 (HMM, PCA, Cov)
    # [주의] pickle 파일 경로가 맞는지 확인하세요.
    hmm = HMM_Model('model_pickle/hmm_model.pkl')
    pca = PCA_Model('model_pickle/pca_model.pkl', 'model_pickle/scaler_model.pkl')
    cov = Cov_Model()
    
    # 3. 환경(Environment) 초기화
    env = Environment(
        time_window=20,        # 20일치 데이터를 봄
        budget=100_000_000,    # 초기 자본 1억
        kor=kor_df,
        us=us_df,
        kfb=bs_df,             # 재무 데이터
        kospi=kospi_df,
        pca_model=pca,
        hmm_model=hmm,
        cov=cov
    )
    
    # 4. PPO Agent & ActorCritic 모델 초기화
    # in_channels=5 (우리가 모든 노드 차원을 5로 맞췄으므로)
    model = ActorCritic(in_channels=5, hidden=64, heads=4).to(device)
    agent = PPOAgent(model, lr=0.0003, concentration=10.0, device=device)
    memory = Memory()
    
    # 5. 하이퍼파라미터 설정
    max_episodes = 500
    update_timestep = 200 # 200 step마다 PPO 업데이트
    timestep = 0
    
    # =====================================================
    # 학습 시작
    # =====================================================
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        while True:
            timestep += 1
            
            # (1) 데이터 변환 (Env State -> Graph Data)
            num_us = env.us_ticker_size 
            graph_data = observation_to_graph(state, num_us, device=device)
            
            # (2) Action 선택 (PPO Agent)
            # 학습 중에는 탐험(Dirichlet)을 위해 stochastic하게 선택
            action, log_prob, value = agent.select_action(graph_data)
            
            # (3) 환경 진행 (Step)
            next_state, reward, done, info = env.step(action)
            
            # (4) 메모리에 저장 (PPO 업데이트용)
            # GPU 메모리를 아끼기 위해 CPU로 내려서 저장
            memory.states.append(graph_data.to('cpu')) 
            memory.actions.append(action)
            memory.log_probs.append(log_prob)
            memory.values.append(value)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            
            # (5) PPO 업데이트 (일정 step마다 수행)
            if timestep % update_timestep == 0:
                print(f" >> [Update] PPO Model Update at timestep {timestep}")
                agent.update(memory)
                memory.clear()
                
            if done:
                break
        
        # 에피소드 종료 후 로그 출력
        # info 딕셔너리에 'reason' 등이 있다면 같이 출력 가능
        print(f"Episode {episode}/{max_episodes} | Reward: {episode_reward:.2f} | PF Value: {env.portfolio_value:.0f}")
        
        # 모델 저장 (10 에피소드마다)
        if episode % 10 == 0:
            save_path = f"ppo_gat_ep{episode}.pth"
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()