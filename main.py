from Data.process_data import process_data
from model_pickle.models import HMM_Model, PCA_Model, Cov_Model
from Env.trade_env import Environment
from Policy.opt_policy import ActorCritic
from Agent.ppo_agent import PPOAgent, Memory
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
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
    # 1. 기본 정보 가져오기
    regime = observation['regime']
    pca_vals = observation['pca'] # (4,)
    cov_matrix = observation['cov'] # (N_stocks, N_stocks)
    
    # Env에서 현재 비중(weights)을 가져옵니다. (Cash 포함)
    # 예: [Cash, Stock1, Stock2, ...] 형태라고 가정
    current_weights = observation.get('weights', np.zeros(total_nodes - num_pca - num_regime + 1))
    
    # 2. 텐서 변환
    # (1) PCA (4개)
    pca_tensor_flat = torch.tensor(pca_vals, dtype=torch.float, device=device)
    
    # (2) Regime (3개, One-hot)
    regime_onehot = torch.zeros(3, device=device)
    regime_onehot[regime] = 1.0
    
    # (3) [Cov 반영] 현재 포트폴리오 리스크 계산 (1개)
    # weights에서 현금(보통 첫번째 혹은 마지막)을 제외하고 주식 비중만 가져와야 함
    # 여기서는 weights[1:]이 주식 비중이라고 가정 (Cash가 0번 인덱스일 경우)
    # 만약 weights 맨 뒤가 현금이면 weights[:-1] 사용. 
    # **중요: cov_matrix 크기와 stock_weights 크기가 같아야 함**
    
    stock_weights_np = current_weights[1:] # Cash 제외 (가정)
    
    # 혹시 크기가 안 맞으면 0으로 채우거나 자름 (안전장치)
    if len(stock_weights_np) != len(cov_matrix):
        # 크기 다르면 리스크 0으로 처리 (에러 방지)
        current_risk = 0.0
    else:
        # Risk = w^T * Cov * w
        # 결과는 스칼라 값 (내 포트폴리오의 분산)
        current_risk = np.dot(stock_weights_np.T, np.dot(cov_matrix, stock_weights_np))
        
    risk_tensor = torch.tensor([current_risk], dtype=torch.float, device=device)

    # 3. Gate 입력 벡터 합치기 (Size: 4 + 3 + 1 = 8)
    gate_input = torch.cat([pca_tensor_flat, regime_onehot, risk_tensor], dim=0)

    # -------------------------------------------------------
    # Data 객체에 할당
    # -------------------------------------------------------

    # 모델로 전달할 데이터


    final_edge_index = torch.cat(edge_indices, dim=1)
    final_edge_attr = torch.cat(edge_attrs, dim=0).view(-1, 1)
    
    # Mask 생성
    stock_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    stock_mask[num_us_nodes : num_us_nodes + num_kr] = True 
    
    data = Data(x=x, edge_index=final_edge_index, edge_attr=final_edge_attr)
    data.stock_mask = stock_mask
    data.old_weight = torch.tensor(current_weights, dtype=torch.float, device=device)
    data.input_data = gate_input.unsqueeze(0) # (1, 8) 배치 차원 추가
    return data
def train():
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Start on: {device}")
    
    # 1. 데이터 로드
    print("Loading Data...")
    loader = process_data()
    
    us_df = loader.fetch_us()
    kospi_df = loader.fetch_kospi()
    kor_df = loader.fetch_kor()
    bs_df = loader.fetch_bs()
    
    # 데이터 전처리 (중복 제거 & 날짜 변환)
    bs_df = bs_df.loc[:, ~bs_df.columns.duplicated()]
    us_df['Date'] = pd.to_datetime(us_df['Date'])
    kospi_df['Date'] = pd.to_datetime(kospi_df['Date'])
    kor_df['Date'] = pd.to_datetime(kor_df['Date'])
    if 'Date' in bs_df.columns:
        bs_df['Date'] = pd.to_datetime(bs_df['Date'])

    print("Data Loaded Successfully.")
    print(f"KR Stocks: {len(kor_df['Tick_id'].unique())} tickers")
    print(f"US Stocks: {len(us_df['Tick_id'].unique())} tickers")

    # 2. 모델 로드
    hmm = HMM_Model('model_pickle/hmm_model.pkl')
    pca = PCA_Model('model_pickle/pca_model.pkl', 'model_pickle/scaler_model.pkl')
    cov = Cov_Model()
    
    # 3. 환경 초기화
    env = Environment(
        time_window=20,
        budget=100_000_000,
        kor=kor_df,
        us=us_df,
        kfb=bs_df,
        kospi=kospi_df,
        pca_model=pca,
        hmm_model=hmm,
        cov=cov
    )
    
    # 4. Agent 초기화
    model = ActorCritic(in_channels=5, hidden=64, heads=4, input_size=8).to(device)
    agent = PPOAgent(model, lr=0.0003, concentration=10.0, device=device)
    memory = Memory()
    
    # 5. 하이퍼파라미터
    max_episodes = 100
    update_timestep = 200
    timestep = 0
    
    # [🔥 추가 1] 성능 기록용 리스트 생성
    history = {'reward': [], 'portfolio_value': []}
    
    # =====================================================
    # 학습 시작
    # =====================================================
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        agent.concentration = 10.0 + (episode * 0.2)
        while True:
            timestep += 1
            
            num_us = env.us_ticker_size 
            graph_data = observation_to_graph(state, num_us, device=device)
            
            action, log_prob, value = agent.select_action(graph_data)
            next_state, reward, done, info = env.step(action)
            
            memory.states.append(graph_data.to('cpu')) 
            memory.actions.append(action)
            memory.log_probs.append(log_prob)
            memory.values.append(value)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            
            if timestep % update_timestep == 0:
                print(f" >> [Update] PPO Model Update at timestep {timestep}")
                agent.update(memory)
                memory.clear()
                
            if done:
                break
        
        # [🔥 추가 2] 이번 에피소드 결과 저장
        history['reward'].append(episode_reward)
        history['portfolio_value'].append(env.portfolio_value)
        
        print(f"Episode {episode}/{max_episodes} | Reward: {episode_reward:.2f} | PF Value: {env.portfolio_value:.0f}")
        
        if episode % 10 == 0:
            save_path = f"ppo_gat_ep{episode}.pth"
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # =====================================================
    # [🔥 추가 3] 학습 종료 후 그래프 그리기 & 저장
    # =====================================================
    print("Training Finished. Saving results graph...")
    
    plt.figure(figsize=(12, 5))
    
    # 1. 보상(Reward) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['reward'], label='Total Reward', color='blue')
    plt.title("Training Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # 2. 포트폴리오 가치(Portfolio Value) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['portfolio_value'], label='Portfolio Value', color='orange')
    plt.axhline(y=100000000, color='red', linestyle='--', label='Initial Budget') # 원금선
    plt.title("Final Portfolio Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Value (KRW)")
    plt.grid(True)
    plt.legend()
    
    # 그래프 파일로 저장
    plt.savefig('training_result.png')
    print("Graph saved as 'training_result.png'. Check your folder!")
    plt.show() # 창으로도 띄우기

if __name__ == '__main__':
    train()