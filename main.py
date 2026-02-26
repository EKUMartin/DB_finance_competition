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
import os
def observation_to_graph(observation, num_us_nodes, device='cuda'):

    regime = observation['regime']
    pca_vals = observation['pca']        
    cov_matrix = observation['cov']        
    kor_feat = np.array(observation['kor_feat'])
    us_feat = np.array(observation['us_feat'])   
    
    num_kr = len(kor_feat)
    num_regime = 3          

    us_tensor = torch.tensor(us_feat, dtype=torch.float, device=device)
    us_pad = torch.zeros((num_us_nodes, 4), device=device) 
    x_us = torch.cat([us_tensor, us_pad], dim=1)

    kr_tensor = torch.tensor(kor_feat, dtype=torch.float, device=device)
    pca_tensor = torch.tensor(pca_vals, dtype=torch.float, device=device)
    x_kr = torch.cat([kr_tensor, pca_tensor], dim=1) 

    regime_tensor = torch.zeros((num_regime, 6), device=device) 
    regime_tensor[regime, 0] = 1.0 
    x_regime = regime_tensor

    x = torch.cat([x_us, x_kr, x_regime], dim=0)

    total_nodes = num_us_nodes + num_kr + num_regime

    # 엣지 연결 시작
    edge_indices = []
    edge_attrs = []

    cov_tensor = torch.tensor(cov_matrix, dtype=torch.float, device=device)
    mask = torch.abs(cov_tensor) > 0.0
    stock_edge_idx = mask.nonzero().t()
    stock_edge_attr = cov_tensor[mask]
    
    edge_indices.append(stock_edge_idx)
    edge_attrs.append(stock_edge_attr)


    kr_indices = torch.arange(num_us_nodes, num_us_nodes + num_kr, device=device)
    regime_indices = torch.arange(total_nodes - num_regime, total_nodes, device=device)
    
    s_grid, r_grid = torch.meshgrid(kr_indices, regime_indices, indexing='ij')
    sr_edges = torch.stack([s_grid.flatten(), r_grid.flatten()], dim=0)
    sr_attr = torch.ones(sr_edges.shape[1], device=device)
    
    edge_indices.append(sr_edges)
    edge_attrs.append(sr_attr)

    current_weights = observation.get('weights', np.zeros(total_nodes - num_regime + 1))#환경에서 주식 비중 가져오기, +1은 현금 노드
    
    # PCA 값을 1차원 한 줄로 쫙 펴서 요약본에 넣음
    pca_tensor_flat = pca_tensor.mean(dim=0)
    regime_onehot = torch.zeros(3, device=device)
    regime_onehot[regime] = 1.0   # 해당하는 regime만 1로
    stock_weights_np = current_weights[1:]
    
    if len(stock_weights_np) != len(cov_matrix):
        current_risk = 0.0
    else:
        current_risk = np.dot(stock_weights_np.T, np.dot(cov_matrix, stock_weights_np)) 
    risk_tensor = torch.tensor([current_risk], dtype=torch.float, device=device)
    
    # 요약본 완성
    gate_input = torch.cat([pca_tensor_flat, regime_onehot, risk_tensor], dim=0)

    # 최종 PyTorch Geometric Data 생성
    final_edge_index = torch.cat(edge_indices, dim=1)
    final_edge_attr = torch.cat(edge_attrs, dim=0).view(-1, 1)
    
    stock_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    stock_mask[num_us_nodes : num_us_nodes + num_kr] = True #한국 주식만 True로 마스크를 줌
    
    data = Data(x=x, edge_index=final_edge_index, edge_attr=final_edge_attr)
    data.stock_mask = stock_mask
    data.old_weight = torch.tensor(current_weights, dtype=torch.float, device=device)#과거 비중을 이용해서 여러가지 계산함->포트폴리오 리스크, 수수료 등등
    data.input_data = gate_input.unsqueeze(0) 
    
    return data

def train():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = process_data()
    
    us_df = loader.fetch_us()
    kospi_df = loader.fetch_kospi()
    kor_df = loader.fetch_kor()
    bs_df = loader.fetch_bs()

    bs_df = bs_df.loc[:, ~bs_df.columns.duplicated()]
    us_df['Date'] = pd.to_datetime(us_df['Date'])
    kospi_df['Date'] = pd.to_datetime(kospi_df['Date'])
    kor_df['Date'] = pd.to_datetime(kor_df['Date'])
    if 'Date' in bs_df.columns:
        bs_df['Date'] = pd.to_datetime(bs_df['Date'])

    hmm = HMM_Model('model_pickle/hmm_model.pkl')
    pca = PCA_Model('model_pickle/pca_model.pkl', 'model_pickle/scaler_model.pkl')
    cov = Cov_Model()
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

    model = ActorCritic(in_channels=6, hidden=64, heads=4, input_size=8).to(device)
    agent = PPOAgent(model, lr=1e-4 , device=device,ent_coef=0.0001,gamma=0.99,gae_lambda=0.95,eps_clip=0.2,k_epochs=4)
    memory = Memory()
    max_episodes = 4000
    update_timestep = 512
    timestep = 0
    history = {
            'episode': [],
            'reward': [], 
            'portfolio_value': [], 
            'loss': [],
            'pg_loss': [],
            'v_loss': [],
            'ent_loss': [],
            'approx_kl': [],
            'clip_frac': []
        }
    if os.path.exists('training_log.csv'):
        old_df = pd.read_csv('training_log.csv')
    else:
        old_df = pd.DataFrame()
    for episode in range(1, max_episodes + 1):
        env.current_episode = episode
        state = env.reset()
        episode_reward = 0

        ep_metrics = {'loss': [], 'pg': [], 'v': [], 'ent': [], 'kl': [], 'clip': []}
        
        while True:
            timestep += 1
            num_us = env.us_ticker_size 
            graph_data = observation_to_graph(state, num_us, device=device)
            
            action, log_prob, value = agent.select_action(graph_data)
            if action.ndim == 2:
                action = action.flatten()
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
                if len(memory.states) <= 1:
                    continue
                print(f" >> [Update] PPO Model Update at timestep {timestep}")
                next_graph_data = observation_to_graph(next_state, num_us, device=device)
                with torch.no_grad():
                    _, _, next_val = agent.select_action(next_graph_data)
                metrics=agent.update(memory, next_value=next_val)
                memory.clear()
                ep_metrics['loss'].append(metrics['loss'])
                ep_metrics['pg'].append(metrics['pg_loss'])
                ep_metrics['v'].append(metrics['v_loss'])
                ep_metrics['ent'].append(metrics['ent_loss'])
                ep_metrics['kl'].append(metrics['approx_kl'])
                ep_metrics['clip'].append(metrics['clip_frac'])
            if done:
                break
        if len(memory.states) > 1:
            metrics=agent.update(memory, next_value=0)
            memory.clear()
            ep_metrics['loss'].append(metrics['loss'])
            ep_metrics['pg'].append(metrics['pg_loss'])
            ep_metrics['v'].append(metrics['v_loss'])
            ep_metrics['ent'].append(metrics['ent_loss'])
            ep_metrics['kl'].append(metrics['approx_kl'])
            ep_metrics['clip'].append(metrics['clip_frac'])
        else:
            memory.clear()
        current_episode=episode+516
        history['episode'].append(current_episode)
        history['reward'].append(episode_reward)
        history['portfolio_value'].append(env.portfolio_value)

        if len(ep_metrics['loss']) > 0:
            history['loss'].append(np.mean(ep_metrics['loss']))
            history['pg_loss'].append(np.mean(ep_metrics['pg']))
            history['v_loss'].append(np.mean(ep_metrics['v']))
            history['ent_loss'].append(np.mean(ep_metrics['ent']))
            history['approx_kl'].append(np.mean(ep_metrics['kl']))
            history['clip_frac'].append(np.mean(ep_metrics['clip']))
            
            current_loss = np.mean(ep_metrics['loss'])
        else:
            history['loss'].append(0.0) 
            history['pg_loss'].append(0.0)
            history['v_loss'].append(0.0)
            history['ent_loss'].append(0.0)
            history['approx_kl'].append(0.0)
            history['clip_frac'].append(0.0)
            current_loss = 0.0

        print(f"Ep {episode}/{max_episodes} | R: {episode_reward:.2f} | Val: {env.portfolio_value:,.0f} | Loss: {current_loss:.4f}")
        
        if episode % 5 == 0:
            current_df = pd.DataFrame(history)
            if not old_df.empty:
                combined_df = pd.concat([old_df, current_df], ignore_index=True)
            else:
                combined_df = current_df
            combined_df.to_csv('training_log.csv', index=False)
        if episode_reward > 0:
            torch.save(agent.model.state_dict(), f"ppo_gat_ep{current_episode}_reward_{episode_reward}.pth")
        if episode % 1000 == 0:
            torch.save(agent.model.state_dict(), f"ppo_gat_ep{current_episode}.pth")

    # =====================================================
    # [최종 저장] CSV 및 상세 그래프
    # =====================================================
    print("Training Finished. Saving data and graphs...")
    
    # 1. CSV 저장 (최종본)
    df = pd.DataFrame(history)
    if not old_df.empty:
        final_df = pd.concat([old_df, current_df], ignore_index=True)
    else:
        final_df = current_df
    final_df.to_csv('training_log.csv', index=False)
    print(" >> Data saved to 'training_log.csv'")


    print("Training Finished. Saving results graph...")
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    # 1. Total Reward
    axes[0, 0].plot(history['reward'], color='blue')
    axes[0, 0].set_title("Total Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].grid(True)
    
    # 2. Portfolio Value
    axes[0, 1].plot(history['portfolio_value'], color='orange')
    axes[0, 1].axhline(y=100000000, color='red', linestyle='--')
    axes[0, 1].set_title("Portfolio Value")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(True)
    
    # 3. Total Loss
    axes[0, 2].plot(history['loss'], color='black')
    axes[0, 2].set_title("Total Loss")
    axes[0, 2].set_xlabel("Update Step")
    axes[0, 2].grid(True)

    # 4. Approximate KL (중요!)
    # 이게 계속 커지면(0.05 이상) Policy가 너무 급격히 변한다는 뜻 -> lr 줄이거나 clip range 조절
    axes[0, 3].plot(history['approx_kl'], color='green')
    axes[0, 3].set_title("Approx KL (Policy Change)")
    axes[0, 3].set_xlabel("Update Step")
    axes[0, 3].grid(True)

    # 5. Clip Fraction
    # 0.5가 넘어가면 데이터의 절반 이상이 잘려나갔다는 뜻 -> clip_range를 키우거나 lr 줄여야 함
    axes[1, 0].plot(history['clip_frac'], color='purple')
    axes[1, 0].set_title("Clip Fraction")
    axes[1, 0].set_xlabel("Update Step")
    axes[1, 0].grid(True)

    # 6. Policy Gradient Loss
    # 이게 음수로 계속 내려가야 Policy가 좋아지는 것 (Objective Maximize = Loss Minimize)
    axes[1, 1].plot(history['pg_loss'], color='teal')
    axes[1, 1].set_title("Policy Gradient Loss")
    axes[1, 1].set_xlabel("Update Step")
    axes[1, 1].grid(True)

    # 7. Value Loss
    # 0으로 수렴해야 Critic이 예측을 잘하는 것
    axes[1, 2].plot(history['v_loss'], color='brown')
    axes[1, 2].set_title("Value Loss (Critic MSE)")
    axes[1, 2].set_xlabel("Update Step")
    axes[1, 2].grid(True) # Log Scale 권장: axes[1, 2].set_yscale('log')

    # 8. Entropy Loss (Negative Entropy)
    # 초반엔 높다가(랜덤) 서서히 낮아져야 함(확신). 너무 빨리 낮아지면 조기 수렴.
    axes[1, 3].plot(history['ent_loss'], color='magenta')
    axes[1, 3].set_title("Entropy Loss (-Entropy)")
    axes[1, 3].set_xlabel("Update Step")
    axes[1, 3].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_result_detailed.png')
    print("Graph saved as 'training_result_detailed.png'.")
    plt.show()

if __name__ == '__main__':
    train()