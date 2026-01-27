import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
# from torch_geometric.data import Batch  <-- 이거 필요 없음

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, concentration=10.0, ent_coef=0.01, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ent_coef = ent_coef
        self.concentration = concentration
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            weights, value, _ = self.model(data)
            
        alpha = (weights * self.concentration) + 1e-8
        dist = Dirichlet(alpha)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.cpu().numpy(), log_prob.item(), value.item()

    def update(self, memory):
        """
        [복원됨] Loop 방식으로 하나씩 처리 (속도는 느리지만 로직은 단순)
        """
        # 1. 텐서 변환 (CPU -> GPU)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(memory.values, dtype=torch.float32).to(self.device)
        
        try:
            actions = torch.stack(memory.actions).to(self.device)
        except:
            actions = torch.tensor(memory.actions, dtype=torch.float32).to(self.device)

        # 2. Monte Carlo Estimate of Returns
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 3. K-Epochs 만큼 업데이트
        for _ in range(self.k_epochs):
            
            # -------------------------------------------------------
            # [Loop 방식 복원] 데이터를 하나씩 꺼내서 리스트에 담음
            # -------------------------------------------------------
            new_log_probs_list = []
            new_values_list = []
            entropy_list = []
            
            # memory.states에 있는 그래프 데이터를 하나씩 순회
            for i, data in enumerate(memory.states):
                data = data.to(self.device)
                
                # 모델 Forward (하나씩 들어감 -> if result < 0.3 에러 안 남)
                weights, value, _ = self.model(data)
                
                # 분포 생성
                alpha = (weights * self.concentration) + 1e-8
                dist = Dirichlet(alpha)
                
                # 현재 Action에 대한 Log Prob 재계산
                # actions[i]는 해당 시점의 행동
                cur_action = actions[i]
                new_log_prob = dist.log_prob(cur_action)
                dist_entropy = dist.entropy()
                
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(value)
                entropy_list.append(dist_entropy)
            
            # 리스트를 다시 텐서로 변환 (Stack)
            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list).squeeze() # (Batch, 1) -> (Batch, )
            dist_entropy = torch.stack(entropy_list).mean()     # 전체 엔트로피 평균
            
            # -------------------------------------------------------
            # 4. Ratio & Surrogate Loss (PPO 계산)
            # -------------------------------------------------------
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            advantages = returns - new_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(new_values, returns) - self.ent_coef * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = [] 
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]