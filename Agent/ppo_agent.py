import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, concentration=10.0, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.concentration = concentration # 탐험 강도 (클수록 모델 믿음, 작을수록 무작위)
        self.device = device
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, data):
        """
        데이터(Graph)를 받아 Action(비중)을 반환
        Training 중에는 Dirichlet 분포로 탐험 수행
        """
        data = data.to(self.device)
        with torch.no_grad():
            weights, value, _ = self.model(data)
            
        # [탐험 전략] Dirichlet 분포 사용
        # 모델이 뱉은 weights에 concentration을 곱해 alpha로 사용
        # epsilon 추가: 0이 되면 에러나므로 방지
        alpha = (weights * self.concentration) + 1e-8
        dist = Dirichlet(alpha)
        
        # Action 샘플링 (비중 합 1 자동 만족)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.cpu().numpy(), log_prob.item(), value.item()

    def update(self, memory):
        """
        메모리에 저장된 데이터를 바탕으로 모델 업데이트
        """
        # 1. 데이터 변환 (List -> Tensor)
        # PyG 데이터는 Batch로 처리하기 복잡하므로, 여기서는 간단하게 반복문이나 List 처리
        # (그래프 데이터는 구조가 같으므로 배치 처리가 가능하지만, 설명 복잡도를 낮추기 위해 루프 사용 가능)
        # 하지만 효율성을 위해 state(Graph Data)는 재계산하거나 유지해야 함.
        # 여기서는 간단히 Value와 LogProb만 가지고 업데이트하는 방식 사용 (Graph는 다시 forward)
        
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(memory.values, dtype=torch.float32).to(self.device)
        actions = torch.tensor(memory.actions, dtype=torch.float32).to(self.device)
        
        # 2. Monte Carlo Estimate of Returns (보상 계산)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalizing the returns (학습 안정성)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # 3. K-Epochs 만큼 업데이트
        for _ in range(self.k_epochs):
            # 현재 상태에 대한 평가 (Batch로 묶어서 다시 forward 해야 함)
            # 메모리에 저장된 state(Graph Data) 리스트를 하나씩 꺼내서 계산
            
            # [주의] 그래프 데이터 배치는 torch_geometric.data.Batch를 써야 함
            # 여기서는 코드를 간단히 하기 위해 루프를 돌지만, 실제론 느릴 수 있음
            new_log_probs_list = []
            new_values_list = []
            entropy_list = []
            
            for i, data in enumerate(memory.states):
                data = data.to(self.device)
                weights, value, _ = self.model(data)
                
                alpha = (weights * self.concentration) + 1e-8
                dist = Dirichlet(alpha)
                
                # 저장했던 action에 대한 현재 확률 계산
                cur_action = actions[i]
                new_log_prob = dist.log_prob(cur_action)
                dist_entropy = dist.entropy()
                
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(value.squeeze())
                entropy_list.append(dist_entropy)
            
            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            dist_entropy = torch.stack(entropy_list).mean()
            
            # 4. Ratio & Surrogate Loss (PPO 핵심)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            advantages = returns - new_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(new_values, returns) - 0.01 * dist_entropy
            
            # Backprop
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = [] # Graph Data 객체 저장
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