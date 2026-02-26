import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import torch.nn.functional as F
#open AI baseline 참고
class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, k_epochs=4, concentration=10.0, ent_coef=0.01, device='cpu', vf_coef=0.5, batch_size=128):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ent_coef = ent_coef
        self.concentration = concentration
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.val_norm = RunningMeanStd(shape=(1,))
        
    def select_action(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            weights, value, _ = self.model(data)
        alpha = F.softplus(weights) + 1# nan 에러 방지용 epsilon 추가
        alpha = torch.clamp(alpha, max=50.0)
        dist = Dirichlet(alpha)
        action = dist.sample()#다이클레 분포에서 뽑은 샘플들

        safe_action = torch.clamp(action, min=1e-4, max=1.0)#log prob이 무한대로 가지 않게 max와 min을 설정
        safe_action = safe_action / safe_action.sum(dim=-1, keepdim=True)#전체 비중 1로 맞추기(softplus와 동일)
        
        log_prob = dist.log_prob(safe_action)#action에 대한 확률분포가 필요하기 때문에 log prob을 이용
        # print(np.round(safe_action.cpu().numpy(), 3))
        return safe_action.cpu().numpy(), log_prob.item(), value.item()

    def update(self, memory, next_value=0):
        # 1. 데이터 정리
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(memory.values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(memory.is_terminals, dtype=torch.float32).to(self.device)
        
        try:
            actions = torch.stack(memory.actions).to(self.device)
        except:
            actions = torch.tensor(np.array(memory.actions), dtype=torch.float32).to(self.device)#넘파이로 입력되면 텐서로 바꾸기

        # GAE 계산
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):#미래->과거
            if t == len(rewards) - 1:#마지막
                next_val = next_value#만약 마지막이라면 외부에서 들어오는 next_Value를 사용
            else:
                next_val = old_values[t + 1]#아니라면 다음을 사용
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - old_values[t]#실제 보상+내일 보상-예상 했던 보상; 1-dones[t]:종료 됐냐, 안됐냐 종료가 안됐으면 미래가치를 사용
            #실제 보상이 예측과 얼마나 차이냐나= delta
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae#gae함수
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        returns = advantages + old_values
        returns_np = returns.cpu().numpy().reshape(-1, 1)
        self.val_norm.update(returns_np)#미래+예상에 대한 평균 분산 기억
        
        mean = torch.tensor(self.val_norm.mean, dtype=torch.float32).to(self.device)
        std = torch.sqrt(torch.tensor(self.val_norm.var, dtype=torch.float32)).to(self.device)
        
        #nomralization을 통해 값의 차이의 변동을 줄임
        returns_normalized = (returns - mean) / (std + 1e-5)#보상 normalization
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)#예상도 normalization

        # 3. 학습 루프
        batch_size = self.batch_size
        data_len = len(memory.states)
        indices = np.arange(data_len)
        
        avg_loss = 0; avg_pg = 0; avg_v = 0; avg_ent = 0; avg_kl = 0; avg_clip = 0
        update_count = 0

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, data_len, batch_size):
                end_idx = min(start_idx + batch_size, data_len)#batch_size만큼 남아 있는 지
                batch_indices = indices[start_idx:end_idx]
                curr_batch_size = len(batch_indices)

                self.optimizer.zero_grad()
                batch_loss = 0
                
                for idx in batch_indices:
                    state = memory.states[idx].to(self.device)
                    action = actions[idx]
                    old_log_prob = old_log_probs[idx]
                    advantage = advantages[idx]
                    ret_norm = returns_normalized[idx]
                    old_val = old_values[idx]

                    weights, value, _ = self.model(state)#업데이트된 정책망으로부터 과거 상태에 대한 비중을 받음
                    
                    # update 시에도 동일하게
                    alpha = F.softplus(weights) + 1
                    alpha = torch.clamp(alpha, max=50.0)
                    dist = Dirichlet(alpha)
                    
                    safe_action = torch.clamp(action, min=1e-4, max=1.0)
                    safe_action = safe_action / safe_action.sum(dim=-1, keepdim=True)
                    
                    new_log_prob = dist.log_prob(safe_action)#과거 action에 대한 확률분포를 받음
                    dist_entropy = dist.entropy()
                    #좋아지면 확률을 높이고, 안 좋아지면 확률을 낮춤 줄임
                    ratio = torch.exp(new_log_prob - old_log_prob)#과거 확률과 현재 확률 비율,
                    surr1 = ratio * advantage#평균보다 올라갔다면 +, 아니면 -
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage#surr1에 clamp를 적용
                    pg_loss = -torch.min(surr1, surr2)# 둘 중 더 낮을 걸 선택
                    
                    # Value Loss Clipping
                    value_pred_clipped = old_val + (value.squeeze() - old_val).clamp(-self.eps_clip, self.eps_clip)
                    
                    # 정규화된 ret_norm을 사용하여 거대한 스케일 문제 해결
                    v_loss1 = self.mse_loss(value.squeeze(), ret_norm)
                    v_loss2 = self.mse_loss(value_pred_clipped, ret_norm)
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2)
                    
                    ent_loss = -dist_entropy

                    loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss
                    loss = loss / curr_batch_size
                    loss.backward()
                    
                    batch_loss += loss.item() * curr_batch_size
                    
                    with torch.no_grad():
                         avg_pg += pg_loss.item()
                         avg_v += v_loss.item()
                         avg_ent += ent_loss.item()
                         avg_kl += ((ratio - 1) - torch.log(ratio)).item()
                         avg_clip += (torch.abs(ratio - 1.0) > self.eps_clip).float().item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                avg_loss += batch_loss
                update_count += curr_batch_size

        return {
            "loss": avg_loss / update_count,
            "pg_loss": avg_pg / update_count,
            "v_loss": avg_v / update_count,
            "ent_loss": avg_ent / update_count,
            "approx_kl": avg_kl / update_count,
            "clip_frac": avg_clip / update_count
        }

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