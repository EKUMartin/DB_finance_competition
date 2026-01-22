from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool 

class ActorCritic(torch.nn.Module):
    def __init__(self, in_channels=5, hidden=32, heads=4):
        super().__init__()
        
        # -----------------------------------------------------------
        # 1. Shared Body (공유 레이어)
        # -----------------------------------------------------------
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.2, edge_dim=1)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, concat=False, dropout=0.2, edge_dim=1)

        # -----------------------------------------------------------
        # 2. Actor Head (Policy) -> 비중(Weights) 출력
        # -----------------------------------------------------------
        self.actor_head = torch.nn.Linear(hidden, 1)
        self.cash = torch.nn.Parameter(torch.zeros(1)) # 현금 점수 파라미터

        # -----------------------------------------------------------
        # 3. Critic Head (Value) -> 가치(Value) 출력
        # -----------------------------------------------------------
        self.critic_head = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # [Shared GAT]
        x, (edge_index_alpha, alpha) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) 
        
        # [Actor: Action 비중 계산]
        scores_all = self.actor_head(x).squeeze(-1)
        
        # 한국 주식 노드만 마스킹
        stock_scores = scores_all[data.stock_mask]
        
        # 현금 점수와 합치기
        final_scores = torch.cat([self.cash, stock_scores], dim=0)
        
        # [복구 완료] Softmax 적용 -> 합이 1인 비중(Weights) 출력
        weights = F.softmax(final_scores, dim=0) 

        # [Critic: State Value 계산]
        # 전체 그래프를 하나로 압축 (Global Pooling)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embed = global_mean_pool(x, batch)
        value = self.critic_head(graph_embed)

        return weights, value, (edge_index_alpha, alpha)