from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool 

class ActorCritic(torch.nn.Module):
    def __init__(self, in_channels=5, hidden=32, heads=4, input_size=10):
        super().__init__()
        # Gate 관련 레이어 (self.linear, linear2, act1) -> 전부 삭제하세요.
        
        # 1. Shared Body
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.2, edge_dim=1)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, concat=False, dropout=0.2, edge_dim=1)
        
        # 2. Actor Head
        self.actor_head = torch.nn.Linear(hidden, 1)
        self.cash = torch.nn.Parameter(torch.zeros(1))

        # 3. Critic Head
        self.critic_head = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # GAT Layers
        x, (edge_index_alpha, alpha) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) 
        
        # Actor
        scores_all = self.actor_head(x).squeeze(-1)
        stock_scores = scores_all[data.stock_mask]
        final_scores = torch.cat([self.cash, stock_scores], dim=0)
        
        # [중요] Gate 로직 없이 그냥 Softmax 결과만 내보냅니다.
        # "Hold"를 하고 싶다면, 모델이 알아서 old_weight와 비슷한 weights를 내뱉도록 학습해야 합니다.
        weights = F.softmax(final_scores, dim=0) 

        # Critic
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embed = global_mean_pool(x, batch)
        value = self.critic_head(graph_embed)
        
        return weights, value, (edge_index_alpha, alpha)