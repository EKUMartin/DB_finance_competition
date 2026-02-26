import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class ActorCritic(nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4, input_size=8):
        super(ActorCritic, self).__init__()

        self.gat1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=0.3, edge_dim=1)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=1, dropout=0.3, edge_dim=1)

        self.gate_fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )# linear layer- GAT와 동일한 입력값을 별도로 학습하는 레이어- GAT의 경우 어텐션으로 나왔기 때문에 기존의 데이터가 희석될 수 있어서 따로 추가로 전달

        # Actor (행동 결정: GAT 64 + Gate 16 = 80)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden + 16, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )# 각 종목별로 concat해서 나온 피처들을 처리
        
        # Critic
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )#actor의 행동을 평가하는 용도
        self.cash_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)# batch(그래프)의 인덱스 정보
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1 #num_graph=배치 단위

        gate_input = data.input_data #input data를 입력
        if gate_input.dim() == 1:
            gate_input = gate_input.unsqueeze(0)#파이토치는 2차원을 받아야 하기 때문에 차원 추가

        x, (edge_index1, alpha1) = self.gat1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )#x:노드 특성 , edge_index:노드 링크
        x = F.elu(x)
        x, (edge_index2, alpha2) = self.gat2(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        x = F.elu(x) 

        stock_x = x[data.stock_mask] #포트폴리오 주식에 해당하는 주식 여부에 대한 bool 타입 리스트로 주식 node 식별
        stock_batch = batch[data.stock_mask]# 각 포트폴리오가 몇번째 배치인지도 식별

        gate_feat = self.gate_fc(gate_input) #그래프를 만들 때 사용한 PCA, Regime,현재 포트폴리오 분산
        
        gate_feat_expanded = gate_feat[stock_batch] #배치 단위에 맞게 인덱싱
        

        combined_feat = torch.cat([stock_x, gate_feat_expanded], dim=1) # (Num_Stocks, 80) ; 거래할 주식+현재 상태에 대해서 뽑은 피처 concat
        
        action_logits = self.actor_fc(combined_feat) # (Num_Stocks, 1) # 모든 종목에 대해서 logit을 출력
        action_logits = action_logits.view(num_graphs, -1)#view: shape만 변경하는 함수, reshape과 유사
        cash_logits = self.cash_bias.expand(num_graphs, 1)#기존의 배치수 만큼 cash_bias 파마리터를 추가->학습에 사용
        action_logits = torch.cat([action_logits, cash_logits], dim=-1)


        global_feat = global_mean_pool(stock_x, stock_batch) # (Batch_Size, 64),전체 배치가 아닌 배치 단위에서 전체의 평균
        global_combined = torch.cat([global_feat, gate_feat], dim=1) # (Batch_Size, 80)
        value = self.critic_fc(global_combined) # (Batch_Size, 1); 각 배치에 대해서 평가 점수
        

        action_logits = action_logits.view(num_graphs, -1)#cash+주식수에 맞게 reshape해주고 출력
        attentions = {
            'layer1': (edge_index1, alpha1),
            'layer2': (edge_index2, alpha2)
        }
        return action_logits, value, attentions