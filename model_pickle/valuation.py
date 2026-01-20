from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DB금융공모전
sys.path.insert(0, str(PROJECT_ROOT))
import DB.db_conn as db
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Valuation:
    def __init__(self,Netincome,Totalequity,Totalassets,Operatingincome,Revenue,Totalliabilities,Currentassets,Currentliabilities,Pretaxincome,Retainedearnings,Noncurrentliabilities,Noncurrentassets):
        self.Netincome=Netincome
        self.Totalequity=Totalequity
        self.Totalassets=Totalassets
        self.Operatingincome=Operatingincome
        self.Revenue=Revenue
        self.Totalliabilities=Totalliabilities
        self.Currentassets=Currentassets
        self.Currentliabilities=Currentliabilities
        self.Pretaxincome=Pretaxincome
        self.Retainedearnings=Retainedearnings
        self.Noncurrentliabilities=Noncurrentliabilities
        self.Noncurrentassets=Noncurrentassets
    def calculate_roe(self):
        roe=self.Netincome/self.Totalequity
        return roe
    def calculate_roa(self):
        roa=self.Netincome/self.Totalassets
        return roa
    def calculate_OPM(self):
        opm=self.Operatingincome / self.Revenue
        return opm
    def calculate_npm(self):
        npm=self.Netincome / self.Revenue
        return npm
    def calculate_dr(self):
        debt_ratio=self.Totalliabilities / self.Totalequity
        return debt_ratio
    def calculate_cr(self):
        current_ratio=self.Currentassets / self.Currentliabilities
        return current_ratio 
    def calculate_lr(self):
        liability_ratio=self.Totalliabilities / self.Totalassets
        return liability_ratio
    def calculate_at(self):
        asset_turnover=self.Revenue / self.Totalassets
        return asset_turnover
    def calculate_et(self):
        equity_turnover=self.Revenue / self.Totalequity
        return equity_turnover
    def calculate_of(self):
        operating_focus=self.Operatingincome/self.Pretaxincome
        return operating_focus
    def calculate_tb(self):
        tax_burden=self.Netincome/self.Pretaxincome
        return tax_burden 
    def calculate_wc(self):
        working_capital_on_assets=(self.Currentassets-self.Currentliabilities)/self.Totalassets
        return working_capital_on_assets
    def calculate_re(self):
        retained_earnings_to_assets=self.Retainedearnings/self.Totalassets
        return retained_earnings_to_assets
    def calculate_nc(self):
        non_current_fit_ratio=self.Noncurrentassets/(self.Totalequity+self.Noncurrentliabilities)
        return non_current_fit_ratio
    def calculate_cl(self):
        current_liability_ratio=self.Currentliabilities/self.Totalliabilities
        return current_liability_ratio
    def calculate_em(self):
        equity_multiplier=self.Totalassets/self.Totalequity
        return equity_multiplier

def fetch_column(cur):
    select_query = f"""SELECT 
            F.Tick_id,
            IFNULL(F.NetIncome, 0) as Netincome,
            IFNULL(B.Totalequity, 0) as Totalequity,
            IFNULL(B.Totalassets, 0) as Totalassets,
            IFNULL(F.Operatingincome, 0) as Operatingincome,
            IFNULL(F.Revenue, 0) as Revenue,
            IFNULL(B.Totalliabilities, 0) as Totalliabilities,
            IFNULL(B.Currentassets, 0) as Currentassets,
            IFNULL(B.Currentliabilities, 0) as Currentliabilities,
            IFNULL(F.PretaxIncome, 0) as PretaxIncome,
            IFNULL(B.Retainedearnings, 0) as Retainedearnings,
            IFNULL(B.Noncurrentliabilities, 0) as Noncurrentliabilities,
            IFNULL(B.Noncurrentassets, 0) as Noncurrentassets
        FROM kFinancials AS F
        LEFT JOIN kBalance_sheet AS B 
            ON F.Tick_id = B.Tick_id AND Year(B.Date) = 2014
        WHERE Year(F.Date) = 2014
        ORDER BY F.Tick_id;"""
    cur.execute(select_query)
    result = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(result,columns=cols)

conn,cur=db.open_db()
pca = PCA(n_components=5)
df=fetch_column(cur)
valuation = Valuation(
    Netincome=df['Netincome'].values,
    Totalequity=df['Totalequity'].values,
    Totalassets=df['Totalassets'].values,
    Operatingincome=df['Operatingincome'].values,
    Revenue=df['Revenue'].values,
    Totalliabilities=df['Totalliabilities'].values,
    Currentassets=df['Currentassets'].values,
    Currentliabilities=df['Currentliabilities'].values,
    Pretaxincome=df['PretaxIncome'].values,
    Retainedearnings=df['Retainedearnings'].values,
    Noncurrentliabilities=df['Noncurrentliabilities'].values,
    Noncurrentassets=df['Noncurrentassets'].values
)
roe=valuation.calculate_roe()
at=valuation.calculate_at()
cl=valuation.calculate_cl()
cr=valuation.calculate_cr()
dr=valuation.calculate_dr()
em=valuation.calculate_em()
et=valuation.calculate_et()
lr=valuation.calculate_lr()
nc=valuation.calculate_nc()
npm=valuation.calculate_npm()
of=valuation.calculate_of()
opm=valuation.calculate_OPM()
re=valuation.calculate_re()
wc=valuation.calculate_wc()
roa=valuation.calculate_roa()
tb=valuation.calculate_tb()
x=np.column_stack((at,cl,cr,dr,em,et,lr,nc,npm,of,opm,re,wc,roa,tb))
feature_names = [
    "Asset Turnover (활동성)", 
    "Curr Liab Ratio (부채구조)", 
    "Current Ratio (유동성)", 
    "Debt Ratio (부채비율)", 
    "Equity Multiplier (레버리지)", 
    "Equity Turnover (자본회전)", 
    "Liab Ratio (부채의존)", 
    "Non-Current Fit (장기적합)", 
    "NPM (순이익률)", 
    "Operating Focus (영업집중)", 
    "OPM (영업이익률)", 
    "Retained Earnings (유보율)", 
    "Working Capital (운전자본)", 
    "ROA (총자산이익률)", 
    "Tax Burden (세금부담)"
]

features = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.nan_to_num(features_scaled, nan=0.0)
principalComponents = pca.fit_transform(features_scaled)
principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component1', 'principal component2',"principal component3","principal component4","principal component5"])
loadings = pd.DataFrame(pca.components_, columns=feature_names)
print(loadings.T)
plt.figure(figsize=(12, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('PCA Loadings: What does each PC represent?')
plt.show()
# print(principalDf)
# print(pca.explained_variance_ratio_)
