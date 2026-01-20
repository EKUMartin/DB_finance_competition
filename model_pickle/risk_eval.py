import numpy as np
class risk_eval:
    def __init__(self,us,kr):
        self.us=us
        self.kr=kr
    def eval_cov(self):
        cov=[]
        for rows in self.kr:
            kr_cov=[]
            for u_row in self.us:
                val=np.cov(rows,u_row)[0,1]
                kr_cov.append(val)
            cov.append(kr_cov)
        return cov
    def eval_corr(self):
        corr=[]
        for rows in self.kr:
            kr_corr=[]
            for u_row in self.us:
                val=np.corrcoef(rows,u_row)[0,1]
                kr_corr.append(val)
            corr.append(kr_corr)
        return corr