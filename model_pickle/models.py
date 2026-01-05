import joblib
import torch
import torch.nn as nn
import numpy as np

class Models:
    def __init__(self,model_1_path,model_2_path):
        self.model_1=joblib.load(model_1_path)
        self.model_2=joblib.load(model_2_path)
    def _2d_tensor(self,y):
        t=torch.tensor(y,dtype=torch.float32)
        if t.ndim==1:
            t=t.unsqueeze(1)
        return t
    def concat(self,data_1,data_2):
        m1_output=self.model_1.predict(data_1)
        m2_output=self.model_2.predict(data_2)
        m1_tensor=self._2d_tensor(m1_output)
        m2_tensor=self._2d_tensor(m2_output)
        return torch.hstack((m1_tensor,m2_tensor))
    def return_output(self,data_1,data_2):
        m1_output=self.model1.predict(data_1)
        m2_output=self.model2.predict(data_2)
        m1_tensor=self._2d_tensor(m1_output)
        m2_tensor=self._2d_tensor(m2_output)
        return m1_tensor,m2_tensor