import pandas as pd
class buy_and_hold:
    def __init__(self,initial_capital):
        self.initial_capital=initial_capital
        self.equity=None
        self.performance={}
    def backtest(self,price_series):
        if not isinstance(price_series,pd.series):
            price_series=pd.series(price_series)
        daily_returns=price_series.pct_change().fillna(0)
        self.equity=self.initial_capital*(1+daily_returns).cumprod()
        self._evaluate()
        return self.equity, self.performance
    def _evaluate(self):
        if self.equity is None:
            return
        final_balance=self.equity.iloc[-1]
        roi=(final_balance / self.initial_capital - 1) * 100
        peak = self.equity.cummax()
        drawdown = (self.equity - peak) / peak
        mdd = drawdown.min() * 100
        self.performance={
            'Initial Capital': self.initial_capital,
            'Final Balance': final_balance,
            'ROI': roi,
            'MDD': mdd
        }