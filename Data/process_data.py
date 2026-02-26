from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DB금융공모전
sys.path.insert(0, str(PROJECT_ROOT))
import DB.db_conn as db
import pandas as pd

class process_data:
    def __init__(self):
        self.conn,self.cur=db.open_db()
    def fetch_us(self):
        sql_query="select Tick_id, Date, Adj_close, Volume from uDaytrading where Tick_id in(505,506,507,508,509,510,511,513,514) and year(Date)>=2015 and year(Date)<=2023;"
        return pd.read_sql(sql_query, self.conn)

    def fetch_kospi(self):
        sql_query="select Open,High,Low,Close,Volume,`Change`,Date from kKospi where year(Date)>=2015 and year(Date)<=2023;"
        return pd.read_sql(sql_query, self.conn)
    def fetch_kor(self):
        # sql_query="select * from kDaytradingAdj where Tick_id in (831,819,142,916,459,110,903,897,527,259,156,101,143,170,616,804,237,918,108,96,162,909) and year(Date)>=2015 and year(Date)<=2023 and Date IN ( SELECT Date FROM kDaytradingAdj WHERE Tick_id IN (831,819,142,916,459,110,903,897,527,259,156,101,143,170,616,804,237,918,108,96,162,909) AND Date >= '2015-01-01' AND Volume > 0 GROUP BY Date HAVING COUNT(Tick_id) = 22);"
        sql_query="select * from kDaytradingAdj where Tick_id in (831,142,459,903,527,156,143,616,237,108,162) and year(Date)>=2015 and year(Date)<=2023 and Date IN ( SELECT Date FROM kDaytradingAdj WHERE Tick_id IN (831,142,459,903,527,156,143,616,237,108,162) AND Date >= '2015-01-01' AND Volume > 0 GROUP BY Date HAVING COUNT(Tick_id) = 11);"
        return pd.read_sql(sql_query, self.conn)

    def fetch_bs(self):
        # sql_query="select * from kFinancials as F left join kBalance_sheet as B on F.Tick_id=B.Tick_id AND Year(F.Date) = Year(B.Date)  where F.Tick_id in (831,819,142,916,459,110,903,897,527,259,156,101,143,170,616,804,237,918,108,96,162,909) and Year(F.Date)>=2014 and Year(F.Date)<=2022;"
        sql_query="select * from kFinancials as F left join kBalance_sheet as B on F.Tick_id=B.Tick_id AND Year(F.Date) = Year(B.Date)  where F.Tick_id in (831,142,459,903,527,156,143,616,237,108,162) and Year(F.Date)>=2014 and Year(F.Date)<=2022;"
        return pd.read_sql(sql_query, self.conn)