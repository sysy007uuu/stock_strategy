from jqdatasdk import *
from jqdatasdk import finance
import pandas as pd
import numpy as np

cost_sold = 0.0013  # 卖出手续费比率
cost_buy = 0.0003  # 买入手续费比率
benjin = 1000000  # 本金为1000000
auth('13585176258', 'Wlc990108***')  # 登录joinquant
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列
delta_T = 7  # 默认是隔天收盘价卖出，如为2就是隔两天收盘价卖出

if __name__ == '__main__':
    df = pd.read_csv("etf/etf1.csv")
    q = query(finance.FUND_PORTFOLIO_STOCK).filter(finance.FUND_PORTFOLIO_STOCK.code ==
                                                   str(df.at[0, '基金代码'])).limit(10)
    df2 = finance.run_query(q)
    print(df2)
