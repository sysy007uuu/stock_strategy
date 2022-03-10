import numpy as np
import pandas as pd
import os
from jqdatasdk import *
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import math

# 本回测默认为开盘价买入，收盘价卖出

# 全局变量初始化
cost_sold = 0.0013  # 卖出手续费比率
cost_buy = 0.0003  # 买入手续费比率
benjin = 1000000  # 本金为1000000
if_plt = True  # 是否图像化，默认为True
auth('13585176258', 'Wlc990108***')  # 登录joinquant
data_path = "D:\\WLC_Project\\Wlc's_Python_project\\stock_strategy\\data"
result_path = "D:\\WLC_Project\\Wlc's_Python_project\\stock_strategy\\result\\"

my_stock = dict()  # 存储当前的股票库，用键值对的方式存储，即股票代码-(股票数量,股票买入价，股票当日收盘价)
syl_list = []  # 收益率列表，用于记录每天累计的收益率
syl = 0.0
# 计算每天的收益率
def cal_syl_daily(x, y, money, date):
    # print(money)
    for key, values in my_stock.items():
        money += values[0] * values[2]  # 算出当日收盘后的总资金
    # print(money)
    # 计算收益率
    syl = (money - benjin) / benjin
    syl = round(syl * 100, 2)
    # print(date + "的收益率为" + str(syl) + '%')
    syl_list.append([date, str(syl) + '%'])

    x = np.append(x, date)
    y = np.append(y, syl)
    return x, y


# 计算最大回撤率
def cal_max_huiche(y, max_huiche):
    for i in range(len(y) - 1):
        min_price = min(y[i:])  # 计算当前时间点到最后的最小值
        huiche = (y[i] - min_price) / (1 + y[i] / 100)
        if huiche > max_huiche:
            max_huiche = huiche
    max_huiche = round(max_huiche, 2)
    return max_huiche


# 计算夏普比率
def cal_sharp(year_syl, z, n):
    # 无风险收益率为0.03
    syl_avg = np.average(z)
    sum2 = 0
    for i in range(len(z)):
        sum2 += math.pow(z[i] - syl_avg, 2)
    syl_std = math.sqrt(252 / n * sum2)
    syl_std = round(syl_std, 2)  # 这里顺带计算出了收益波动率
    print("收益波动率为" + str(syl_std) + '%')
    sharp = (year_syl - 3) / syl_std
    sharp = round(sharp, 2)
    return sharp, syl_std


# 绘制plot
def plot_syl(x, y, file):
    plt.figure(dpi=100)
    plt.plot(x, y, linewidth=2, marker='.')
    plt.xticks(x, x, rotation=60)
    # 如果数据过长可以加这个，每隔10个点显示一次
    plt.xticks(range(0, len(y), 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.xlabel('日期')
    plt.ylabel('收益率%')
    plt.title('收益率回测图')
    # # 给数据点加标签
    # for a, b in zip(x, y):
    #     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8, rotation=45)
    plt.savefig(result_path + file + 'syl.png', bbox_inches='tight')  # 要在show前save，不然会空白
    # bbox_inches='tight'是为了防止保存的图片不完整
    plt.show()


# 回测函数
def trace_back(mydata, file, ifplt=True):
    x = np.array([])  # 收益率横坐标和纵坐标
    y = np.array([])
    i = len(mydata)  # 获取总的数据条数
    money = benjin  # money是当前剩余的资金数量
    timestr = mydata.loc[i - 1]['成交日期']
    enddate = mydata.loc[0]['成交日期']  # csv文件的最后一天
    # 注意joinquant这里要求日期格式为year-month-day，中间为-，目前的交割单导入python可以由于编码问题，默认就是-
    timestr = datetime.strptime(timestr, '%Y-%m-%d')
    enddate = datetime.strptime(enddate, '%Y-%m-%d')
    sum = 0
    sum_yinli_count = 0  # 总盈利次数
    sum_jiaoyi_count = 0  # 总交易次数
    sum_kuisun_count = 0  # 总亏损次数
    sum_yinli = 0
    sum_kuisun = 0
    # ---------------------------------计算收益率------------------------------------------------
    print('正在回测...')
    trade_days = get_trade_days(start_date=timestr, end_date=enddate)
    count = 0  # 用来遍历交易日表
    while timestr.__le__(enddate):  # 时间从开始至结束每天计算收益率，股市不开盘的日子跳过
        # 这是转换时间格式，tushare的pro接口只支持'%Y%m%d'这种格式的时间
        date = timestr
        my_str = datetime.strftime(timestr, '%Y-%m-%d')
        # 字符串格式转换，datetime转字符串会变成2021/01/01这类格式，与csv中数据不符合，得去掉月和日开头的0，如果符合的可以注释掉下面几行
        # if my_str[8] == '0':
        #     my_str = my_str[0:8] + my_str[9:]
        # if my_str[5] == '0':
        #     my_str = my_str[0:5] + my_str[6:]
        # print(my_str)
        tmp_data = mydata.loc[mydata['成交日期'] == my_str]  # 获取日期为该日的所有数据
        tmp_len = len(tmp_data)
        sum += tmp_len
        # 先遍历已有的股票，修改其收盘价

        for key in my_stock.keys():
            if key[7:] == 'SZA':
                key = key[0:7] + 'XSHE'
            elif key[7:] == 'SHA':
                key[7:] = key[0:7] + 'XSHG'
            OHLC = get_price(key, start_date=timestr, end_date=timestr, frequency='daily',
                             fields=['open', 'close', 'high', 'low', 'volume', 'money'], panel=False)
            # 获取当日该股票的OHLC等数据
            closeval = OHLC.loc[my_str]['close']  # 当日收盘价
            my_stock[key][2] = closeval
        # print(my_stock)
        for k in range(tmp_len):
            stock_code = mydata.loc[i - 1]['证券代码']
            if stock_code[7:] == 'SZA':
                stock_code = stock_code[0:7] + 'XSHE'
            elif stock_code[7:] == 'SHA':
                stock_code = stock_code[0:7] + 'XSHG'
            if stock_code == '******':  # 还有部分交割单存在******，直接采取跳过操作
                i -= 1
                continue
            vol = mydata.loc[i - 1]['成交数量']
            OHLC = get_price(stock_code, start_date=timestr, end_date=timestr, frequency='daily',
                             fields=['open', 'close', 'high', 'low', 'volume', 'money'], panel=False)
            # 获取当日该股票的OHLC等数据
            # 这个获取的OHLC标签为日期而不是0123这种
            openval = OHLC.loc[my_str]['open']  # 当日开盘价
            closeval = OHLC.loc[my_str]['close']  # 当日收盘价
            print(OHLC)
            if mydata.loc[i - 1]['买卖标志'] == '买入':
                if not my_stock.__contains__(stock_code):
                    my_stock[stock_code] = [vol, openval, closeval]
                else:
                    my_stock[stock_code][0] += vol
                # 总资金减去手续费
                money -= vol * openval * cost_buy  # 减去手续费
                money -= vol * openval  # 买入后剩余资金减少
            else:  # 卖出操作
                sum_jiaoyi_count += 1
                closeval = mydata.loc[i-1]['成交价格']   # 有可能是开盘价卖出，不一定是收盘价的，要保证交割单数据正确
                if my_stock.__contains__(stock_code) == 0:  # 防止只买不卖
                    i -= 1
                    continue
                my_stock[stock_code][0] -= vol
                if closeval > my_stock[stock_code][1]:
                    sum_yinli_count += 1
                    sum_yinli += vol * (closeval - my_stock[stock_code][1])
                elif closeval < my_stock[stock_code][1]:
                    sum_kuisun_count += 1
                    sum_kuisun += vol * (my_stock[stock_code][1] - closeval)
                if my_stock[stock_code][0] == 0:  # 剩余股票量为0，则从字典中删掉这个股票
                    del my_stock[stock_code]
                money -= vol * closeval * cost_sold
                money += vol * closeval  # 卖出后剩余资金增加
            i -= 1  # i-1，读取csv中下一条数据
        tmp_money = money
        # print(my_str + "    " + str(money))
        # print(my_stock)
        x, y = cal_syl_daily(x, y, tmp_money, my_str)
        count += 1

        if count == len(trade_days):
            break
        timestr = trade_days[count]  # 找下一个交易日

    syl_table = pd.DataFrame(syl_list, columns=['日期', '收益率'])  # 创建收益率表并打印
    print(syl_table)
    syl_table.to_csv(result_path + file + 'syl.csv')
    year_syl = syl_table.loc[len(syl_table) - 1]['收益率']
    # 使用bigquant网站上的公式计算年化收益率，和网站结果一致，精度上有一些细微误差,由于他时间前移一天，所以持有天数要+1
    if 1 + float(year_syl[:len(year_syl) - 1]) / 100 < 0:
        year_syl = -(math.pow(math.fabs(1 + float(year_syl[:len(year_syl) - 1]) / 100),
                              252 / (len(syl_table) + 1)) - 1) * 100
    else:
        year_syl = (math.pow((1 + float(year_syl[:len(year_syl) - 1]) / 100), 252 / (len(syl_table) + 1)) - 1) * 100
    year_syl = round(year_syl, 2)
    print('年化收益率为' + str(year_syl) + '%')  # 打印年化收益率

    # -----------------------------------计算胜率-------------------------------------------
    shenglv = sum_yinli_count / sum_jiaoyi_count * 100
    shenglv = round(shenglv, 1)
    print('胜率为' + str(shenglv) + '%')

    # -----------------------------------计算盈亏比-------------------------------------------
    # 条件判断，防止分母为0
    if sum_kuisun_count == 0:
        yinkuibi = 100.00
    elif sum_yinli_count == 0:
        yinkuibi = 0
    else:
        yinkuibi = (sum_yinli / sum_yinli_count) / (sum_kuisun / sum_kuisun_count)
    yinkuibi = round(yinkuibi, 2)
    print('盈亏比为' + str(yinkuibi))
    # -----------------------------------计算最大回撤率---------------------------------------
    max_huiche = -999999
    max_huiche = cal_max_huiche(y, max_huiche)
    print("最大回撤率为" + str(max_huiche) + '%')

    # -----------------------------------计算夏普比率----------------------------------------
    # 夏普比率需要每日收益率而非累计收益率，因此需要先计算每日收益率，计算标准差的同时顺带计算出了收益波动率
    z = np.zeros(len(y))
    z[0] = y[0]
    # print(z)
    # z[0] = y[0]
    for i in range(1, len(y)):
        z[i] = (y[i] - y[i - 1]) / (1 + y[i - 1] / 100)
    # print(z
    sharp, syl_std = cal_sharp(year_syl, z, len(syl_table))
    print('夏普比率为' + str(sharp))
    zhibiao = [
        [str(year_syl) + '%', str(shenglv) + '%', str(yinkuibi), str(max_huiche) + '%', str(syl_std) + '%', str(sharp)]]
    zhibiao_pd = pd.DataFrame(columns=['年化收益率', '胜率', '盈亏比', '最大回撤率', '收益波动率', '夏普比率'], data=zhibiao)
    zhibiao_pd.to_csv(result_path + file + 'zhibiao.csv')
    # -----------------------------------绘制折线图----------------------------------------
    if ifplt:
        plot_syl(x, y, file)

    # 时间格式转换
    # time = '2019-08-01'
    #
    # date = datetime.strptime(time, '%Y-%m-%d')
    # date = datetime.strftime(date, '%Y%m%d')
    # print(date)

    # data.to_csv("a.csv")
    # data = np.array(data)
    # print(data.loc[data['trade_date'] == '20181123','amount'] == 318522.557)


def huice():
    my_path = data_path
    dirs = os.listdir(my_path)
    error_table = []  # 用来找错误的文件，交割单正确后可以不用
    for file in dirs:
        my_stock.clear()  # 清空mystock
        print(file)
        jump_flag = 0  # 跳过delay的文件，因为原始文件错了delay的文件都不用跑了
        for i in range(len(error_table)):
            if len(file) > len(error_table[i]):
                if file[0:len(error_table[i])] == error_table[i]:
                    jump_flag = 1
                    continue
        if jump_flag == 1:
            continue
        syl_list.clear()
        global syl
        syl = 0.0  # 初始收益率为0
        # 读取交割单数据
        data = pd.read_csv(my_path + '\\' + file)
        err_flag = 0
        # 中间有*就不能继续操作了,在结束日则无所谓
        for i in range(len(data)):
            if (data.loc[i]['证券代码'] == '******') and (data.loc[i]['成交日期'] != data.loc[0]['成交日期']):
                error_table.append(file)
                err_flag = 1
                break
        if err_flag == 1:
            continue
        # print(error_table)
        # print(data)
        pd.set_option('display.width', None)  # 可以让print中间显示不省略
        pd.set_option('display.max_rows', None)
        trace_back(data, file[0:-4], if_plt)

    # err_pd = pd.DataFrame(columns=['filename'], data=error_table)
    # err_pd.to_csv('yilong60_err.csv')
