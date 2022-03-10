from jqdatasdk import *
import pandas as pd
import numpy as np
import talib
from huice_joinquant import huice
cost_sold = 0.0013  # 卖出手续费比率
cost_buy = 0.0003  # 买入手续费比率
benjin = 1000000  # 本金为1000000
auth('13585176258', 'Wlc990108***')  # 登录joinquant
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列
delta_T = 7  # 默认是隔天收盘价卖出，如为2就是隔两天收盘价卖出

# 股票池以及获取数据的时间参数
do_get_data = False  # 是否爬取数据，爬一次以后之后只要直接从本地读就可以了，设为True则是爬取
stock = list(get_all_securities(types='stock').index)  # 默认股票池为所有股票
start_date = '2020-8-1'  # 开始日期
end_date = '2021-11-28'  # 结束日期

# 策略选股是否执行
# 1表示麻花策略选股， 2表示五星策略选股， 3表示红三兵策略选股，其他数字表示不执行策略选股,由于都执行过一次，所以设为0，直接从本地读就可以
# 这一步耗时较长
select_strategy = 1

# 生成交割单是否执行以及所用的策略
jgd_strategy = 1  # 1表示生成麻花策略交割单， 2表示生成五星策略交割单， 3表示生成五星策略交割单
do_get_jgd = True  # 是否生成交割单， 回测代码需要对应的交割单，默认为生成

# 使用止盈和止损策略
stop_profit = 0.2
stop_loss = 0.15

# 保存交割单的路径
path = "D:\\WLC_Project\\Wlc's_Python_project\\stock_strategy\\data\\"


def ma(series, n):  # 简单移动平均
    # print(series)
    # return talib.MA(series, timeperiod = n)
    return pd.Series.rolling(series, n).mean()


def hhv(series, n):  # 最大值
    return pd.Series.rolling(series, n).max()


def llv(series, n):  # 最小值
    return pd.Series.rolling(series, n).min()


# 爬取数据
def get_stock_data(flag):
    if flag:
        tmp_df = get_price(stock, start_date=start_date, end_date=end_date,
                           fields=['open', 'close', 'low', 'high'], panel=False, skip_paused=True)     #!!!!!!!!!!!!!!
        tmp_df = tmp_df.sort_values(by=['code', 'time'])
        tmp_df.to_csv('stock.csv', index=False)
        print(tmp_df)


# 麻花策略选股
def mahua(df, flag):
    if flag:
        print("正在执行麻花策略选股...")
        print("现在需要遍历股票列表的数据")
        selection = dict()
        for security in stock:
            stock_data = df.loc[df['code'] == security]
            if len(stock_data) == 0:
                continue
            print(stock_data)
            stock_data = stock_data.reset_index(drop=True)  # 重置索引，不然用的是df的索引
            CLOSE = stock_data['close']
            HIGH = stock_data['high']
            LOW = stock_data['low']
            OPEN = stock_data['open']
            MA5 = round(ma(CLOSE, 5), 2)
            MA8 = round(ma(CLOSE, 8), 2)
            MA13 = round(ma(CLOSE, 13), 2)
            MA21 = round(ma(CLOSE, 21), 2)
            MA34 = round(ma(CLOSE, 34), 2)
            MA55 = round(ma(CLOSE, 55), 2)
            stock_data['MA5'] = MA5
            stock_data['MA8'] = MA8
            stock_data['MA13'] = MA13
            print("正在对" + security + "进行回测..")
            for i in range(54, len(stock_data) - 1):  # 到最后一天的前1天，因为这里获取的是买入的时间（第二天开盘买），若是到最后一天会越界
                # A1赋值:B1=B2 OR B2=B3 OR B1=B3 且要求最近3日存在A1
                A1 = 0
                if MA5[i] == MA8[i] or MA5[i - 1] == MA8[i - 1] or MA5[i - 2] == MA8[i - 2] or MA8[i] == MA13[i] or MA8[
                    i - 1] == MA13[i - 1] or MA8[i - 2] == MA13[i - 2] or MA5[i] == MA13[i] or MA5[i - 1] == MA13[
                    i - 1] or \
                        MA5[i - 2] == MA13[i - 2]:
                    A1 = 1
                if A1 == 0:
                    continue
                # A2赋值:最近34日存在收盘价/1日前的收盘价>1.09 AND C=最高价
                A2 = 0
                for j in range(i - 33, i + 1):
                    if CLOSE[j] / CLOSE[j - 1] > 1.09 and CLOSE[j] == HIGH[j]:
                        A2 = 1
                        break
                if A2 == 0:
                    continue
                # A3赋值:从前4日到前0日持续收盘价的13日简单移动平均>收盘价的34日简单移动平均AND收盘价的21日简单移动平均>收盘价的55日简单移动平均
                A3 = 1
                for j in range(i - 4, i + 1):
                    if MA13[j] <= MA34[j]:
                        A3 = 0
                        break
                if MA21[i] <= MA55[i]:
                    A3 = 0
                if A3 == 0:
                    continue
                # A4没用到，略过
                # A5赋值:开盘价<B1和B2的较小值和B3的较小值 AND 收盘价>B1和B2的较大值和B3的较大值
                A5 = 0
                if OPEN[i] < min(min(MA5[i], MA8[i]), MA13[i]) and CLOSE[i] > max(max(MA5[i], MA8[i]), MA13[i]):
                    A5 = 1
                if A5 == 0:
                    continue
                # A6赋值:(最高价-收盘价)>=(开盘价-最低价) AND 收盘价/开盘价<1.08 AND 收盘价=5日内收盘价的最高值
                A6 = 0
                if (HIGH[i] - CLOSE[i]) >= (OPEN[i] - LOW[i]) and CLOSE[i] / OPEN[i] < 1.08 and CLOSE[i] == \
                        hhv(CLOSE, 5)[
                            i]:
                    A6 = 1
                # A7赋值:B1和B2的较大值和B3的较大值/B1和B2的较小值和B3的较小值<1.012
                A7 = 0
                if max(max(MA5[i], MA8[i]), MA13[i]) / min(min(MA5[i], MA8[i]), MA13[i]) < 1.012:
                    A7 = 1
                if A7 == 0:
                    continue
                # A8赋值:收盘价/开盘价>1.08 AND 最高价=21日内最高价的最高值
                A8 = 0
                if CLOSE[i] / OPEN[i] > 1.08 and HIGH[i] == hhv(HIGH, 21)[i]:
                    A8 = 1
                time = stock_data.loc[i + 1]['time']  # 第二天开盘买入
                code = stock_data.loc[i + 1]['code']
                # print(time)
                # print(A1)
                # print(A2)
                # print(A3)
                # print(A5)
                # print(A7)
                # print(MA13[i] > MA21[i])
                # print(A6)
                # print(A8)
                if MA13[i] > MA21[i] and (A6 == 1 or A8 == 1):
                    if selection.__contains__(time):
                        selection[time].append(code)
                    else:
                        selection[time] = [code]
            # print(selection)

        np.save('mahua_selection.npy', selection)


# 红三兵策略选股
def hongsanbing(df, flag):
    if flag:
        print("正在执行红三兵策略选股...")
        print("现在需要遍历股票列表的数据")
        selection = dict()
        for security in stock:
            stock_data = df.loc[df['code'] == security]
            if len(stock_data) == 0:
                continue
            stock_data = stock_data.reset_index(drop=True)  # 重置索引，不然用的是df的索引
            print("正在对" + security + "进行回测..")
            for i in range(3, len(stock_data) - 1):  # 到最后一天的前1天，因为这里获取的是买入的时间（第二天开盘买），若是到最后一天会越界
                # 三个红线
                if stock_data.loc[i - 1]['close'] > stock_data.loc[i - 1]['open'] and stock_data.loc[i - 2]['close'] > \
                        stock_data.loc[i - 2]['open'] and stock_data.loc[i - 3]['close'] > stock_data.loc[i - 3][
                    'open']:
                    # k线长度>=2倍
                    if (stock_data.loc[i - 1]['close'] - stock_data.loc[i - 1]['open']) >= 2 * (
                            stock_data.loc[i - 2]['close'] - stock_data.loc[i - 2]['open']) and \
                            (stock_data.loc[i - 2]['close'] - stock_data.loc[i - 2]['open']) >= 2 * (
                            stock_data.loc[i - 3]['close'] - stock_data.loc[i - 3]['open']):
                        # 开盘价大于前一天收盘价
                        if stock_data.loc[i - 1]['open'] >= stock_data.loc[i - 2]['close'] and \
                                stock_data.loc[i - 2]['open'] >= stock_data.loc[i - 3]['close']:
                            print(stock_data.loc[i - 3:i + 1])
                            time = stock_data.loc[i]['time']
                            code = stock_data.loc[i]['code']
                            if selection.__contains__(time):  # 构建该天的前三天出现红三兵现象股票的字典，做t+1
                                selection[time].append(code)
                            else:
                                selection[time] = [code]
        np.save('hongsanbing_selection.npy', selection)


# 五星策略选股
def wuxing(df, flag):
    if flag:
        print("正在执行五星策略选股...")
        print("现在需要遍历股票列表的数据")
        selection = dict()
        for security in stock:
            stock_data = df.loc[df['code'] == security]
            if len(stock_data) == 0:
                continue
            stock_data = stock_data.reset_index(drop=True)  # 重置索引，不然用的是df的索引
            CLOSE = stock_data['close']
            HIGH = stock_data['high']
            LOW = stock_data['low']
            OPEN = stock_data['open']
            MA5 = ma(CLOSE, 5)
            HIGH5 = hhv(HIGH, 5)
            HIGH55 = hhv(HIGH, 55)
            print("正在对" + security + "进行回测..")
            for i in range(54, len(stock_data)):  # 到最后一天的前1天，因为这里获取的是买入的时间（第二天开盘买），若是到最后一天会越界
                # A1赋值:从前4日到前0日持续收盘价>=开盘价 AND 4日前的收盘价/5日前的收盘价>1.04
                # AND 从前4日到前0日持续开盘价>收盘价的5日简单移动平均
                print(stock_data.loc[i]['time'])
                A1 = 0
                if CLOSE[i] >= OPEN[i] and CLOSE[i - 1] >= OPEN[i - 1] and CLOSE[i - 2] >= OPEN[i - 2] and CLOSE[
                    i - 3] >= OPEN[i - 3] \
                        and CLOSE[i - 4] >= OPEN[i - 4] and CLOSE[i - 4] / CLOSE[i - 5] > 1.04 and OPEN[i] > MA5[i] and \
                        OPEN[i - 1] > \
                        MA5[i - 1] and OPEN[i - 2] > MA5[i - 2] and OPEN[i - 3] > MA5[i - 3] and OPEN[i - 4] > MA5[
                    i - 4]:
                    A1 = 1
                if A1 == 0:
                    continue
                # B赋值:收盘价-开盘价
                B = CLOSE - OPEN
                # A2赋值:B<4日前的B AND 1日前的B<4日前的B AND 2日前的B<4日前的B AND 1日前的B<4日前的B
                A2 = 0
                if B[i] < B[i - 4] and B[i - 1] < B[i - 4] and B[i - 2] < B[i - 4] and B[i - 3] < B[i - 4]:
                    A2 = 1
                if A2 == 0:
                    continue
                # A3赋值:从前3日到前0日持续收盘价/1日前的收盘价<1.09
                A3 = 0
                if CLOSE[i] / CLOSE[i - 1] < 1.09 and CLOSE[i - 1] / CLOSE[i - 2] < 1.09 and CLOSE[i - 2] / CLOSE[
                    i - 3] < 1.09 and \
                        CLOSE[i - 3] / CLOSE[i - 4] < 1.09:
                    A3 = 1
                    print(A3)
                if A3 == 0:
                    continue
                time = stock_data.loc[i + 1]['time']  # 第二天开盘买入
                code = stock_data.loc[i + 1]['code']
                if HIGH55[i] == HIGH5[i]:
                    if selection.__contains__(time):
                        selection[time].append(code)
                    else:
                        selection[time] = [code]
        print(selection)
        np.save('wuxing_selection.npy', selection)


def get_jgd(df, flag):
    if flag:
        global benjin  # 本金需要变动，因此使用global声明
        if jgd_strategy == 1:
            sel = np.load('mahua_selection.npy', allow_pickle=True).item()
            print("正在生成麻花策略交割单")
        elif jgd_strategy == 2:
            sel = np.load('wuxing_selection.npy', allow_pickle=True).item()
            print("正在生成五星策略交割单")
        elif jgd_strategy == 3:
            sel = np.load('hongsanbing_selection.npy', allow_pickle=True).item()
            print("正在生成红三兵策略交割单")
        selection = dict()
        for i in sorted(sel):  # 对字典按时间进行排序
            selection[i] = sel[i]
        print(selection)

        trade_days = get_trade_days(start_date=start_date, end_date=end_date)  # 获取交易日期
        transaction = dict()  # 判断某一日是否用来买
        chicang = dict()  # 持仓表，表示当前持有的股票和其买入价
        for i in trade_days:
            transaction[str(i)] = 0  # 1表示买,2表示卖
        # print(transaction)
        keys = list(transaction.keys())
        trade_days = list(selection.keys())
        # print(len(trade_days))
        j = 0
        # 得到哪些天用来买入
        for i in range(len(keys)):
            if j == len(trade_days):
                break
            if keys[i] != trade_days[j]:
                continue
            j += 1
            transaction[keys[i]] = 1  # 该天设为买入
        print(transaction)

        # 生成交割单
        j = 0
        jgd = pd.DataFrame(columns=['成交日期', '证券代码', '买卖标志', '成交数量', '成交价格'])
        for day in transaction.keys():
            # print(day)
            # print(chicang)
            # 判断是否达到止损或止盈条件进行卖出
            stock_tmp = list(chicang.keys())
            delta_benjin = 0
            for i in stock_tmp:
                chicang[i][2] -= 1
                tmp = get_price(start_date=day, end_date=day, frequency='daily', security=i,
                             fields=['open', 'close'], panel=False)
                # tmp = df.loc[(df['time'] == day) & (df['code'] == i)]
                # print("----------------------------------------------------------------")
                # print(tmp)
                open_price = float(tmp['open'])
                close_price = float(tmp['close'])
                vol = chicang[i][1]  # 购买量
                chengben = chicang[i][0]  # 成本价
                if (open_price - chengben) / chengben >= stop_profit or (
                        open_price - chengben) / chengben <= - stop_loss:  # 开盘价达到止盈或止损条件
                    if i[7:] == 'XSHE':  # 依旧按照bigquant的交割单格式
                        tmp_code = i[0:7] + 'SZA'
                    elif i[7:] == 'XSHG':
                        tmp_code = i[0:7] + 'SHA'
                    tmp_df = pd.DataFrame(
                        {'成交日期': day, '证券代码': tmp_code, '买卖标志': '卖出', '成交数量': vol,
                         '成交价格': open_price}, index=[0])
                    # print(tmp_df)
                    jgd = jgd.append(tmp_df, ignore_index=True)  # 交割单添加一行
                    # print(day + "  收益率:" + tmp_code + str((open_price - chengben) *100 / chengben))
                    # print(day + "  " +tmp_code + " 开盘价"+ str(open_price) + "卖出")
                    # print("benjin=" + str(benjin))
                    benjin += chicang[i][1] * open_price * (1 - cost_sold)
                    # print("卖出后benjin=" + str(benjin))
                    del chicang[i]
                elif (close_price - chengben) / chengben >= stop_profit or (  # 收盘价达到止盈或止损条件，或达到delta_T天
                        close_price - chengben) / chengben <= -stop_loss or chicang[i][2] == 0:
                    if i[7:] == 'XSHE':  # 依旧按照bigquant的交割单格式
                        tmp_code = i[0:7] + 'SZA'
                    elif i[7:] == 'XSHG':
                        tmp_code = i[0:7] + 'SHA'
                    tmp_df = pd.DataFrame(
                        {'成交日期': day, '证券代码': tmp_code, '买卖标志': '卖出', '成交数量': vol,
                         '成交价格': close_price}, index=[0])
                    jgd = jgd.append(tmp_df, ignore_index=True)  # 交割单添加一行
                    delta_benjin += chicang[i][1] * close_price * (1 - cost_sold)  # 这里需要累积收盘价卖出股票钱的总和
                    # print(day + "  " + tmp_code + " 收盘价" + str(close_price) + "卖出")
                    # print("benjin="+str(benjin))
                    # print(day + "  收益率:" + tmp_code + str((close_price - chengben) * 100 / chengben))
                    del chicang[i]
                    # 这里本金先不增加，放到循环最后增加，这样就无法在当日开盘的时候买入
            if transaction[day] == 1:  # 买入操作
                print("正在生成" + day + "的交易信息")
                stock_list = selection[day]
                stock_count = len(stock_list)
                money_per_stock = benjin / stock_count  # 如果有多个股就平均持仓买入
                for i in stock_list:
                    tmp = df.loc[(df['time'] == day) & (df['code'] == i)]
                    if i[7:] == 'XSHE':  # 依旧按照bigquant的交割单格式
                        tmp_code = i[0:7] + 'SZA'
                    elif i[7:] == 'XSHG':
                        tmp_code = i[0:7] + 'SHA'
                    price = float(tmp['open'])
                    vol = int(money_per_stock / ((1 + cost_buy) * 100 * price)) * 100  # 股票只能一手一手买入
                    if vol == 0:  # 虽然选中但没钱买了
                        continue
                    tmp_df = pd.DataFrame({'成交日期': day, '证券代码': tmp_code, '买卖标志': '买入', '成交数量': vol, '成交价格': price},
                                          index=[0])
                    jgd = jgd.append(tmp_df, ignore_index=True)  # 交割单添加一行
                    j += 1
                    benjin -= vol * price * (1 + cost_buy)
                    chicang[i] = [price, vol, delta_T]  # 持仓表增加一行,剩余delta_T天把它在收盘价时卖掉
            benjin += delta_benjin
            # print(day + "    " + str(benjin))
            # if delta_benjin != 0:
            #     print("卖出后benjin="+str(benjin))

        jgd = jgd.sort_values(by=['成交日期'], ascending=False)
        print(jgd)
        if jgd_strategy == 1:
            jgd.to_csv(path + "mahua_deltaT=" + str(delta_T) + "stop_profit=" + str(stop_profit) + "stop_loss=" + str(
                stop_loss) + ".csv", index=False)
        elif jgd_strategy == 2:
            jgd.to_csv(path + "wuxing_deltaT=" + str(delta_T) + "stop_profit=" + str(stop_profit) + "stop_loss=" + str(
                stop_loss) + ".csv", index=False)
        elif jgd_strategy == 3:
            jgd.to_csv(path + "hongsanbing_deltaT=" + str(delta_T) + "stop_profit=" + str(stop_profit) + "stop_loss=" + str(
                stop_loss) + ".csv", index=False)


if __name__ == '__main__':
    do_wuxing_strategy = False  # 是否执行策略进行选股，执行一次以后会把选股字典保存在本地，第二次可以直接读取，不用再执行，要重新执行就设为True
    do_hongsanbin_strategy = False
    do_mahua_strategy = False
    if select_strategy == 1:
        do_mahua_strategy = True
    elif select_strategy == 2:
        do_wuxing_strategy = True
    elif select_strategy == 3:
        do_hongsanbin_strategy = True
    get_stock_data(do_get_data)

    stock_df = pd.read_csv('stock.csv')
    # print(stock_df.head())

    mahua(stock_df, do_mahua_strategy)
    hongsanbing(stock_df, do_hongsanbin_strategy)
    wuxing(stock_df, do_wuxing_strategy)
    get_jgd(stock_df, do_get_jgd)
    print(get_query_count())
    huice()
