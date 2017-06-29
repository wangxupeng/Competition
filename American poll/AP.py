# coding:utf-8
import numpy as np
import datetime
import matplotlib.pyplot as plt


def is_convert_float(s):
    """
         判断一个字符串能否转换为float
    """
    try:
        float(s)
    except:
        return False
    
    return True


def get_sum(str_array):
    """
        返回字符串数组中数字的总和
    """
    # 去掉不能转换成数字的数据
    cleaned_data = list(filter(is_convert_float, str_array))  #python2 与python3不同,python3这里要加上list 
    
    # 转换数据类型
    float_array = np.array(cleaned_data, np.float)
    
    return np.sum(float_array)


def run_main():
    filename="presidential_polls.csv"
    with open(filename,"r") as f:
        col_names_str = f.readline()[:-1] #[:-1]表示不读取末尾的换行符
    #讲字符串拆分,并组成列表
    col_name_lst= col_names_str.split(",")
    # 使用列名
    use_col_name_lst = ['enddate' , 'rawpoll_clinton' , 'rawpoll_trump' , 'adjpoll_clinton' , 'adjpoll_trump' ]
    # 获取相应的列名索引号
    use_col_index_lst = [col_name_lst.index(use_col_name) for use_col_name in use_col_name_lst]
    
    #-------------------------------------------------------读取数据
    data_array = np.loadtxt(filename,      #文件名
                            delimiter=',', #分隔符
                            skiprows=1,    #跳过第一行
                            dtype=bytes,     #数据类型
                            usecols=use_col_index_lst).astype(str) #用指定列
    '''
    I think np.loadtxt("tile", dtype=bytes, delimiter="\n").astype(str) might work, 
    but I agree completely with the overkill point (解决编码不一样的问题)
    '''
    #--------------------------------------------------------数据处理
    #处理日期格式的数据
    enddate_idx = use_col_name_lst.index('enddate')

    enddate_lst= data_array[:,enddate_idx].tolist()
    
    # 将日期字符串格式统一,即'yy/dd/mm'
    enddate_lst = [enddate.replace('-','/') for enddate in enddate_lst]
    
#    将日期字符串转换成日期
    date_lst = [datetime.datetime.strptime(enddate,'%m/%d/%Y') for enddate in enddate_lst] #python2 与pyhton3 在这里不同
    month_lst = ['%d-%02d' %(date_obj.year, date_obj.month) for date_obj in date_lst]
    
    month_array = np.array(month_lst)
    months = np.unique(month_array)
#    -------------------------------------------------------数据分析
#    统计民意投票数
#    cliton
#    原始数据 rawpoll
    rawpoll_clinton_idx = use_col_name_lst.index("rawpoll_clinton")
    rawpoll_clinton_data = data_array[:, rawpoll_clinton_idx]

               
    #调整之后的数据 adhpool
    adjpoll_clinton_idx = use_col_name_lst.index('adjpoll_clinton')
    adjpoll_clinton_data = data_array[:, adjpoll_clinton_idx]
    
    # trump
    # 原始数据 rawpoll
    rawpoll_trump_idx = use_col_name_lst.index('rawpoll_trump')
    rawpoll_trump_data = data_array[:, rawpoll_trump_idx]

    # 调整后的数据 adjpoll
    adjpoll_trump_idx = use_col_name_lst.index('adjpoll_trump')
    adjpoll_trump_data = data_array[:, adjpoll_trump_idx]
    
    # 结果保存
    results = []

    for month in months:   
    # clinton
    # 原始数据 rawpoll
        rawpoll_clinton_month_data = rawpoll_clinton_data[month_array == month]
    # 统计当月的总票数
        rawpoll_clinton_month_sum = get_sum(rawpoll_clinton_month_data)
    
    # 调整数据 adjpoll
        adjpoll_clinton_month_data = adjpoll_clinton_data[month_array == month]  
    # 统计当月的总票数
        adjpoll_clinton_month_sum = get_sum(adjpoll_clinton_month_data)
    
    # trump
    # 原始数据 rawpoll
        rawpoll_trump_month_data = rawpoll_trump_data[month_array == month]
    # 统计当月的总票数
        rawpoll_trump_month_sum = get_sum(rawpoll_trump_month_data)
    
    # 调整数据 adjpoll
        adjpoll_trump_month_data = adjpoll_trump_data[month_array == month]
    # 统计当月的总票数
        adjpoll_trump_month_sum = get_sum(adjpoll_trump_month_data)
    
        results.append((month, rawpoll_clinton_month_sum, adjpoll_clinton_month_sum, rawpoll_trump_month_sum, adjpoll_trump_month_sum))           
    print(results)

    months, raw_cliton_sum, adj_cliton_sum, raw_trump_sum, adj_trump_sum = zip(*results)          
    
    # * 可视化分析结果        
    fig, subplot_arr = plt.subplots(2,2, figsize=(15,10))

    # 原始数据趋势展示
    subplot_arr[0,0].plot(raw_cliton_sum, color='r')
    subplot_arr[0,0].plot(raw_trump_sum, color='g')

    width = 0.25
    x = np.arange(len(months))
    subplot_arr[0,1].bar(x, raw_cliton_sum, width, color='r')
    subplot_arr[0,1].bar(x + width, raw_trump_sum, width, color='g')
    subplot_arr[0,1].set_xticks(x + width)
    subplot_arr[0,1].set_xticklabels(months, rotation='vertical')
    
    # 调整数据趋势展示
    subplot_arr[1,0].plot(adj_cliton_sum, color='r')
    subplot_arr[1,0].plot(adj_trump_sum, color='g')
    
    width = 0.25
    x = np.arange(len(months))
    subplot_arr[1,1].bar(x, adj_cliton_sum, width, color='r')
    subplot_arr[1,1].bar(x + width, adj_trump_sum, width, color='g')
    subplot_arr[1,1].set_xticks(x + width)
    subplot_arr[1,1].set_xticklabels(months, rotation='vertical')
    
    plt.subplots_adjust(wspace=0.2)
    
    plt.show()

if __name__ == '__main__':
    run_main()       
