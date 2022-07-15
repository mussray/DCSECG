
import torch
import numpy as np

import wfdb
torch.set_default_tensor_type(torch.DoubleTensor)

def getDataSet(number,ecgdata):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()        #每一个患者心电信号长度650000
    # 将650000信号分成65段，每段长为10000
    for i in range(0, 204800, 256):#for i in range(0, 650000, 10000):
        data1=np.array(data[i:i +256])
        ecgdata.append(data1)

    return ecgdata

def data_loader():

    #trainset=['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 #'116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 #'210', '212', '213', '214', '215', '217', '219', '220', '221']
    #validset=['222', '223', '228', '230', '231', '232', '233', '234']
    trainset = ['108','117','208','219']
    validset = ['231','223']
    '''trainset = ['103', '105', '106', '108', '112', '113', '114', '116', '121', '122', '123', '124', '200', '201', '202',
                '203', '205', '207', '208', '209',
                '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232',
                '233', '234']
    validset = ['100', '101', '107', '109', '111', '115', '117', '118', '119']  # ,'104','102'
    validset = ['100', '101', '107', '109', '117']'''
    #trainset = ['100']
    #validset = ['222']
    trainset = ['103', '105', '106', '108', '112', '113', '114', '116', '121', '122',
                '210', '212', '213', '214', '215', '217', '219', '220', '221', '222']
    validset = ['100', '101', '107', '109', '117']


    '''
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    '''
    traindata=[]
    validdata=[]
    for n in trainset:
        getDataSet(n, traindata)
    for n in validset:
        getDataSet(n, validdata)

    # 转numpy数组,打乱顺序
    traindata = np.array(traindata).reshape(-1, 256)
    np.random.shuffle(traindata)
    traindata=torch.from_numpy(traindata)
    traindata=torch.unsqueeze(traindata,1)

    traindata = torch.unsqueeze(traindata, 1)


    validdata = np.array(validdata).reshape(-1, 256)
    validdata = torch.from_numpy(validdata)
    validdata = torch.unsqueeze(validdata, 1)

    validdata = torch.unsqueeze(validdata, 1)
    return traindata, validdata
