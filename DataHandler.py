import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log    


def transpose(mat):
    coomat = sp.coo_matrix(mat)
    #sp.coo_matrix 标格式的稀疏矩阵。
    return csr_matrix(coomat.transpose())
    #return （0，1） 1 这种格式的稀疏矩阵


def transToLsts(mat, mask=False, norm=False):
    #传进来的是prepare里的adj
    shape = [mat.shape[0], mat.shape[1]]
    coomat = sp.coo_matrix(mat)
    indices = np.array(list(map(list,zip(coomat.row, coomat.col))), dtype=np.int32)
    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    data = coomat.data.astype(np,float32)

    if norm:
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1)+ 1e-8)+ 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1)+ 1e-8)+ 1e-8)))
        for i in range(len(data)):
            row = indices[i, 0]
            col = indices[i, 1]
            data[i] = data[i] * rowD[row] * colD[col]

    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
        data = data * spMask

    if inidices.shape[0] == 0:
        inidices = np.array([[0,0]], dtype=np.int32)
        data = np.array([0.0],np.float32)
    return indices, data, shape



class DataHandler:
    def __init__(self):
        if args.data == 'tmall':
            predir = './Datasets/Tmall/'
            behs = ['pv', 'fav', 'cart', 'buy']
        elif args.data == 'beibei':
            predir = './Datasets/beibei/'
            behs = ['pv', 'cart', 'buy']
			#page view, add to cart, purchase
        elif args.data == 'ijcai':
            predir = './Datasets/ijcai/'
            behs = ['click', 'fav', 'cart', 'buy']
        self.predir = predir
        self.behs = behs
        self.trnfile = predir + 'trn_'
        self.tstfile = predir + 'tst_'

    def LoadData(self):
        trnMats = list()
        #把所有的数据都放到了一起
        #train set????
        for i in range(len(self.behs)):
            beh = self.behs[i]
            path = self.trnfile + beh
            with open(path, 'rb') as fs:
                mat = (pickle.load(fs) != 0).astype(np.float32)
            trnMats.append(mat)
        #test set
        path = self.tstfile + 'int'
        with open(path, 'rb') as fs:
            tstInt = np.array(pickle.load(fs))
            #tstInt [None None 7804 ... None 354 None]
        tstStat = (tstInt != None)
        #tstStat [False False  True ... False  True False]
        tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
        #tstUser [    2     7     8 ... 21709 21711 21714]
        self.trnMats = trnMats
        self.tstInt = tstInt
        self.tstUsrs = tstUsrs
        args.user, args.item = self.trnMats[0].shape
        #21716 7977
        args.behNum = len(self.behs)
        #3
        self.prepareGlobalData()
        #get到了global的之后继续返回进入他的函数
    
    def prepareGlobalData(self):
        #准备全局数据
        adj = 0
        for i in range(args.behNum):
            adj = adj + self.trnMats[i]
        adj = (adj != 0).astype(np.float32)  
        self.labeIP = np.squeeze(np.array(np.sum(adj, axis=0)))
        #len(self.labeIP) 7977 //也就是说这里只保留了物品
        #np.sum()  adj-用于进行加法运算的数组形式的元素//axis=none 将所有元素进行相加
        #axis=0  即将每一列的元素相加,将矩阵压缩为一行 axis=1压缩列
        #np.array() 创建一个数组
        #np.squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        # print(adj)
        # (0, 53)       1.0
        # (0, 198)      1.0
        # (0, 415)      1.0
        # (0, 422)      1.0
        # (0, 539)      1.0
        # print(np.sum(adj, axis=0))
        # [[ 95.  96.  88. ... 122.  79. 305.]]
        # print(np.array(np.sum(adj, axis=0)))
        # [[ 95.  96.  88. ... 122.  79. 305.]]
        # print(np.squeeze(np.array(np.sum(adj, axis=0))))
        # [ 95.  96.  88. ... 122.  79. 305.]
        tpadj = transpose(adj)
        print(tpadj)
        #tp_adj
#           (0, 220)      1.0
#           (0, 257)      1.0
#           (0, 328)      1.0
#           (0, 373)      1.0
#           (0, 502)      1.0
#           (0, 782)      1.0
#           (0, 920)      1.0
        adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
        # print(adjNorm)
        # [ 95.  64.  59. ... 182.  87. 138.]
        #np.reshape()给数组一个新的形状而不改变其数据,原来是竖着的现在改成横着的
        tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
        # [ 95.  96.  88. ... 122.  79. 305.]
        for i in range(adj.shape[0]):
            # shape[0]宽度 shape[1]高度
            for j in range(adj.indptr[i], adj.indptr[i+1]):
                adj.data[j] /= adjNorm[i]
        for i in range(tpadj.shape[0]):
            for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
                tpadj.data[j] /= tpadjNorm[i]
        self.adj = adj
        self.tpadj = tpadj
        #拿到了adj 和 tpadj的值， 返回到他的进来时候的函数，LoadData

# def sampleLargeGraph(self, pckUsers, pckItems=None, sampDepth=2, sampNum=args.sampleSampleN, preSamp=False):
#     adj = self.adj
#     tpadj = self.tpadj
#     def makeMask(nodes, size):
#         mask = np.ones(size)
#         print("mask pre", mask)
#         if not nodes is None:
#             #如果节点不是空
#             mask[nodes] = 0.0
#         print("mask after:", mask)
#         return mask



















    



            
