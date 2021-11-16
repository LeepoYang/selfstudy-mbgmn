import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender():
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER',args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['loss','preLoss','HR','NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
   
    def makePrint(self, name, ep, reses, save):
        pass

    def run(self):
        self.prepareModel()
        #去到准备模型


    

    def defineModel(self):
        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim//2], reg=True)
        #return net
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim//2], reg=True)
        behEmbeds = NNs.defineParam('behEmbeds', [args.behNum + 1, args.latdim//2])

        self.ulat = [0] * (args.behNum + 1)
        self.ilat = [0] * (args.behNum + 1)

        for beh in range(args.behNum):
            params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[beh, [self.adjs[beh]], [self.tpAdjs[beh]]])
            behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)

            ulats = [behUEmbed0]
            ilats = [behIEmbed0]

            for i in range(args.gnn_layer):
                ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
                ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
                ulats.append(ulat + ulats[-1])
                ilats.append(ilat + ilats[-1])
            self.ulat[beh] = tf.add_n(ulats)
            self.ilat[beh] = tf.add_n(ilats)
        
        params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[-1], self.adjs, self.tpAdjs)
        behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)

        ulats = [behUEmbed0]
        ilats = [behIEmbed0]

        for i in range(args.gnn_layer):
            ubehLats = []
            ibehLats = []

            for beh in range(args.behNum):
                ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
                ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
                ubehLats.append(ulat)
                ibehLats.append(ilat)
            ulat = tf.add_n(NNs.lightSelfAttention(ubehLats, args.behNum, args.latdim, args.att_head))
            ilat = tf.add_n(NNs.lightSelfAttention(ibehLats, args.behNum, args.latdim, args.att_head))

            ulats.append(ulat)
            ilats.append(ilat)
        self.ulat[-1] = tf.add_n(ulats)
        self.ilat[-1] = tf.add_n(ilats)
        
    def _predict(self, ulat, ilat, params):
        predEmbed = tf.expand_dims(tf.concat([ulat * ilat, ulat, ilat], axis=-1), axis=1)
        predEmbed = Activate(predEmbed @ params['w1'] + params['b1'], self.actFunc)
        preds = tf.squeeze(predEmbed @ params['w2'])
        return preds

    def predict(self, src, tgt):
        #source target
        uids = self.uids[tgt]
        iids = self.iids[tgt]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)
        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)
        # tf.nn.embedding_lookup 的用途主要是选取一个张量里面索引对应的元素

        predParams = self.metaForPredict(src_ulat, src_ilat, tgt_ulat, tgt_ilat)
        return self._predict(src_ulat, src_ilat, predParams) * args.mult

    def messagePropagate(self, lats, adj, lats2):
        return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)

    def metaForSpecialize(self, uEmbed, iEmbed, behEmbed, adjs, tpAdjs):
        latdim = args.latdim // 2
        rank = args.rank
        assert len(adjs) == len(tpAdjs)
        uNeighbor = iNeighbor = 0
        for i in range(len(adjs)):
            uNeighbor += tf.sparse.sparse_dense_matmul(adjs[i], iEmbed)
            #tf.sparse.sparse_dense_matmul 将稀疏张量(或密实矩阵)(等级为2)"A "乘以密实矩阵。
            iNeighbor += tf.sparse.sparse_dense_matmul(tpAdjs[i], uEmbed)
        ubehEmbed = tf.expand_dims(behEmbed, axis=0) * tf.ones_like(uEmbed)
        ibehEmbed = tf.expand_dims(behEmbed, axis=0) * tf.ones_like(iEmbed)
        
        uMetaLat = FC(tf.concat([ubehEmbed, uEmbed, uNeighbor], axis=-1), latdim, useBias=True, activation=self.actFunc, reg=True, name='specMeta_FC1', reuse=True)
        iMetaLat = FC(tf.concat([ibehEmbed, iEmbed, iNeighbor], axis=-1), latdim, useBias=True, activation=self.actFunc, reg=True, name='specMeta_FC1', reuse=True)
        
        uW1 = tf.reshape(FC(uMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC2', reuse=True), [-1, latdim, rank])
        uW2 = tf.reshape(FC(uMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC3', reuse=True), [-1, rank, latdim])
        iW1 = tf.reshape(FC(iMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC4', reuse=True), [-1, latdim, rank])
        iW2 = tf.reshape(FC(iMetaLat, rank * latdim, useBias=True, reg=True, biasInitializer='xavier', biasReg=True, name='specMeta_FC5', reuse=True), [-1, rank, latdim])

        params = {'uW1': uW1, 'uW2': uW2, 'iW1': iW1, 'iW2': iW2}
        return params


        


    
    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.tpAdjs = []
        self.uids, self.iids = [], []

        for i in range(args.behNum):
            adj = self.handler.trnMats[i]
            idx, data, shape = transTolsts(adj, norm=True)
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))
            idx, data, shape = transToLsts(transpose(adj), norm=True)
            self.tpAdjs.append(tf.sparse.SparseTensor(idx, data, shape))
            self.uids.append(tf.placeholder(name='uids'+str(i), dtype=tf.int32, shape=[None]))
            #tf.placeholder减少开销
            self.iids.append(tf.placeholder(name='iids'+str(i), dtype=tf.int32, shape=[None]))

        self.defineModel()
        self.preLoss = 0

        for src in range(args.behNum + 1):
            for tgt in range(args.behNum):
                preds = self.predict(src, tgt)
                sampNum = tf.shape(self.uids[tgt])[0] // 2
                posPred = tf.slice(preds, [0], [sampNum])
                negPred = tf.slice(preds, [sampNum], [-1])

                self.preLoss += tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
                if src == args.behNum and tgt == args.behNum - 1:
                    self.targetPreds = preds
            
        self.regLoss = args.reg * Regularize()
        self.loss = self.preLoss + self.regLoss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

        





            



    








if __name__ == '__man__':
    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('开始')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    
    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        #进入到recommender class中
        recom.run()
        # 开始执行run