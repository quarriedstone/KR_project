import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import random

df_X = pd.read_csv('data/features_ALL.txt', sep = ',',header=None)
df_Y = pd.read_csv('data/ratings.txt', sep = ',',header=None)
print(df_X.shape)
print(df_Y.shape)
combine=pd.concat([df_X,df_Y], axis=1,)
print(combine.shape)
print(df_Y.iloc[1,0])
print(combine.iloc[1,-1])

BATCH_SIZE = 10
ROW_SIZE = 323

#TS AR LF
CONV_KERNELS=[[3,3,1,128],[3,3,128,256]]
POOL_STRIDES =[1,2,2,1]
FC_KERNELS=[256]

LEARNING_RATE = 0.001
EPS = 0.01
STDDEV =0.01

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)

train=np.array(combine[0:490],dtype=np.float)
test=np.array(combine[490:500],dtype=np.float)

class data_gen:
    def __init__(self, file_src):
            self.x_train= []
            self.x_train_tmp=train[:,0:11628]
            for i in range(490):
                self.x_train.append(self.x_train_tmp[i,:].reshape(-1,36))
            self.x_train=np.array(self.x_train)
            self.x_test=[]
            self.x_test_tmp=test[:,0:11628]
            for i in range(10):
               self.x_test.append(self.x_test_tmp[i,:].reshape(-1,36))
            self.x_test=np.array(self.x_test)
            self.y_train=train[:,11628:11629]      
            self.y_test=test[:,11628:11629]
            
    def get_feed_dict(self,idx, train=True):
        if (train ==True):
            x_seg=self.x_train[idx:idx+BATCH_SIZE].reshape(-1,323,36,1)
            y_seg=self.y_train[idx:idx+BATCH_SIZE,0:1].reshape(-1,1)
        else:
            x_seg=self.x_test[idx:idx+BATCH_SIZE].reshape(-1,323,36,1)
            y_seg=self.y_test[idx:idx+BATCH_SIZE,0:1].reshape(-1,1)

        return({X:x_seg,Y:y_seg})

data=data_gen('./src/convolution.csv')
print(data.x_train.shape)
TRAIN_BATCH=int(data.x_train.shape[0]/BATCH_SIZE)
TEST_BATCH=int(data.x_test.shape[0]/BATCH_SIZE)
print(TRAIN_BATCH)
print(TEST_BATCH)

print(data.x_test.shape)
print(data.y_test.shape)
print(data.x_train.shape)
print(data.y_train.shape)
print(data.x_train[0].shape)
print(data.x_train[0])
print(data.y_train[0:3,0:1])

with tf.variable_scope("flight"):
    global_step = tf.Variable(0,trainable=False,name='global_step')
    
    X=tf.placeholder(tf.float32,[None,ROW_SIZE,36,1],name="X")
    Y=tf.placeholder(tf.float32,[None,1], name="Y")
    
    W1=tf.get_variable(shape=CONV_KERNELS[0],initializer = tf.contrib.layers.xavier_initializer(uniform=False),dtype = tf.float32,name="W1")
    
    L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    L1 = tf.nn.softmax(L1)
    
    L2 = tf.nn.max_pool(L1,ksize=POOL_STRIDES,strides = POOL_STRIDES, padding="SAME")
    
    W2 = tf.get_variable(shape=CONV_KERNELS[1],initializer = tf.contrib.layers.xavier_initializer(uniform = False), dtype = tf.float32, name="W2")
    
    L3 = tf.nn.conv2d(L2,W2,strides=[1,1,1,1],padding="SAME")
    L3 = tf.nn.softmax(L3)
    
    L4 = tf.nn.max_pool(L3,ksize=POOL_STRIDES,strides=POOL_STRIDES, padding="SAME")
    
    flatten =int(np.prod(L4.shape[1:]))
    
    W3 = tf.get_variable(shape=[flatten,FC_KERNELS[0]], initializer = tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32,name="W3")
    B3 = tf.Variable(tf.random_normal([BATCH_SIZE,FC_KERNELS[0]],stddev=STDDEV),name="B3")
    
    L5 = tf.reshape(L4,[-1,flatten])
    L5 = tf.nn.softmax(tf.add(tf.matmul(L5,W3),B3))

    W4 = tf.get_variable(shape=[FC_KERNELS[0],1],initializer = tf.contrib.layers.xavier_initializer(uniform=False),
                        dtype=tf.float32, name="W4")
    B4 = tf.Variable(tf.random_normal([BATCH_SIZE,1], stddev=STDDEV),name="B4")

    model = tf.add(tf.matmul(L5,W4),B4)
    
summary_train_tmp = []
summary_test_tmp = []

cost = tf.reduce_mean(tf.square(Y-model))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op =optimizer.minimize(cost,global_step=global_step)

summary_train_tmp.append(tf.summary.scalar('cost',cost))

eps_tensor = tf.constant(EPS,dtype=tf.float32,
                        shape=[BATCH_SIZE,1],
                        name='eps_tensor')

is_correct = tf.less(tf.abs(Y-model),eps_tensor)
correct_count = tf.reduce_sum(tf.cast(is_correct,tf.float32))

summary_test_tmp.append(tf.summary.scalar('correct_count',correct_count))

summary_train = tf.summary.merge(summary_train_tmp)
summary_test = tf.summary.merge(summary_test_tmp)

store_dict={"W1":W1,"W2":W2,"W3":W3,"W4":W4,"B3":B3,"B4":B4,"global_step":global_step}
SAVE_PATH="./models/convolution_basic"

with tf.Session() as sess:
    
    saver = tf.train.Saver(var_list = store_dict,name="flight")
    ckpt = tf.train.get_checkpoint_state('./models')
    
    sess.run(tf.global_variables_initializer())
    
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess,ckpt.model_checkpoint_path)
    
    LOG_DIR='./logs'
    TRAINING_EPOCH = 30
    writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    
    for epoch in range(100):
        total_cost=0
        idx = 0
        for batch_n in range(TRAIN_BATCH):
            dict_to_feed = data.get_feed_dict(idx,True)
            _,cost_val = sess.run([train_op,cost],feed_dict=dict_to_feed)
            
            if batch_n %10 == 0:
                summary=sess.run(summary_train,feed_dict=dict_to_feed)
                writer.add_summary(summary,batch_n)
            
            total_cost += cost_val
            idx += BATCH_SIZE
        
        print('Epoch:',epoch+1,'Average_cost:',total_cost/TRAIN_BATCH)
        
        total_correct = 0
        idx = 0
        
    for batch_n in range(TEST_BATCH):
        total_correct += sess.run(correct_count, feed_dict=data.get_feed_dict(idx,False))

        if(batch_n %10 ==0):
            summary = sess.run(summary_test, feed_dict = data.get_feed_dict(idx,False))
            writer.add_summary(summary,batch_n)

        idx += BATCH_SIZE

    accuracy = total_correct / float(TEST_BATCH*BATCH_SIZE)

    saver.save(sess,SAVE_PATH,global_step=global_step)
    

with tf.Session() as sess:
    saver = tf.train.Saver(var_list = store_dict,name="flight")
    ckpt = tf.train.get_checkpoint_state('./models')
    
    sess.run(tf.global_variables_initializer())
    
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess,ckpt.model_checkpoint_path)
    idx=0
    for i in range(TEST_BATCH):
        prediction=sess.run(model,feed_dict=data.get_feed_dict(idx,False))
        real=data.get_feed_dict(idx,False)[Y]
        for ii in range(BATCH_SIZE):
            #print('x_dict:',xval[ii:ii+1,0:1,:,:].flatten())
            print('prediction:',prediction[ii],'real:',real[ii])