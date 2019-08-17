import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import shutil
import os

class Model:
    def __init__(self, batch_size=32, learning_rate=1e-3, reg_lambda=1e-3):
        self.batch_size = batch_size
        self.lr_rate = learning_rate
        self.reg_lambda =  reg_lambda
        tf.reset_default_graph()

    @staticmethod
    def generate(x, y):
        data_len = x.shape[0]
        idx = np.arange(data_len)
        np.random.shuffle(idx)
        for i in idx:
            yield x[i], y[i]
    
    @staticmethod
    def get_batch(gen, batch_size=20000):
        x_batch, y_batch, isEndOfList = [],[], False
        for i in range(batch_size):
            try:
                x, y = gen.__next__()
                x_batch.append(x)
                y_batch.append(y)
            except StopIteration:
                if i == 0:
                    isEndOfList = True
                break
        return x_batch, y_batch, isEndOfList
    
    @staticmethod
    def kFoldValidationSet(data, k=10):
        while True:
            kf = KFold(n_splits=k)
            for train_ind, validation_ind in kf.split(data):
                yield train_ind, validation_ind
    @staticmethod
    def hyperparam_tune(model, train_input, train_output, test_input, test_output):
        best_lr_rate, best_reg = 0.0, 0.0
        best_acc = 100000.0
        lr_rates = [1e-2, 6e-2, 4e-1, 2e-1, 1e-1]
        reg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for lr in lr_rates:
            for reg_lambda in reg:
                tf.reset_default_graph()
                try:
                    shutil.rmtree('tmp')
                except Exception as err:
                    print(err)
                model.train(train_input, train_output)
                acc = model.evaluate(test_input, test_output)
                if acc < best_acc:
                    best_acc = acc
                    best_lr_rate = lr
                    best_reg = reg_lambda
                    print('best_acc={:f} best_lr_rate={:f} best_reg={:f}'.format(best_acc, best_lr_rate, best_reg))
        return best_acc, best_lr_rate, best_reg

    def create(self, input_dim, output_dim):
        hidden_dim = 100
        
        self.x = tf.placeholder(tf.float32, (None, input_dim), name='input_placeholder')
        self.y = tf.placeholder(tf.float32, (None, output_dim), name='output_placeholder')
        self.rate = tf.placeholder(tf.float32, (), name='dropout_prob_placeholder')
        weights, biases = {}, {}

        weights['W0'] = tf.get_variable(name='W0', shape=[input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        weights['W1'] = tf.get_variable(name='W1', shape=[hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        weights['W2'] = tf.get_variable(name='W2', shape=[hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        weights['W3'] = tf.get_variable(name='W3', shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())

        biases['b0'] = tf.get_variable(name='b0', shape=[hidden_dim], initializer=tf.constant_initializer(0.))
        biases['b1'] = tf.get_variable(name='b1', shape=[hidden_dim], initializer=tf.constant_initializer(0.))
        biases['b2'] = tf.get_variable(name='b2', shape=[hidden_dim], initializer=tf.constant_initializer(0.))
        biases['b3'] = tf.get_variable(name='b3', shape=[output_dim], initializer=tf.constant_initializer(0.))

        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
        
        # create computation graph
        inp = self.x
        regularized_loss =0.0
        for w, b, activation, ind in zip(weights, biases, activations, range(len(activations))):
            out = tf.add(tf.matmul(inp, weights[w]), biases[b], name='out'+str(ind) if activation else 'pred')
            
            if activation is not None: 
                inp = tf.nn.dropout(activation(out), rate=self.rate, name='act'+str(ind))
            
            regularized_loss += tf.nn.l2_loss(weights[w])
        
        self.pred = tf.identity(out, name='prediction')
        self.mse_loss = tf.math.add(tf.losses.mean_squared_error(self.y, self.pred), self.reg_lambda * regularized_loss, name='regularized_loss')
        

    def train(self, input_data, output_data, num_epochs = 200, env_name='random'):
        print('Training for {:s}'.format(env_name))
        CKPT_DIR='tmp/'+env_name+'/'
        CKPT_FILE = 'model.ckpt'

        loss_summary = tf.summary.scalar(name="loss", tensor=self.mse_loss)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.lr_rate).minimize(self.mse_loss, global_step=self.global_step)
        vali_set_gen = self.kFoldValidationSet(input_data)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            try:
                shutil.rmtree('./logs/train')
                shutil.rmtree('./logs/val')
            except Exception as err:
                print(err)
            train_writer = tf.summary.FileWriter(logdir="./logs/train", graph=sess.graph)
            val_writer = tf.summary.FileWriter(logdir="./logs/val", graph=sess.graph)
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for i in range(num_epochs):
                train_indices, val_indices = vali_set_gen.__next__()
                train_gen, val_gen = self.generate(input_data[train_indices], output_data[train_indices]), self.generate(input_data[val_indices], output_data[val_indices])
                x_batch, y_batch, isEndOfList = self.get_batch(train_gen)
                while not isEndOfList:
                    _, l, l_summary = sess.run([self.optimizer, self.mse_loss, loss_summary], feed_dict={self.x: x_batch, self.y: y_batch, self.rate:0.0})
                    x_batch, y_batch, isEndOfList = self.get_batch(train_gen)

                train_writer.add_summary(l_summary, global_step=i)
                #print('Epoch number={:d} Training Loss={:f}'.format(i,l))
                if i % 100 == 0 or i == num_epochs - 1:
                    x_batch, y_batch, isEndOfList = self.get_batch(val_gen)
                    while not isEndOfList:
                        val_l, val_l_summary = sess.run([self.mse_loss, loss_summary], feed_dict={self.x: x_batch, self.y: y_batch, self.rate:0})
                        x_batch, y_batch, isEndOfList = self.get_batch(val_gen)
                    val_writer.add_summary(val_l_summary, global_step=i)
                    print('{:*^100}'.format('Epoch number={:d} Validation Loss={:f}'.format(i,val_l)))
            os.makedirs(os.path.dirname(CKPT_DIR), exist_ok=True)
            saver.save(sess, CKPT_DIR+CKPT_FILE, global_step=self.global_step)
    
    def evaluate(self, test_data, test_labels, env_name='random'):
        CKPT_DIR='tmp/'+env_name+'/'
        CKPT_FILE = 'model.ckpt'

        with tf.Session() as sess:
            try:
                shutil.rmtree('./logs/test')
            except Exception as err:
                print(err)
            #restoring the model
            saver = tf.train.Saver()
            saver.restore(sess, CKPT_DIR+CKPT_FILE)

            test_writer = tf.summary.FileWriter(logdir="./logs/test", graph=sess.graph)
            test_loss_summary = tf.summary.scalar(name="loss", tensor=self.mse_loss)
            test_l, test_l_summary = sess.run([self.mse_loss, test_loss_summary], feed_dict={self.x: test_data, self.y: test_labels, self.rate:0})
            test_writer.add_summary(test_l_summary)
            return test_l

    def predict_helper(self, x, rate, op):
        def predict(data):
            sess=tf.get_default_session()
            return sess.run(op, feed_dict={x: data, rate:0})
        return predict

    
    def load_trained_policy(self, env_name):
        CKPT_DIR='tmp'+env_name+'/'
        CKPT_FILE = 'model.ckpt'

        sess = tf.get_default_session()
        imported_meta = tf.train.import_meta_graph(CKPT_DIR+CKPT_FILE+'.meta')
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_DIR+CKPT_FILE)
        graph = tf.get_default_graph()
        #get the placeholder and operation to evaluate
        x = graph.get_tensor_by_name("input_placeholder:0")
        rate = graph.get_tensor_by_name("dropout_prob_placeholder:0")
        op = graph.get_tensor_by_name("prediction:0")
        return self.predict_helper(x, rate, op)

            


                    








