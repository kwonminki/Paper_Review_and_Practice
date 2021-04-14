import os

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import numpy as np

from model import *


class Trainer():
    def __init__(self, opt, job_name, support=5, pretrain=True, lr_multiplier=True):
        self.opt = opt
        self.job_name = job_name
        self.pretrain = pretrain
        self.base_class = opt['base_class']
        self.novel_class = opt['novel_class']

        if self.pretrain:
            self.model = Net(self.base_class)
            lr1 = 0.0001
        else:
            self.model = Net(self.base_class + self.novel_class)
            lr1 = 0.00001 # 0.0001

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr1, momentum=0.9)
        if lr_multiplier :
            self.optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr1 * 10, momentum=0.9)
        else:
            self.optimizer2 = None

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                    reduction= tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.checkpoint_path = os.path.join(opt['checkpoint_dir'],job_name)
        self.checkpoint_path_ft = os.path.join(opt['checkpoint_dir'],job_name+"_ft{}".format(support))
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.checkpoint_path_ft):
            os.makedirs(self.checkpoint_path_ft)
    

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            pred = self.model(data[0])
            loss = self.loss_fn(data[1], pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if self.optimizer2 is None:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        else:
            var1 = self.model.extractor.trainable_variables
            var2 = self.model.trainable_variables[len(var1):]

            grads1 = gradients[:len(var1)]
            grads2 = gradients[len(var1):]
            self.optimizer.apply_gradients(zip(grads1, var1))
            self.optimizer2.apply_gradients(zip(grads2, var2))
        
        self.train_loss(loss)

        return self.train_loss.result(), pred

    @tf.function
    def train_step_(self, data):
        with tf.GradientTape() as tape:
            pred = self.model(data[0])
            loss = self.loss_fn(data[1], pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.train_loss(loss)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self.train_loss.result(), pred

    @tf.function
    def train_step_classifier(self, data):
        with tf.GradientTape() as tape:
            pred = self.model.classify(data[0])
            loss = self.loss_fn(data[1], pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.train_loss(loss)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self.train_loss.result(), pred


    @tf.function
    def test_step(self, data):
        predictions = self.model(data[0])
        return predictions

    def imprint(self, data_loader, random=False):
        random = False
        if not random:
            for batch_idx, (input, target) in enumerate(data_loader): 
                output = self.model.extract(input)
                if batch_idx == 0:
                    output_stack = output
                    target_stack = target
                else:
                    output_stack = tf.concat([output_stack, output], axis=0)
                    target_stack = tf.concat([target_stack, target], axis=0)
    
            new_weight = []
            for i in range(self.novel_class):
                tmp = tf.math.reduce_mean(output_stack[target_stack==(i+self.base_class)], axis=0)
                tmp = tf.math.l2_normalize(tmp)
                new_weight.append(tmp)
            new_weight = tf.stack(new_weight, axis=1)
            # print(new_weight.shape)
        else:
            new_weight = tf.random.uniform([256, 100])

        self.model.build(input_shape=(None,224,224,3))
        weight = self.model.classifier.get_weights()
        new_weight = tf.concat((weight[0], new_weight), axis=1)
    


        self.model.classifier =  Dense(self.base_class+self.novel_class, kernel_initializer='he_uniform'\
                                        ,use_bias=False)   
        self.model.build(input_shape=(None, 224, 224, 3))
        if not random:
            self.model.classifier.set_weights([new_weight])
        self.save_checkpoint(0, imprinting=True)

    def lr_scheduling(self, epoch):
        step_size = 4
        decay_rate = 0.94
        
        if (epoch + 1) % step_size == 0:
            current_lr = self.optimizer._decayed_lr(tf.float32) #self.optimizer.learning_rate.numpy()
            updated = current_lr * decay_rate
            self.optimizer.lr.assign(updated)
            updated = self.optimizer._decayed_lr(tf.float32) 
            print("OPTIMIZER | current lr: {} |updated: {}".format(current_lr, updated))
            
            if self.optimizer2 is not None:
                current_lr = self.optimizer2._decayed_lr(tf.float32) #self.optimizer.learning_rate.numpy()
                updated = current_lr * decay_rate
                self.optimizer2.lr.assign(updated)
                updated = self.optimizer2._decayed_lr(tf.float32) 
                print("OPTIMIZER2 | current lr: {} |updated: {}".format(current_lr, updated))                

    def imprint_(self, data_loader):
        for batch_idx, (input, target) in enumerate(data_loader): 
            output = self.model.extract(input)
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = tf.concat([output_stack, output], axis=0)
                target_stack = tf.concat([target_stack, target], axis=0)
 
        new_weight = []
        for i in range(100):
            tmp = tf.math.reduce_mean(output_stack[target_stack==(i+100)], axis=0)
            tmp = tf.math.l2_normalize(tmp)
            new_weight.append(tmp)
        new_weight = tf.stack(new_weight, axis=1)

        self.model.build(input_shape=(None,224,224,3))
        weight = self.model.classifier.get_weights()
        new_weight = tf.concat((weight[0], new_weight), axis=1)

        self.model.classifier =  Dense(200, use_bias=False)   
        self.model.build(input_shape=(None, 224, 224, 3))
        self.model.classifier.set_weights([new_weight])
        self.save_checkpoint(0, imprinting=True)

    def get_predictions(self, data_loader):
        for i, data in enumerate(data_loader):
            pred = self.test_step(data)
            pred = pred.numpy()
            if i == 0:
                predictions= pred
                labels = data[1]
            else:
                predictions = np.concatenate([predictions, pred], axis=0)
                labels = np.concatenate([labels, data[1]], axis=0)

        return predictions, labels

    def validate(self, data_loader):
        acc = []
        acc_novel = []

        for i, data in enumerate(data_loader):
            pred = self.test_step(data)
            pred = pred.numpy()
            pred = pred.argmax(axis=1)


            if( len(data[1][data[1]>=100])>0 ):
                acc = np.concatenate([acc, (pred[data[1]<100]==data[1][data[1]<100])], axis=0)                
                acc_novel = np.concatenate([acc_novel, (pred[data[1]>=100]==data[1][data[1]>=100])], axis=0)
            else:
                acc = np.concatenate([acc, (pred==data[1])], axis=0)

            
        acc_all = np.concatenate([acc, acc_novel], axis=0)
        acc = np.mean(acc) * 100
        print("Base Accuracy: ", acc)
        acc_novel = np.mean(acc_novel) * 100
        print("Novel Accuracy: ", acc_novel)
        acc_all = np.mean(acc_all) * 100
        print("Accuracy: ", acc_all)
        return acc_all


            
    def save_checkpoint(self, epoch, imprinting=False):
        if self.pretrain and not imprinting:
            self.model.save_weights(os.path.join(self.checkpoint_path, '{:04d}'.format(epoch)))
        else:
            self.model.save_weights(os.path.join(self.checkpoint_path_ft, '{:04d}'.format(epoch)))
 

    def load_checkpoint(self, epoch):
        if self.pretrain:
            print(os.path.join(self.checkpoint_path, '{:04d}'.format(epoch)))
            self.model.load_weights(os.path.join(self.checkpoint_path, '{:04d}'.format(epoch)))
        else:
            self.model.load_weights(os.path.join(self.checkpoint_path_ft, '{:04d}'.format(epoch)))

