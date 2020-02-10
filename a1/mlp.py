"""
Finn Womack
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import pickle
except:
   import _pickle as pickle


import numpy as np
import math
import matplotlib.pyplot as plt

color_list = ["#984ea3",
                "#ff7f00",
                "#4daf4a",
                "#e41a1c",
                "#377eb8",
                "#ffff33",
                "#a65628",
                "#f781bf",
                "#999999"]

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
    # DEFINE __init function
        self.W = W
        self.W_update = np.zeros(W.shape)
        self.b = b
        self.b_update = np.zeros(b.shape)
        self.w_grads = np.zeros(shape=(W.shape[0], W.shape[1]), dtype=float)
        self.b_grads = np.zeros(shape=(W.shape[1]), dtype=float)

    def forward(self, x):
    # DEFINE forward function
        self.x = x
        self.ans=np.dot(self.x,self.W) + self.b
        return self.ans

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        x_grad = np.dot(grad_output, self.W.T)
        w_grad = np.dot(self.x.T, grad_output)
        b_grad = np.sum(grad_output, axis=0)
        self.w_grads = momentum*self.w_grads + w_grad
        self.b_grads = momentum*self.b_grads + b_grad
        self.W_update = self.W_update - learning_rate*self.w_grads - l2_penalty*self.W
        self.b_update = self.b_update - learning_rate*self.b_grads
        return x_grad

    def update(self):
        self.W = self.W + self.W_update
        self.b = self.b + self.b_update
        self.W_update = np.zeros(self.W.shape)
        self.b_update = np.zeros(self.b.shape)

# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
    # DEFINE forward function
        self.ans = np.maximum(x,0)
        return self.ans

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        new_grad = np.sign(self.ans)
        return np.multiply(new_grad,grad_output)
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x, y):
        # DEFINE forward function
        sig = lambda a: 1/(1+np.exp(-a))
        self.y = y
        self.sig = sig(x)
        self.loss = -np.multiply(y,np.log(self.sig))-np.multiply((1-y),np.log(1-self.sig))
        return self.loss

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
        # DEFINE backward function
        new_grad = self.y - self.sig
        return -np.multiply(new_grad,grad_output)
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units

        W = np.random.normal(0,.01,(input_dims,hidden_units))
        b = np.random.normal(0,.01,(1,hidden_units))
        W2 = np.random.normal(0,.01,(hidden_units,1))
        b2 = np.random.normal(0,.01,(1,1))
        self.lt = LinearTransform(W,b)
        self.r = ReLU()
        self.lt2 = LinearTransform(W2,b2)
        self.sc = SigmoidCrossEntropy()

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty
    ):
        n = len(y_batch)
        self.n = n
        f1 = self.lt.forward(x_batch)
        f2 = self.r.forward(f1)
        f3 = self.lt2.forward(f2)
        loss = self.sc.forward(f3,y_batch)
        b1 = self.sc.backward(1, 
            learning_rate=learning_rate,
            momentum=momentum,
            l2_penalty=l2_penalty)
        b2 = self.lt2.backward(b1, 
            learning_rate=learning_rate,
            momentum=momentum,
            l2_penalty=l2_penalty)
        b3 = self.r.backward(b2[0], 
            learning_rate=learning_rate,
            momentum=momentum,
            l2_penalty=l2_penalty)
        b4 = self.lt.backward(b3, 
            learning_rate=learning_rate,
            momentum=momentum,
            l2_penalty=l2_penalty)
        self.f3 = f3
        self.lt.update()
        self.lt2.update()
        return loss



    def evaluate(self, x, y):
        acc = 0.0
        avg_loss = 0.0
        n = len(y)
        x2 = self.lt.forward(x)
        x3 = self.r.forward(x2)
        x4 = self.lt2.forward(x3)
        loss = self.sc.forward(x4,y)
        avg_loss = np.sum(loss,0)[0]
        pred = np.floor(self.sc.sig + 0.5)
        acc = np.sum(1-np.absolute(y - pred),0)[0]
        acc = acc/n
        avg_loss = avg_loss/n
        return acc, avg_loss

    def predict(self, x):
        x2 = self.lt.forward(x)
        x3 = self.r.forward(x2)
        x4 = self.lt2.forward(x3)
        sig = lambda a: 1/(1+np.exp(-a))
        x_sig = sig(x4)
        return np.floor(x_sig + 0.5)
        

# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action='store_true', help="Add if you want to generate the plots")

    args = parser.parse_args()
    
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        with open('cifar_2class_py2.p', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

    try:
        xrange
    except NameError:
        xrange = range

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    num_examples, input_dims = train_x.shape
    np.random.seed(11235813)
    idx = np.array(range(len(train_x)))
    np.random.shuffle(idx)
    train_mean = np.mean(train_x,0)
    train_x = train_x - train_mean
    test_x = test_x - train_mean
    train_std = np.std(train_x,0)
    train_x = train_x / train_std
    test_x = test_x / train_std
    num_epochs = 50
    num_batches = 5000
    hidden_units = 40
    mlp = MLP(input_dims, hidden_units)

    learning_rate = 0.00001
    momentum = 0.85
    l2_penalty = 0.0000001

    train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
    test_accuracy, test_loss = mlp.evaluate(test_x, test_y)

    print()
    print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
        train_loss,
        100. * train_accuracy,
    ))
    print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        test_loss,
        100. * test_accuracy,
    ))

    # l_max = [0.467,0.6595,0.7545,0.7755,0.6855,0.626]
    # lr_list = [0.01,0.001,0.0001,0.00001,0.000001,0.0000001]

    # plt.figure()
    # plt.plot(np.arange(len(lr_list)), l_max, label="Best Test Accuracy", color=color_list[0]) 
    # plt.title("Training Accuracy for different learning rates")
    # plt.xticks(np.arange(len(lr_list)),lr_list)
    # plt.xlabel("Learning Rate")
    # plt.ylabel("Testing Accuracy")
    # plt.legend()
    # plt.savefig("lr_plot_best.png", dpi=100)

    # b_max = [0.675,0.6685,0.6755,0.75,0.7775,0.813]
    # b_list = [10,50,100,500,1000,5000]

    # plt.figure()
    # plt.plot(b_list, b_max, label="Best Test Accuracy", color=color_list[0]) 
    # plt.title("Training Accuracy for different numbers of batches")
    # plt.xlabel("Batches")
    # plt.ylabel("Testing Accuracy")
    # plt.legend()
    # plt.savefig("b_plot_best.png", dpi=100)

    # h_max = [0.7485,0.756,0.7695,0.7755,0.7795,0.7825]
    # h_list = [5,10,20,40,80,160]

    # plt.figure()
    # plt.plot(h_list, h_max, label="Best Test Accuracy", color=color_list[0]) 
    # plt.title("Training Accuracy for different numbers of hidden units")
    # plt.xlabel("Hidden Units")
    # plt.ylabel("Testing Accuracy")
    # plt.legend()
    # plt.savefig("h_plot_best.png", dpi=100)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    for epoch in xrange(num_epochs):
        train_x = train_x[idx]
        train_y = train_y[idx]
        np.random.shuffle(idx)
        learning_rate = learning_rate*0.99999

        for b in xrange(num_batches):
            total_loss = 0.0
            start = math.ceil((b+0.)*num_examples/num_batches)
            end = math.ceil((b+1.)*num_examples/num_batches)
            n = end-start
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]
            loss = mlp.train(x_batch, 
                y_batch, 
                learning_rate, 
                momentum,
                l2_penalty)
            total_loss = np.sum(loss,0)[0]
            avg_loss = total_loss/n
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    avg_loss,
                ),
                end='',
            )
            sys.stdout.flush()
        train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
        test_accuracy, test_loss = mlp.evaluate(test_x, test_y)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))

    if args.plot:
        plt.figure()
        plt.plot(np.arange(len(train_accuracies)), train_accuracies, label="Train Accuracy", color=color_list[0]) 
        plt.plot(np.arange(len(test_accuracies)), test_accuracies, label="Test Accuracy", color=color_list[1])
        plt.title("Accuracy for Each Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("epoch_acc.png", dpi=100)

        plt.figure()
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss", color=color_list[0]) 
        plt.plot(np.arange(len(test_losses)), test_losses, label="Test Loss", color=color_list[1])
        plt.title("Loss for Each Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("epoch_loss.png", dpi=100)

        # Test learning rate
        l_max = [0.0,0.0,0.0,0.0,0.0,0.0]
        l_final = [0.0,0.0,0.0,0.0,0.0,0.0]
        lr_list = [0.01,0.001,0.0001,0.00001,0.000001,0.0000001]

        i = 0
        print("learning rates:")
        for lr in lr_list:

            num_epochs = 50
            num_batches = 1000
            hidden_units = 40
            mlp = MLP(input_dims, hidden_units)

            learning_rate = lr
            momentum = 0.85
            l2_penalty = 0.0000001
            for epoch in xrange(num_epochs):
                train_x = train_x[idx]
                train_y = train_y[idx]
                np.random.shuffle(idx)
                learning_rate = learning_rate*0.99999

                for b in xrange(num_batches):
                    total_loss = 0.0
                    start = math.ceil((b+0.)*num_examples/num_batches)
                    end = math.ceil((b+1.)*num_examples/num_batches)
                    n = end-start
                    x_batch = train_x[start:end]
                    y_batch = train_y[start:end]
                    loss = mlp.train(x_batch, 
                        y_batch, 
                        learning_rate, 
                        momentum,
                        l2_penalty)
                train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
                test_accuracy, test_loss = mlp.evaluate(test_x, test_y)
                
                if test_accuracy > l_max[i]:
                    l_max[i] = test_accuracy
            print(lr_list[i], ": ", l_max[i])
            i+=1
        
        i=0
        plt.figure()
        plt.plot(np.arange(len(lr_list)), l_max, label="Best Test Accuracy", color=color_list[0]) 
        plt.title("Training Accuracy for different learning rates")
        plt.xticks(np.arange(len(lr_list)),lr_list)
        plt.xlabel("Learning rate")
        plt.ylabel("Testing Accuracy")
        plt.legend()
        plt.savefig("lr_plot.png", dpi=100)

        # Test batch sizes
        b_max = [0.0,0.0,0.0,0.0,0.0,0.0]
        b_list = [10,50,100,500,1000,5000]
        print("batches:")
        i = 0
        for bat in b_list:

            num_epochs = 50
            num_batches = bat
            hidden_units = 40
            mlp = MLP(input_dims, hidden_units)

            learning_rate = 0.00001
            momentum = 0.85
            l2_penalty = 0.0000001
            for epoch in xrange(num_epochs):
                train_x = train_x[idx]
                train_y = train_y[idx]
                np.random.shuffle(idx)
                learning_rate = learning_rate*0.99999

                for b in xrange(num_batches):
                    total_loss = 0.0
                    start = math.ceil((b+0.)*num_examples/num_batches)
                    end = math.ceil((b+1.)*num_examples/num_batches)
                    n = end-start
                    x_batch = train_x[start:end]
                    y_batch = train_y[start:end]
                    loss = mlp.train(x_batch, 
                        y_batch, 
                        learning_rate, 
                        momentum,
                        l2_penalty)
                train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
                test_accuracy, test_loss = mlp.evaluate(test_x, test_y)
                
                if test_accuracy > b_max[i]:
                    b_max[i] = test_accuracy
            print(b_list[i], ": ", b_max[i])
            i+=1
            
        i=0
        plt.figure()
        plt.plot(b_list, b_max, label="Best Test Accuracy", color=color_list[0]) 
        plt.title("Training Accuracy for different batch sizes")
        plt.xlabel("Batch Size")
        plt.ylabel("Testing Accuracy")
        plt.legend()
        plt.savefig("b_plot.png", dpi=100)

        # Test hidden units
        h_max = [0.0,0.0,0.0,0.0,0.0,0.0]
        h_list = [5,10,20,40,80,160]
        print("hidden units:")
        i = 0
        for h in h_list:

            num_epochs = 50
            num_batches = 1000
            hidden_units = h
            mlp = MLP(input_dims, hidden_units)

            learning_rate =0.00001 
            momentum = 0.85
            l2_penalty = 0.0000001
            for epoch in xrange(num_epochs):
                train_x = train_x[idx]
                train_y = train_y[idx]
                np.random.shuffle(idx)
                learning_rate = learning_rate*0.99999

                for b in xrange(num_batches):
                    total_loss = 0.0
                    start = math.ceil((b+0.)*num_examples/num_batches)
                    end = math.ceil((b+1.)*num_examples/num_batches)
                    n = end-start
                    x_batch = train_x[start:end]
                    y_batch = train_y[start:end]
                    loss = mlp.train(x_batch, 
                        y_batch, 
                        learning_rate, 
                        momentum,
                        l2_penalty)
                train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
                test_accuracy, test_loss = mlp.evaluate(test_x, test_y)
                

                # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
                if test_accuracy > h_max[i]:
                    h_max[i] = test_accuracy
            print(h_list[i], ": ", h_max[i])
            i+=1
            
        i=0
        plt.figure()
        plt.plot(h_list, h_max, label="Best Test Accuracy", color=color_list[0]) 
        plt.title("Training Accuracy for different numbers of hidden units")
        plt.xlabel("Hidden Units")
        plt.ylabel("Testing Accuracy")
        plt.legend()
        plt.savefig("h_plot.png", dpi=100)
