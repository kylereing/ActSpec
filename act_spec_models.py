import numpy as np  # np.max
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from keras import utils

import torch.nn as nn
from collections import OrderedDict
import torch

class NetworkObj:
    
    def __init__(self, network, layer_flag='INPUT', layer_id=None, layer_size=None):
        self.network = network
        self.layer_flag = layer_flag
        self.layer_id = layer_id
        self.layer_size = layer_size
        

def retrieve_model(model_flag, params):
    
    network = None
    
    if model_flag == 'MLP':
        #param format: (input_dim, n_hidden, n_class)
         
        class MLP(nn.Module):
            def __init__(self, input_dims, n_hiddens, n_class):
                super(MLP, self).__init__()
                assert isinstance(input_dims, int), 'Please provide int for input_dims'
                self.input_dims = input_dims
                current_dims = input_dims
                layers = OrderedDict()

                if isinstance(n_hiddens, int):
                    n_hiddens = [n_hiddens]
                else:
                    n_hiddens = list(n_hiddens)
                for i, n_hidden in enumerate(n_hiddens):
                    layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
                    layers['relu{}'.format(i+1)] = nn.ReLU()
                    layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
                    current_dims = n_hidden
                layers['out'] = nn.Linear(current_dims, n_class)

                self.model= nn.Sequential(layers)

            def forward(self, input):
                input = input.view(input.size(0), -1)
                assert input.size(1) == self.input_dims
                return self.model.forward(input)
            
        network = MLP(params[0], params[1], params[2])
        
    elif model_flag == 'MLP_NO_DROPOUT':
        
        class MLP_NO_DROP(nn.Module):
            def __init__(self, input_dims, n_hiddens, n_class):
                super(MLP_NO_DROP, self).__init__()
                assert isinstance(input_dims, int), 'Please provide int for input_dims'
                self.input_dims = input_dims
                current_dims = input_dims
                layers = OrderedDict()

                if isinstance(n_hiddens, int):
                    n_hiddens = [n_hiddens]
                else:
                    n_hiddens = list(n_hiddens)
                for i, n_hidden in enumerate(n_hiddens):
                    layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
                    layers['relu{}'.format(i+1)] = nn.ReLU()
                    current_dims = n_hidden
                layers['out'] = nn.Linear(current_dims, n_class)

                self.model= nn.Sequential(layers)

            def forward(self, input):
                input = input.view(input.size(0), -1)
                assert input.size(1) == self.input_dims
                return self.model.forward(input)
            
        network = MLP_NO_DROP(params[0], params[1], params[2])

    elif model_flag == 'LOW_RES': 
        #param format: (data)
        class ModelRes:
            def __init__(self, resolution, x_train, y_train,
                        x_test, y_test):
                """resolution and data are inherently connected
                """
                self.resolution = resolution

                self.result_dir = 'result'
                if not os.path.exists(self.result_dir):
                    os.mkdir(self.result_dir)

                self.result_fname = 'keras_mnist' + str(self.resolution) + '.h5'
                self.model_path = os.path.join(self.result_dir, self.result_fname)

                # 1) reshape data to (m, n)
                # 2) normalize data
                self.x_train = x_train.reshape(x_train.shape[0], -1) / 255
                self.x_train = self.x_train.astype('float32')
                self.x_test = x_test.reshape(x_test.shape[0], -1) / 255
                self.x_test = self.x_test.astype('float32')

                # encode the labels like 3 --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                n_classes = 10
                self.y_train = utils.to_categorical(y_train, n_classes)
                self.y_test = utils.to_categorical(y_test, n_classes)

                # Model: sequential

                # hidden layer, activation: relu
                n_nodes = 256
                self.model = Sequential()
                self.model.add(Dense(n_nodes,
                                    input_shape=(self.resolution*self.resolution,)))
                self.model.add(Activation('relu'))
                self.model.add(Dropout(0.2))

                # output layer, activation: softmax
                self.model.add(Dense(n_classes))
                self.model.add(Activation('softmax'))

                # compile the model
                self.model.compile(loss='categorical_crossentropy',
                                metrics=['accuracy'], optimizer='adam')

            def train(self):
                """Train the model.
                Currently parameters batch_size and the number of epochs are hardcoded.
                """
                print('.. Training model for resolution {}x{}'.format(
                    self.resolution, self.resolution
                ))

                print('self.x_train.shape', self.x_train.shape)
                print('self.y_train.shape', self.y_train.shape)

                history = None
                print('model file:', self.model_path)
                if not os.path.exists(self.model_path):
                    # NB: batch_size is irrelevant in our case, use single thread
                    history = self.model.fit(self.x_train, self.y_train,
                                            batch_size=128, epochs=10,
                                            verbose=2,
                                            validation_data=(self.x_test, self.y_test))

                    self.model.save(self.model_path)
                else:
                    print('read the model from the disk')
                    self.model = load_model(self.model_path)

                if history:
                    # plotting the metrics
                    plt.figure()
                    plt.subplot(2, 1, 1)
                    plt.plot(history.history['acc'])
                    plt.plot(history.history['val_acc'])
                    plt.title('model accuracy ' +
                            str(self.resolution) + 'x' + str(self.resolution))
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='lower right')

                    plt.subplot(2, 1, 2)
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss ' +
                            str(self.resolution) + 'x' + str(self.resolution))

                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='upper right')

                    plt.tight_layout()

                plt.figure()
                predicted_proba = self.model.predict_proba(self.x_test)
                p_max = np.max(predicted_proba, axis=1)
                plt.hist(p_max, bins=40)  # ignore histogram content
                plt.title('test model: max probability, {}x{}'.format(
                    self.resolution, self.resolution))
        
        low_res_data = params[0].data_store
        model_path = 'G:/My Drive/data/keras_mnist14.h5'
        low_res_model = ModelRes(14,
                         low_res_data.train14x14,
                         low_res_data.y_train,
                         low_res_data.test14x14,
                         low_res_data.y_test)
        low_res_model.model = load_model(model_path)
        low_res_weights = low_res_model.model.get_weights()
        
        proxy_mlp = retrieve_model('MLP', [196, 256, 10])
        proxy_mlp.model.fc1.weight.data = torch.from_numpy(np.transpose(low_res_weights[0]))
        proxy_mlp.model.fc1.bias.data = torch.from_numpy(np.transpose(low_res_weights[1]))
        proxy_mlp.model.out.weight.data = torch.from_numpy(np.transpose(low_res_weights[2]))
        proxy_mlp.model.out.bias.data = torch.from_numpy(np.transpose(low_res_weights[3]))
        
        network = proxy_mlp
        
    elif model_flag == 'OPT125':
        # Load model directly
        from transformers import AutoModelForCausalLM

        network = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        
    return network