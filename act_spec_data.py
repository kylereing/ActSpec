import os
import torch
import numpy as np
import tensorflow.python.keras
from keras.src.datasets import mnist
from torchvision import datasets, transforms as T

class ActSpecData:
  
  def __init__(self, data, ref_class=None, data_store=None):
    self.samples = data
    self.total_samp = self.samples.size()[0]
    self.n = self.samples.size()[1]
    
    if len(self.samples.size()) == 3:
      self.n = self.n * self.samples.size()[2]
      self.samples = torch.reshape(data, (self.total_samp, self.n))
      
    self.ref_class = ref_class
    self.data_store = data_store
    self.net = None
    self.activations = None
      

def retrieve_data(data_flag, params=None):

  data = None
  ref_class = None
  data_path = 'G:/My Drive/data/'

  if data_flag == 'LR_MNIST':

    class DataStore:
      def __init__(self):
          # read the original MNIST data: test and train

          (self.X_train, self.y_train), (self.X_test, self.y_test) =\
              mnist.load_data()

          self.train28x28 = self.X_train  # alias
          self.train14x14 = None
          self.train7x7 = None
          self.test28x28 = self.X_test    # alias
          self.test14x14 = None
          self.test7x7 = None

          self.fname_train14x14 = 'train14x14.npy'
          self.fname_train7x7 = 'train7x7.npy'

          self.fname_test14x14 = 'test14x14.npy'
          self.fname_test7x7 = 'test7x7.npy'

          self.train14x14 = np.load(data_path + self.fname_train14x14)
          self.test14x14 = np.load(data_path + self.fname_test14x14)

    data_store = DataStore()
    data = torch.from_numpy(data_store.train14x14).float() #samp x 14 x 14
    
    if params:
      ref_class = params[0]
    data = ActSpecData(data, ref_class=ref_class, data_store=data_store)

  ################################################################################

  elif data_flag == 'MNIST':

    data_root='/tmp/public_dataset/pytorch'
    root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))

    dataset = datasets.MNIST(root=root, train=True, download=True,
                            transform=T.transforms.Compose([
                                  T.transforms.ToTensor(),
                                  T.transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    data = torch.zeros(len(dataset),28,28) #samp x 28 x 28
    for i in range(len(dataset)):
      samp, _ = dataset[i]
      data[i,:,:] = samp
      
    if params:
      ref_class = params[0]
      
    data = ActSpecData(data, ref_class)
      
  return data