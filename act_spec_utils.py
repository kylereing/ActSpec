import torch

def binarize(tensor, flag=True):
        tensor_copy = tensor.clone().detach()
        tensor_copy[tensor_copy>0.0]=1.0
        tensor_copy[tensor_copy<=0.0] = 0.0
        pos_neg = 2*tensor_copy - 1

        if flag:
          return pos_neg
        else:
          return tensor_copy
        
def function_output(logits, ref_class):
  pred_class = logits.argmax(1)
  if torch.equal(pred_class, ref_class):
    return torch.tensor([1.0]).float()
  else:
    return torch.tensor([-1.0]).float()
  
def compute_unnorm_inner(x, y):
    return torch.sum(torch.mul(x, y))
  
def compute_norm_inner(x, y):
    return torch.mean(torch.mul(x, y))

def chi(s, x):
  reduction = torch.multiply(s,x)
  reduction[reduction==0.0] = 1.0
  chi_s = torch.prod(reduction)

  return chi_s

def flip_bit(sample, i):
  flip = sample.clone().detach()
  if flip[0, i] == 0.0:
    flip[0, i] = 1.0
  else:
    flip[0, i] = 0.0
  return flip

def store_activations(net, sample, layer_id):
  activation = {}

  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook

  for name, module in net.named_modules():
    if layer_id in name:
      handle = module.register_forward_hook(get_activation(name))

  out = net(sample)
  act = activation[layer_id]
  handle.remove()

  return act


def modify_activations(net, layer_id, mod_input, sample):

      def set_activation(inp_tensor):
          def hook(model, input, output):
              output = inp_tensor
          return hook

      for name, module in net.named_modules():
        if layer_id in name:
          handle = module.register_forward_hook(set_activation(mod_input))

      out1 = net(sample)
      handle.remove()
      return out1
  
  
