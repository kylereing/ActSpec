import torch

def process_data(preproc_flag, data_obj, params = None):
  
  proc_data = None
  data = data_obj.samples
  
  if preproc_flag == 'TWO_DIGIT':
    
    digit_one = []
    digit_two = []
    ref_one = torch.tensor([7]).float()
    ref_two = torch.tensor([1]).float()
    
    assert (len(params) >= 1), 'Must pass a model object'
    model = params[-1]
    orig_shape = data.size()
    
    if len(params) >= 3:
      ref_one = params[0]
      ref_two = params[1]
    
    for i in range(data_obj.total_samp):
      
      sample = torch.flatten(data[i, ...])
      logits = model(sample[None, ...])
      pred_class = logits.argmax(1)
      
      if torch.equal(pred_class, ref_one):
        digit_one.append(sample[None, ...])
        
      if torch.equal(pred_class, ref_two):
        digit_two.append(sample[None, ...])
        
    even_split = min(len(digit_one), len(digit_two))
    processed_samples = digit_one[:even_split] + digit_two[:even_split]
    proc_data = torch.cat(processed_samples, dim=0)
    
    data_obj.samples = proc_data
    data_obj.total_samp = 2*even_split
    
  return data_obj