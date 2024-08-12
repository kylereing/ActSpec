#imports
import torch
import numpy as np
import random
import act_spec_utils as acsu
from torch.autograd import Variable

class GoldreichLevin:
     
    def __init__(self, tau=0.1, num_est_samp=1000):
        self.tau = tau
        self.num_est_samp = num_est_samp
         
    def goldreich_levin_weight(self, data_object, subset_tracker):
        k = subset_tracker[0]
        s = subset_tracker[1]
        
        num_vars = data_object.n
        total_samples = data_object.total_samp
        samples = data_object.samples
        ref_class = data_object.ref_class
        net_obj = data_object.net
        
        indicator = np.array([1 if i < k else 0 for i in range(num_vars)])
        ind_conv = 1.0 - indicator

        indicator = torch.from_numpy(indicator).float()
        ind_conv = torch.from_numpy(ind_conv).float()
        estimate = 0.0

        for i in range(self.num_est_samp):
            s1 = samples[random.randint(0,total_samples-1), :]
            s1 = Variable(torch.FloatTensor(s1[None,:]))

            s2 = samples[random.randint(0,total_samples-1), :]
            s2 = Variable(torch.FloatTensor(s2[None,:]))

            s3 = samples[random.randint(0,total_samples-1), :]
            s3 = Variable(torch.FloatTensor(s3[None,:]))
            
            if net_obj.layer_flag == 'INTER':
                old_s1 = s1.clone().detach()
                old_s2 = s2.clone().detach()
                
                s1 = acsu.store_activations(net_obj.network, s1, net_obj.layer_id)
                s2 = acsu.store_activations(net_obj.network, s2, net_obj.layer_id)
                s3 = acsu.store_activations(net_obj.network, s3, net_obj.layer_id)
                
            original_shape = s1.size()

            z = torch.multiply(ind_conv, torch.flatten(s3))
            x1 = torch.multiply(indicator, torch.flatten(s1))
            x1 = torch.add(x1, z)
            bin_x1 = acsu.binarize(x1)
            
            x2 = torch.multiply(indicator, torch.flatten(s2))
            x2 = torch.add(x2, z)
            bin_x2 = acsu.binarize(x2)

            x1 = torch.reshape(x1, original_shape)
            x2 = torch.reshape(x2, original_shape)
            
            if net_obj.layer_flag == 'INTER':
                logits_y1 = acsu.modify_activations(net_obj.network, net_obj.layer_id, x1, old_s1)
                logits_y2 = acsu.modify_activations(net_obj.network, net_obj.layer_id, x2, old_s2)
                
            else:
                logits_y1 = net_obj.network(x1)
                logits_y2 = net_obj.network(x2)

            bin_y1 = acsu.function_output(logits_y1, ref_class)
            bin_y2 = acsu.function_output(logits_y2, ref_class)

            estimate += bin_y1 * acsu.chi(torch.multiply(s,indicator), bin_x1) * \
                        bin_y2 * acsu.chi(torch.multiply(s,indicator), bin_x2)

        estimate /= self.num_est_samp
        return estimate

