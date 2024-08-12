import act_spec_utils as acsu
import torch
import random
import numpy as np
from torch.autograd import Variable
import act_spec_utils as acsu


class ActSpec:
    
    def __init__(self, gamma=0.2, delta=0.2, verbose=True, num_red_samples=1000):
        self.gamma = gamma
        self.delta = delta
        self.verbose = verbose
        self.num_red_samples = num_red_samples

    def checkRedundancy(self, data_obj, prescreen=True, params=None):
        if prescreen:
            return self.precalc_red(data_obj)
        return self.postcalc_red(data_obj, params)
    
    def compute_red_threshold(self, x, y, inds1 = None, inds2 = None):
        if (acsu.compute_norm_inner(x, y)**2) > self.gamma:
            if self.verbose:
                summ = self.gen_summary(inds1, inds2)
                return [True, summ]
            return [True, None]
        return [False, x]

    def precalc_red(self, data_obj):
        verbose_history = []
        const_filter = []
        fo_filter = []
                
        samp = None
        if data_obj.net.layer_flag == 'INTER':
            samp = acsu.store_activations(data_obj.net.network, data_obj.samples, data_obj.net.layer_id)
            data_obj.activations = samp
        else:
            samp = data_obj.samples
            
        #check null
        print('Checking Null')      
        for i in range(data_obj.n):
              
            var_i = acsu.binarize(samp[:, i])
            null_vector = torch.ones_like(var_i)
            is_vari_constant = self.compute_red_threshold(var_i, null_vector, str(i), 'Null')
            is_constant, out_i = is_vari_constant
            
            if is_constant == True:
                verbose_history.append(out_i)
                continue
            
            const_filter.append([out_i, i])
        
        #check pairwise
        print('Checking Pairwise')
        for item_j in const_filter:
            var_j, j = item_j
            if not fo_filter:
                fo_filter.append(item_j)
                continue
            
            for item_k in fo_filter:
                var_k, k = item_k
                is_varj_redundant = self.compute_red_threshold(var_j, var_k, str(j), str(k))
                is_redundant, out_j = is_varj_redundant
                
                if is_redundant:
                    verbose_history.append(out_j)
                    break
            
            if not is_redundant:    
                fo_filter.append(item_j)
                
        return fo_filter, verbose_history
    
    def postcalc_red(self, data_obj, subset_inds):
        
        data_inds, output_ind = subset_inds
        total_samp = data_obj.total_samp
        
        samp = None
        if data_obj.net.layer_flag == 'INTER':
            assert data_obj.activations is not None, 'Pre compute activation data'
            samp = data_obj.activations
        else:
            samp = data_obj.samples
        
        bin_res_data = acsu.binarize(samp[:, list(data_inds)])
        bin_output = acsu.binarize(samp[:, output_ind])
        
        indicator = np.array([1 if i < 1 else 0 for i in range(len(data_inds))])
        ind_conv = 1.0 - indicator
        
        indicator = torch.from_numpy(indicator).float()
        ind_conv = torch.from_numpy(ind_conv).float()
        estimate1 = 0.0
        estimate2 = 0.0
        
        for i in range(self.num_red_samples):
            s1 = bin_res_data[random.randint(0,total_samp-1), :]
            s1 = Variable(torch.FloatTensor(s1[None,:]))

            s2 = bin_res_data[random.randint(0,total_samp-1), :]
            s2 = Variable(torch.FloatTensor(s2[None,:]))

            s3 = bin_res_data[random.randint(0,total_samp-1), :]
            s3 = Variable(torch.FloatTensor(s3[None,:]))

            z = torch.multiply(ind_conv, s3)
            x1 = torch.multiply(indicator, s1)
            x1 = torch.add(x1, z)
            inds_y1 = torch.where(torch.all((bin_res_data == x1), dim=1))
            num_inds1 = list(inds_y1[0].size())
            sample_ind1 = inds_y1[0][random.randint(0, num_inds1[0]-1)]
            out_y1 = bin_output[sample_ind1, :]
            
            x2 = torch.multiply(indicator, s2)
            x2 = torch.add(x2, z)
            inds_y2 = torch.where(torch.all((bin_res_data == x2),dim=1))
            num_inds2 = list(inds_y2[0].size())
            sample_ind2 = inds_y2[0][random.randint(0, num_inds2[0]-1)]
            out_y2 = bin_output[sample_ind2, :]
            
            estimate1 += out_y1 * out_y2
            estimate2 += out_y1 * x1[0][0] * out_y2 * x2[0][0]
            
        estimate1 /= self.num_red_samples
        estimate2 /= self.num_red_samples
        
        if estimate1 > self.delta or estimate2 > self.delta:
            print('Estimate 1:')
            print(estimate1)
            
            print('Estimate 2:')
            print(estimate2)
            return False

        return True

    def gen_summary(self, x, y):
        #do something else here later
        return (str(x), str(y))