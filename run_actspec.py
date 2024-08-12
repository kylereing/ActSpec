# from google.colab import drive
# content_dir = 'C:/Users/kyler/ActSpec/content'
# drive.mount(content_dir+'/drive', force_remount=True)

import act_spec_data as acsd
import act_spec_models as acsm
import goldreich_levin as gl
import act_spec as acs
import branch_and_bound as bnb
import act_spec_preprocessing as acsp
import act_spec_vis as acsv
import torch

data_flag = 'LR_MNIST' #LR_MNIST, MNIST, NL
layer_flag = 'INPUT' #INPUT, INTERMED
preproc_flag = 'TWO_DIGIT' #TWO_DIGIT
model_flag = 'LOW_RES' #MLP, LOW_RES, OPT125

#loading the data
ref_class = torch.tensor([7]).float()
ref_class2 = torch.tensor([1]).float()
load_data_object = acsd.retrieve_data(data_flag, params=[ref_class])
print('Data Loaded')

#loading the model
load_model_object = acsm.retrieve_model(model_flag, params=[load_data_object])
print('Model Loaded')

#preprocessing the data
# proc_data_obj = acsp.process_data(preproc_flag, load_data_object, params=[ref_class, ref_class2, load_model_object])
print('Data Preprocessed')

#initialize Goldreich-Levin parameters
gl_object = gl.GoldreichLevin(load_model_object, tau=0.5)
print('GL Object Initialized')

#initialize ActSpec filter parameters
act_spec_obj = acs.ActSpec()
print('ActSpec Object Initialized')

#initialize BnB search parameters
act_spec_bnb = bnb.BranchAndBound(load_data_object, gl_object, act_spec_obj)
# act_spec_bnb = bnb.BranchAndBound(proc_data_obj, gl_object, act_spec_obj)
print('BnB Object Initialized')

#running experiment
results = act_spec_bnb.begin_search()

#gathering, reporting, and visualizing results
sets, sizes = acsv.gen_sets(results)
print(sets)

heatmap = acsv.gen_heatmap(sets, load_data_object.n, filename='test_heatmap8.png')