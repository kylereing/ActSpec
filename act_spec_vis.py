import torch
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def gen_sets(sets):
    size_list = []
    sets_list = []
    for i in range(len(sets)):
        size_list.append(torch.nonzero(sets[i]).size()[0])
        sets_list.append(torch.nonzero(sets[i],as_tuple=True))
    return sets_list, size_list

def gen_heatmap(sets, num_vars, filename=None):
    m1_heatmap = torch.zeros((num_vars))
    for img in sets:
        for inds in img[0]:
            m1_heatmap[inds] += 1.0

    m1_heatmap /= len(sets)
    m1_heatmap = torch.reshape(m1_heatmap, (int(np.sqrt(num_vars)), int(np.sqrt(num_vars))))
    plot = sns.heatmap(torch.abs(m1_heatmap), xticklabels = False, yticklabels = False)
    figure = plot.get_figure()
    if filename:
        figure.savefig(filename)
        
    return plot