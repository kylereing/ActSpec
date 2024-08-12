import torch

class BranchAndBound:

    def __init__(self, data_obj, gl_obj, filter_obj, search_flag='BFS', max_branch=128):
        self.search_flag = search_flag
        self.max_branch = max_branch
        self.data_obj = data_obj
        self.gl_obj = gl_obj
        self.filter_obj = filter_obj    
            
    def begin_search(self, params=None):
        starting_vector = torch.zeros((self.data_obj.n))
        if params:
          starting_vector[params] = 1.0
        
        #filter out initial redundant variables
        filter, red_history = self.filter_obj.checkRedundancy(self.data_obj)
                
        #do something with report
        print(red_history)
        print('###########')
        
        if self.search_flag == 'BFS':
          return self.bfs_tree_search([starting_vector.float()], filter=filter)
          
        elif self.search_flag == 'RECURSIVE':
          return self.recursive_tree_search([starting_vector.float()], filter=filter)
          
    def bfs_tree_search(self, initial_state, filter=None):
      
      updated_subsets = initial_state
      if filter:
        inds_list = [item[1] for item in filter]
         
      for i in range(self.data_obj.n):
        if filter and i not in inds_list:
          continue
        
        if self.filter_obj.verbose:
          print(i)
          
        return_set = self.bfs_search_step(i, updated_subsets)
        updated_subsets = return_set

      return updated_subsets
    
    def bfs_search_step(self, depth, subsets):
      
      candidates = []
      if not subsets:
        return candidates
      
      weight_record = torch.zeros(len(subsets))
      for i, s in enumerate(subsets):
        weight = self.gl_obj.goldreich_levin_weight(self.data_obj, (depth,s))
        if weight > (self.gl_obj.tau **2 / 2):
          orth_check = True
          post_filter_inds = torch.nonzero(s)
          
          if len(post_filter_inds) > 1:
            
            fixed_ind = post_filter_inds[-1]
            variable_inds = post_filter_inds[:-1]
            orth_check = self.filter_obj.checkRedundancy(self.data_obj, prescreen=False, params=[variable_inds, fixed_ind])
            
          if orth_check == True:
            weight_record[i] = weight
            
      if self.filter_obj.verbose:
        print('Weight record:')
        print(weight_record)
        print('################################')
        
      if len(torch.nonzero(weight_record)) > self.max_branch:
        _, top_m = weight_record.topk(self.max_branch)
        pruned_subsets = [subsets[i] for i in top_m]
      else:
        pruned_subsets = [subsets[i] for i in torch.nonzero(weight_record)]
        
      if depth == self.data_obj.n-1:
        return pruned_subsets
      
      for set in pruned_subsets:
        candidates.append(set)
        branch_set = set.detach().clone()
        branch_set[depth] = 1.0
        candidates.append(branch_set)
        
      return candidates
      
      
    def recursive_tree_search(self, input, verbose=False):
      
      assert input == None, 'Too tricky to implement right now'
      return None
    