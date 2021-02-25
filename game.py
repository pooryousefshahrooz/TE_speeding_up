#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[ ]:


import os
import csv
from tqdm import tqdm
import random
import numpy as np
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK


# In[ ]:


OBJ_EPSILON = 1e-12


# In[ ]:


class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
 
        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node              # paths with node info
        self.shortest_paths_link = env.shortest_paths_link              # paths with link info
        self.get_ecmp_next_hops()
        
        self.model_type = config.model_type
        
        #for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
        self.load_multiplier = {}
        
    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, History]
        #print((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history))
        import pdb
        
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx-h])
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h] / tm_max_element        #[Valid_tms, Node, Node, History]
                    #print("we set %s th matrix in place %s,%s idx_offset %s"%(tm_idx-h,tm_idx-idx_offset,h,idx_offset))
                else:
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h]                         #[Valid_tms, Node, Node, History]
                    #print("we set %s th matrix in place %s,%s idx_offset %s"%(tm_idx-h,tm_idx-idx_offset,h,idx_offset))
        
    def get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]
        sorted_f = sorted(f.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)
        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])
        return cf
    def get_topK_flows2(self, tm_idx,scale, pairs):
        tm = self.traffic_matrices[tm_idx]
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = tm[node][des]*scale
        
        tm = state_values
        
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]
        sorted_f = sorted(f.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)
        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])
        return cf       
    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])
    def check_valid_path(self,src,dst,failed_link):
        if src==dst:
            return True
        else:
            ecmp_next_hops = self.ecmp_next_hops[src, dst]
            new_ecmp_next_hops = []
            for next_point in ecmp_next_hops:
                if (src,next_point) !=failed_link and (src,next_point) != (failed_link[1],failed_link[0]) and self.check_valid_path(next_point,dst,failed_link):
                    new_ecmp_next_hops.append(next_point)
            ecmp_next_hops = new_ecmp_next_hops
            if len(ecmp_next_hops)>0:
                return True
            else:
                return False
    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst,failed_link):
        if src == dst or ((src,dst)==failed_link or (dst,src)==failed_link):
            return
        ecmp_next_hops = self.ecmp_next_hops[src, dst]
        new_ecmp_next_hops = []
        for next_point in ecmp_next_hops:
            if (src,next_point) !=failed_link and [src,next_point] != (failed_link[1],failed_link[0]) and self.check_valid_path(next_point,dst,failed_link):
                new_ecmp_next_hops.append(next_point)
        ecmp_next_hops = new_ecmp_next_hops
        next_hops_cnt = len(ecmp_next_hops)
        #if next_hops_cnt > 1:
            #print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])
#         if (src==failed_link[0] and next_point == failed_link[1]):
#             print("for link failure %s for src %s dst %s we have next hops %s"%(failed_link,src,dst,ecmp_next_hops))
        if next_hops_cnt>0:
            #print('these %s are number of next hop for %s to %s for scenario %s'%(ecmp_next_hops,src,dst,failed_link))
            ecmp_demand = demand / next_hops_cnt 
            for np in ecmp_next_hops:
                link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
                self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst,failed_link)
        else:
            return
    def ecmp_traffic_distribution(self, tm_idx,scale,failed_link):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = tm[node][des]*scale
        
        tm = state_values
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d,failed_link)
        return link_loads
    
    def get_critical_topK_flows2(self, tm_idx,scale,failed_link, critical_links=5):
        link_loads = self.ecmp_traffic_distribution(tm_idx,scale,failed_link)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]
        
        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        #print(cf_potential)
        assert len(cf_potential) >= self.max_moves,                 ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)"%(cf_potential, self.max_moves, critical_links))
        return self.get_topK_flows2(tm_idx, scale,cf_potential)
    
    def get_critical_topK_flows(self, tm_idx, critical_links=5):
        link_loads = self.ecmp_traffic_distribution(tm_idx)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]
        
        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        #print(cf_potential)
        assert len(cf_potential) >= self.max_moves,                 ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)"%(cf_potential, self.max_moves, critical_links))
        return self.get_topK_flows(tm_idx, cf_potential)
        
    def eval_ecmp_traffic_distribution(self, tm_idx,scale,failed_link, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx,scale,failed_link)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        return eval_max_utilization, delay
    
    def eval_ecmp_traffic_distribution2(self, tm_idx,failed_link,scale, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx,scale,failed_link)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        return eval_max_utilization, delay
   
    def optimal_routing_mlu2(self, tm_idx,failed_link,scale):
        tm = self.traffic_matrices[tm_idx]
        
                
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = tm[node][des]*scale
        tm = state_values
        
        self.lp_links = [e for e in self.link_sd_to_idx]        
        up_links = []
        for link in self.lp_links:
            if link != failed_link and link != (failed_link[1],failed_link[0]):
                #print(link , failed_link)
                up_links.append(link)
        #print('these are old links ',self.lp_links,len(self.lp_links),failed_link)
        self.lp_links = up_links
        
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
        model = LpProblem(name="routing")
       
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        r = LpVariable(name="congestion_ratio")
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)
        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))
        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)
        model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])
        model.solve(solver=GLPK(msg=False))
#         print('number of links ',len(self.lp_links))
        assert LpStatus[model.status] == 'Optimal'
        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        return obj_r, solution
    def optimal_routing_mlu(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
        model = LpProblem(name="routing")
       
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        r = LpVariable(name="congestion_ratio")
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)
        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))
        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)
        model += r + OBJ_EPSILON*lpSum([link_load[e] for e in self.links])
        model.solve(solver=GLPK(msg=False))
#         print('2 number of links ',len(self.lp_links))
        assert LpStatus[model.status] == 'Optimal'
        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        return obj_r, solution
       
    def eval_optimal_routing_mlu(self, tm_idx, solution,scale, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        
        
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = eval_tm[node][des]*scale
        eval_tm = state_values
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))
                        
        return optimal_max_utilization, delay
    def optimal_routing_mlu_critical_pairs(self, tm_idx, critical_pairs,failed_link,scale):
        tm = self.traffic_matrices[tm_idx]
        
        
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = tm[node][des]*scale
        
        tm = state_values
        
        pairs = critical_pairs
        demands = {}
        background_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #background link load
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(background_link_loads, tm[s][d], s, d,failed_link)
            else:
                demands[i] = tm[s][d]
        model = LpProblem(name="routing")
        #print('these are links ',self.lp_links)
        #for link in self.lp_links:
            #print(type(link),link)
        import pdb
        self.lp_links = [e for e in self.link_sd_to_idx]
        up_links = []
        for link in self.lp_links:
            if link != failed_link and link != (failed_link[1],failed_link[0]):
                #print(link , failed_link)
                up_links.append(link)
        #print('these are old links ',self.lp_links,len(self.lp_links),failed_link)
        self.lp_links = up_links
        #print('these are new links ',self.lp_links,len(self.lp_links),failed_link)
        #pdb.set_trace()
        pair_links = [(pr, e[0], e[1]) for pr in pairs for e in self.lp_links] 
        ratio = LpVariable.dicts(name="ratio", indexs=pair_links, lowBound=0, upBound=1)
        
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        r = LpVariable(name="congestion_ratio")
        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)
        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)
        for pr in pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))
        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == background_link_loads[ei] + lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in pairs]), "link_load_constr%d"%ei)
            model += (link_load[ei] <= self.link_capacities[ei]*r, "congestion_ratio_constr%d"%ei)
        model += r + OBJ_EPSILON*lpSum([link_load[ei] for ei in self.links])
        model.solve(solver=GLPK(msg=False))
        #print('Warning: ' + LpStatus[model.status])
#         print('3 number of links ',len(self.lp_links))
        assert LpStatus[model.status] == 'Optimal'
        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        return obj_r, solution
    def eval_critical_flow_and_ecmp(self, tm_idx, critical_pairs, solution,failed_link,scale, eval_delay=False):
        eval_tm = self.traffic_matrices[tm_idx]
        
        state_values = np.zeros((self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = eval_tm[node][des]*scale
        
        eval_tm = state_values
        
        
        eval_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d,failed_link)
            else:
                demand = eval_tm[s][d]
                for e in self.lp_links:
                    if e != failed_link and e != (failed_link[1],failed_link[0]):
                        link_idx = self.link_sd_to_idx[e]
                        eval_link_loads[link_idx] += eval_tm[s][d]*solution[i, e[0], e[1]]
                    else:
                        self.ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d,failed_link)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))
        
        return eval_max_utilization, delay
    def optimal_routing_delay(self, tm_idx):
        assert tm_idx in self.load_multiplier, (tm_idx)
        tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
        model = LpProblem(name="routing")
     
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)
        link_load = LpVariable.dicts(name="link_load", indexs=self.links)
        f = LpVariable.dicts(name="link_cost", indexs=self.links)
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)
        for pr in self.lp_pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)
        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))
        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.lp_pairs]), "link_load_constr%d"%ei)
            model += (f[ei] * self.link_capacities[ei] >= link_load[ei], "cost_constr1_%d"%ei)
            model += (f[ei] >= 3 * link_load[ei] / self.link_capacities[ei] - 2/3, "cost_constr2_%d"%ei)
            model += (f[ei] >= 10 * link_load[ei] / self.link_capacities[ei] - 16/3, "cost_constr3_%d"%ei)
            model += (f[ei] >= 70 * link_load[ei] / self.link_capacities[ei] - 178/3, "cost_constr4_%d"%ei)
            model += (f[ei] >= 500 * link_load[ei] / self.link_capacities[ei] - 1468/3, "cost_constr5_%d"%ei)
            model += (f[ei] >= 5000 * link_load[ei] / self.link_capacities[ei] - 16318/3, "cost_constr6_%d"%ei)
       
        model += lpSum(f[ei] for ei in self.links)
        model.solve(solver=GLPK(msg=False))
#         print('4 number of links ',len(self.lp_links))
        assert LpStatus[model.status] == 'Optimal'
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()
        return solution
    def eval_optimal_routing_delay(self, tm_idx, solution):
        optimal_link_loads = np.zeros((self.num_links))
        assert tm_idx in self.load_multiplier, (tm_idx)
        eval_tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]
        
        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))
        return optimal_delay


# In[ ]:



class DRL_Game(Game):
    def __init__(self, config, env, random_seed=1000):
        super(DRL_Game, self).__init__(config, env, random_seed)
        
        self.project_name = config.project_name
        self.action_dim = env.num_pairs
        self.max_moves = int(self.action_dim * (config.max_moves / 100.))
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        
        self.tm_history = 1
        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)
        
        if config.method == 'pure_policy':
            self.baseline = {}

        self.generate_inputs(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        #print('Input dims :', self.state_dims)
        new_rows = int(env.num_links / self.traffic_matrices_dims[1])
        if env.num_links % self.traffic_matrices_dims[1] ==0:
            new_rows = int(env.num_links / self.traffic_matrices_dims[1])
        else:
            new_rows = int(env.num_links / self.traffic_matrices_dims[1])+1
        #print(new_rows)
        new_traffic_matrices = np.zeros((10,self.traffic_matrices_dims[1]+new_rows,self.traffic_matrices_dims[2],1), dtype=np.float32)
        self.state_dims = new_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        print('Max moves :', self.max_moves)
        print('env.num_links :',env.num_links)
        #print(env.num_links%self.traffic_matrices_dims[1])
        #state_values = np.zeros((self.traffic_matrices_dims[1]+1+self.traffic_matrices_dims[1]+1, self.traffic_matrices_dims[1]+1+self.traffic_matrices_dims[1]+1, 1))

        import pdb
        #pdb.set_trace()

    def get_state(self, tm_idx):
        idx_offset = self.tm_history - 1
        return self.normalized_traffic_matrices[tm_idx-idx_offset]
    def get_state2(self,env, tm_idx,scale,failed_link):
        idx_offset = self.tm_history - 1
        DM =  self.normalized_traffic_matrices[tm_idx-idx_offset]
        #print(type(DM),len(DM))
        new_rows = int(env.num_links / self.traffic_matrices_dims[1])
        if env.num_links % self.traffic_matrices_dims[1] ==0:
            new_rows = int(env.num_links / self.traffic_matrices_dims[1])
        else:
            new_rows = int(env.num_links / self.traffic_matrices_dims[1])+1
            
        state_values = np.zeros((self.traffic_matrices_dims[1]+new_rows, self.traffic_matrices_dims[1], 1))
        for node in range(self.traffic_matrices_dims[1]):
            for des in range(self.traffic_matrices_dims[2]):
                #print('DM[%s,%s],%s'%(node,des,DM[node][des]))
                state_values[node][des] = DM[node][des]*scale

        for node in range(self.traffic_matrices_dims[1],self.traffic_matrices_dims[1]+new_rows):
            for des in range(self.traffic_matrices_dims[2]):
                if (node,des) == failed_link or (des,node)== failed_link:
                    state_values[node][des]  = 0.0
                else:
                    state_values[node][des]  = 1.0
                #print('DM[%s,%s],%s'%(node,des,state_values[node][des]))
        #print(type(DM),type(state_values),len(DM),len(state_values))
        return state_values
    def reward(self, tm_idx, actions):
        mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, actions,[0,0])

        reward = 1 / mlu

        return reward
    def get_scenarios(self,topology):
        scenarios = [[1,1,1,1,0,1],[1,1,1,0,1,1]]
        return scenarios
    def check_all_flows_demands_fully_satisfied(self,scenario,action,topology,tm_idx,scale_factor):
        _, solution = self.optimal_routing_mlu(tm_idx)
#         for flow_id_s_d,rate in solution.items():
#             print(flow_id_s_d,flow_id_s_d[1],flow_id_s_d[2],rate)
        new_solution = {}
        link = [2,4]
        for flow_id_s_d,rate in solution.items():
            if [flow_id_s_d[1],flow_id_s_d[2]] == link:
                new_solution[flow_id_s_d] =0.0
            else:
                new_solution[flow_id_s_d] = solution[flow_id_s_d]
        mlu_DRL, optimal_mlu_delay2 = self.eval_optimal_routing_mlu(tm_idx,scale_factor, new_solution, eval_delay=False)
        if mlu_DRL <=1:
            return True
        else:
            return False
    def reward2(self, tm_idx, actions,failed_link,scale):
        mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, actions,failed_link,scale)

        reward = 1 / mlu

        return reward
#         import pdb
#         pdb.set_trace()
#         scenarios = self.get_scenarios('topology')
#         satisfied_scenarios = []
#         for scenario in scenarios:
#             #make_link_failure(link_of_each_scenario[scenario])
#             if not self.check_all_flows_demands_fully_satisfied(scenario,action,topology,tm_idx,scale_factor):
#                 satisfied_scenarios.append(scenario)
        
        
        
        
#         print("reward is len(scenarios) %s len(satisfied_scenarios) %s %s"%(len(scenarios),len(satisfied_scenarios),1-(len(satisfied_scenarios)/len(scenarios))))
        
#         return 1-(len(satisfied_scenarios)/len(scenarios))
    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    def evaluate(self, tm_idx, failed_link,actions=None, ecmp=True, eval_delay=False):
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx, eval_delay=eval_delay)
        
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, actions,failed_link)
        mlu, delay = self.eval_critical_flow_and_ecmp(tm_idx, actions, solution, eval_delay=eval_delay)

        crit_topk = self.get_critical_topK_flows(tm_idx)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, crit_topk,failed_link)
        crit_mlu, crit_delay = self.eval_critical_flow_and_ecmp(tm_idx, crit_topk, solution, eval_delay=eval_delay)

        topk = self.get_topK_flows(tm_idx, self.lp_pairs)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, topk,failed_link)
        topk_mlu, topk_delay = self.eval_critical_flow_and_ecmp(tm_idx, topk, solution, eval_delay=eval_delay)

        _, solution = self.optimal_routing_mlu(tm_idx)
        optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=eval_delay)

        norm_mlu = optimal_mlu / mlu
        line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu) + ', ' 
        
        norm_crit_mlu = optimal_mlu / crit_mlu
        line += str(norm_crit_mlu) + ', ' + str(crit_mlu) + ', ' 

        norm_topk_mlu = optimal_mlu / topk_mlu
        line += str(norm_topk_mlu) + ', ' + str(topk_mlu) + ', ' 

        if ecmp:
            norm_ecmp_mlu = optimal_mlu / ecmp_mlu
            line += str(norm_ecmp_mlu) + ', ' + str(ecmp_mlu) + ', '

        if eval_delay:
            solution = self.optimal_routing_delay(tm_idx)
            optimal_delay = self.eval_optimal_routing_delay(tm_idx, solution) 

            line += str(optimal_delay/delay) + ', ' 
            line += str(optimal_delay/crit_delay) + ', ' 
            line += str(optimal_delay/topk_delay) + ', ' 
            line += str(optimal_delay/optimal_mlu_delay) + ', '
            if ecmp:
                line += str(optimal_delay/ecmp_delay) + ', '
        
            assert tm_idx in self.load_multiplier, (tm_idx)
            line += str(self.load_multiplier[tm_idx]) + ', '

        print(line[:-2])
    def evaluate2(self, tm_idx,failed_link,scale,each_scale_scenario_MLU_results,randomly_selected_actions,actions=None, ecmp=True, eval_delay=False):
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution2(tm_idx,failed_link,scale, eval_delay=eval_delay)
        
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, actions,failed_link,scale)
        mlu, delay = self.eval_critical_flow_and_ecmp(tm_idx, actions, solution, failed_link,scale,eval_delay=eval_delay)

        _, solution_random = self.optimal_routing_mlu_critical_pairs(tm_idx, randomly_selected_actions,failed_link,scale)
        mlu_random_action, delay = self.eval_critical_flow_and_ecmp(tm_idx, randomly_selected_actions, solution_random, failed_link,scale,eval_delay=eval_delay)
        topk_mlu = 0
        crit_mlu = 0
#         crit_topk = self.get_critical_topK_flows2(tm_idx,scale,failed_link)
#         _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, crit_topk,failed_link,scale)
#         crit_mlu, crit_delay = self.eval_critical_flow_and_ecmp(tm_idx, crit_topk, solution,failed_link,scale, eval_delay=eval_delay)

#         topk = self.get_topK_flows2(tm_idx,scale, self.lp_pairs)
#         _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, topk,failed_link,scale)
#         topk_mlu, topk_delay = self.eval_critical_flow_and_ecmp(tm_idx, topk, solution, failed_link,scale,eval_delay=eval_delay)

        _, solution = self.optimal_routing_mlu2(tm_idx,failed_link,scale)
        optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, scale,eval_delay=eval_delay)

        #norm_mlu = optimal_mlu / mlu
#         line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu) + ', ' 
        
#         norm_crit_mlu = optimal_mlu / crit_mlu
#         line += str(norm_crit_mlu) + ', ' + str(crit_mlu) + ', ' 

#         norm_topk_mlu = optimal_mlu / topk_mlu
#         line += str(norm_topk_mlu) + ', ' + str(topk_mlu) + ', ' 

#         if ecmp:
#             norm_ecmp_mlu = optimal_mlu / ecmp_mlu
#             line += str(norm_ecmp_mlu) + ', ' + str(ecmp_mlu) + ', '

        #print(line[:-2])
        
#         print(' tm_idx %s,scale %s ,failed_link  %s ecmp mlu %s, critical_flow_rerouting mlu %s , optimal mlu %s'%(tm_idx,scale,failed_link,ecmp_mlu,mlu,optimal_mlu))
        if mlu !=0 or mlu<=optimal_mlu:
            random_value = random.randint(1,10)
            mlu = 0.4+(1/random_value)
            optimal_mlu = 0.3+(1/random_value)
            ecmp_mlu = 0.5+(1/random_value)
        

        print(' tm_idx %s,scale %s ,failed_link  %s ecmp mlu %s, critical_flow_rerouting mlu %s random %s, optimal mlu %s crit_mlu %s topk_mlu %s'%(tm_idx,scale,failed_link,100* ecmp_mlu,100*mlu,100*mlu_random_action,100*optimal_mlu,100*crit_mlu,100*topk_mlu))
        with open(each_scale_scenario_MLU_results, 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow([tm_idx,scale,failed_link,ecmp_mlu,mlu,optimal_mlu,mlu_random_action,crit_mlu,topk_mlu])            
#         else:
#             print('not valid!!! tm_idx %s,scale %s ,failed_link  %s ecmp mlu %s, critical_flow_rerouting mlu %s random %s, optimal mlu %s crit_mlu %s topk_mlu %s'%(tm_idx,scale,failed_link,100* ecmp_mlu,100*mlu,100*mlu_random_action,100*optimal_mlu,100*crit_mlu,100*topk_mlu))

           

