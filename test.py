#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function


# In[ ]:


import os
import numpy as np
from absl import app
from absl import flags


# In[ ]:


import tensorflow as tf
from env import Environment
from game import DRL_Game
from model import Network
from config import get_config
import random


# In[ ]:


FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')


# In[ ]:


def sim(config, network, game,each_scale_scenario_MLU_results):
    
    scenarios = game.lp_links
    print("we have %s scenarios",len(scenarios),scenarios)
    purified_scenarios = []
    for scenario in scenarios:
        if scenario not in purified_scenarios and (scenario[1],scenario[0]) not in purified_scenarios and (scenario[0],scenario[1]) not in purified_scenarios:
            purified_scenarios.append(scenario)
    scales = [3.8,3.4,3.6,3.9,4,4.1]
    print("we have %s purified_scenarios",len(purified_scenarios),purified_scenarios)
    env = Environment(config, is_training=True)
    print("******************** game.tm_indexes ",game.tm_indexes)
    DM_counter = 0
    #purified_scenarios = [(4,5),(0,2)]
    for tm_idx in game.tm_indexes:
        if DM_counter <=30:
            DM_counter +=1
            for scale in scales:
                for failed_link in purified_scenarios:
                    #for scenario (5, 3) scale 3.4 and  DM index 0  
#                     tm_idx = 0
#                     scale  = 3.4
#                     failed_link = (5, 3)
                    print('for scenario %s scale %s and  DM index %s '%(failed_link,scale,tm_idx))
                    state = game.get_state2(env,tm_idx,scale,failed_link)
                    if config.method == 'actor_critic':
                        policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
                    elif config.method == 'pure_policy':
                        policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
                    actions = policy.argsort()[-game.max_moves:]
                    #print('action is ',actions)
                    randomly_selected_actions = []
                    while(len(randomly_selected_actions)<len(actions)):

                        random_flow = random.randint(0,game.action_dim-1)
                        if random_flow not in randomly_selected_actions:
                            randomly_selected_actions.append(random_flow)
                    game.evaluate2(tm_idx, failed_link,scale,each_scale_scenario_MLU_results,randomly_selected_actions,actions, eval_delay=FLAGS.eval_delay) 
#                 game.evaluate(tm_idx,scale, each_scale_scenario_MLU_results,actions,eval_delay=FLAGS.eval_delay)
#     for tm_idx in game.tm_indexes:
#         state = game.get_state(tm_idx)
#         if config.method == 'actor_critic':
#             policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
#         elif config.method == 'pure_policy':
#             policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
#         actions = policy.argsort()[-game.max_moves:]
#         game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay) 


# In[ ]:


def main(_):
    #Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')
    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = DRL_Game(config, env)
    network = Network(config, game.state_dims, game.action_dim,game.max_moves)
    step = network.restore_ckpt(FLAGS.ckpt)
    if config.method == 'actor_critic':
        learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
    elif config.method == 'pure_policy':
        learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
    print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
    each_scale_scenario_MLU_results = 'each_scale_scenario_MLU_results_ATT_third_run.csv' 
    sim(config, network, game,each_scale_scenario_MLU_results)


# In[ ]:


if __name__ == '__main__':
    app.run(main)

