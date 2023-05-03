import numpy as np
import gym
import os
import json
import gym_graph
import random
import argparse
import time as tt
import torch
import pickle
import sys
from SACD import SACD
sys.setrecursionlimit(2000)

# This script is used to evaluate a DRL agent on a single instance of a topology and a TM 
# from the repetita dataset. The eval_on_single_topology.py script calls this script for each TM

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_MIDDROUT_AGENT_SP = 'GraphEnv-v16'
ENV_SIMM_ANEAL_AGENT = 'GraphEnv-v15'
ENV_SAP_AGENT = 'GraphEnv-v20'
SEED = 9

percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
torch.manual_seed(1)

# Indicates how many time-steps has an episode
EPISODE_LENGTH_MIDDROUT = 100
NUM_ACTIONS = 15 # Put a very large number if we want to take all actions possible for each topology
KP = True

MAX_NUM_EDGES = 100

def play_middRout_games_sp(tm_id, env_middRout_sp, agent, timesteps):
    demand, source, destination = env_middRout_sp.reset(tm_id)
    rewardAddTest = 0

    initMaxUti = env_middRout_sp.edgeMaxUti[2]
    OSPF_init = initMaxUti
    best_routing = env_middRout_sp.sp_middlepoints_step.copy()

    list_of_demands_to_change = env_middRout_sp.list_eligible_demands
    timesteps.append((0, initMaxUti))

    start = tt.time()
    time_start_DRL = start
    while 1:
        action_dist, tensor = agent.predict(env_middRout_sp, source, destination, demand)
        action = torch.argmax(action_dist)
        
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout_sp.step(action, demand, source, destination)
        rewardAddTest += reward
        if maxLinkUti[2]<initMaxUti:
            initMaxUti = maxLinkUti[2]
            best_routing = env_middRout_sp.sp_middlepoints_step.copy()
            timesteps.append((tt.time()-time_start_DRL, initMaxUti))
        if done:
            break
    end = tt.time()
    return initMaxUti, end-start, OSPF_init, best_routing, list_of_demands_to_change, time_start_DRL

class SIMULATED_ANNEALING_SP:
    def __init__(self, env):
        self.num_actions = env.K
    
    def next_state(self, env):
        source, destination = -1, -1
        while source==destination:
            source = np.random.randint(low=0, high=env.numNodes-1)
            destination = np.random.randint(low=0, high=env.numNodes-1)
        # We explore all the possible actions with all the possible src,dst pairs 
        action = np.random.randint(low=0, high=len(env.src_dst_k_middlepoints[str(source)+':'+str(destination)]))

        # We des-allocate the chosen path to try to allocate it in another place
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        originalMiddlepoint = -1
        if str(source)+':'+str(destination) in env.sp_middlepoints:
            originalMiddlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp(source, originalMiddlepoint, source, destination)
            env.decrease_links_utilization_sp(originalMiddlepoint, destination, source, destination)
            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            env.decrease_links_utilization_sp(source, destination, source, destination)

        # We get the K-middlepoints between source-destination
        middlePointList = list(env.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        env.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            env.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            env.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        # Compute new energy for the corresponding action
        energy = -1000000
        position = 0
        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0]/link_capacity>energy:
                    energy = env.edge_state[position][0]/link_capacity
                position = position + 1
        
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(destination) in env.sp_middlepoints:
            middlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp(source, middlepoint, source, destination)
            env.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            env.decrease_links_utilization_sp(source, destination, source, destination)
        
        # Allocate back the demand whose actions we explored
        # If the current demand had a middlepoint, we allocate src-middlepoint-dst
        if originalMiddlepoint>=0:
            # First we allocate until the middlepoint
            env.allocate_to_destination_sp(source, originalMiddlepoint, source, destination)
            # Then we allocate from the middlepoint to the destination
            env.allocate_to_destination_sp(originalMiddlepoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            env.sp_middlepoints[str(source)+':'+str(destination)] = originalMiddlepoint
        else:
            # Then we allocate from the middlepoint to the destination
            env.allocate_to_destination_sp(source, destination, source, destination)

        return energy, action, source, destination
        

def play_sp_simulated_annealing_games(tm_id):
    env_sim_anneal = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_sim_anneal.seed(SEED)
    env_sim_anneal.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands)

    init_energy = env_sim_anneal.reset_sp(tm_id)
    sim_agent = SIMULATED_ANNEALING_SP(env_sim_anneal)

    Tmax = 1
    Tmin = 0.000001
    cooling_ratio = 0.000001 # best value is 0.0001 but very slow
    T = Tmax
    L = 4 # Number of trials per temperature value. With L=3 I get even better results
    energy = init_energy
    itera = 0

    start = tt.time()
    while T>Tmin:
        for _ in range(L):
            next_energy, action, source, destination = sim_agent.next_state(env_sim_anneal)
            delta_energy = (energy-next_energy)
            itera += 1
            # If we decreased the maximum link utilization we take the action
            if delta_energy>0:
                # We des-allocate the chosen path to apply later the chosen action
                # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
                if str(source)+':'+str(destination) in env_sim_anneal.sp_middlepoints:
                    middlepoint = env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)]
                    originalMiddlepoint = env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)]
                    env_sim_anneal.decrease_links_utilization_sp(source, middlepoint, source, destination)
                    env_sim_anneal.decrease_links_utilization_sp(middlepoint, destination, source, destination)
                    del env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)] 
                else: # Remove the bandwidth allocated from the src to the destination
                    env_sim_anneal.decrease_links_utilization_sp(source, destination, source, destination)
                energy = env_sim_anneal.step_sp(action, source, destination)
            # If not, accept the action with some probability
            elif np.exp(delta_energy/T)>random.uniform(0, 1):
                # We des-allocate the chosen path to apply later the chosen action
                # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
                if str(source)+':'+str(destination) in env_sim_anneal.sp_middlepoints:
                    middlepoint = env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)]
                    originalMiddlepoint = env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)]
                    env_sim_anneal.decrease_links_utilization_sp(source, middlepoint, source, destination)
                    env_sim_anneal.decrease_links_utilization_sp(middlepoint, destination, source, destination)
                    del env_sim_anneal.sp_middlepoints[str(source)+':'+str(destination)] 
                else: # Remove the bandwidth allocated from the src to the destination
                    env_sim_anneal.decrease_links_utilization_sp(source, destination, source, destination)
                energy = env_sim_anneal.step_sp(action, source, destination)
        T -= cooling_ratio
    end = tt.time()
    return energy, end-start

class HILL_CLIMBING:
    def __init__(self, env):
        self.num_actions = env.K 

    def get_value_sp(self, env, source, destination, action):
        # We get the K-middlepoints between source-destination
        middlePointList = list(env.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        env.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            env.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            env.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000
        position = 0
        # Get the maximum loaded link and it's value after allocating to the corresponding middlepoint
        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = env.edge_state[position][0]/link_capacity
                position = position + 1
        
        # Dissolve allocation step so that later we can try another action
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(destination) in env.sp_middlepoints:
            middlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp(source, middlepoint, source, destination)
            env.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            env.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue

    def get_value_sp_kp(self, env, source, destination, action):
        # First we allocate until the middlepoint
        env.allocate_to_destination_sp(source, destination, source, destination, action)
        env.sp_pathk[str(source) + ':' + str(destination)] = action

        currentValue = -1000000
        position = 0
        # Get the maximum loaded link and it's value after allocating to the corresponding middlepoint
        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0] / link_capacity > currentValue:
                    currentValue = env.edge_state[position][0] / link_capacity
                position = position + 1

        # Dissolve allocation step so that later we can try another action
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source) + ':' + str(destination) in env.sp_pathk:
            env.decrease_links_utilization_sp(source, destination, source, destination)
            del env.sp_pathk[str(source) + ':' + str(destination)]
        else:  # Remove the bandwidth allocated from the src to the destination
            env.decrease_links_utilization_sp(source, destination, source, destination)

        return -currentValue
    
    def explore_neighbourhood_sp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        # Iterate for each demand possible
        for source in range(env.numNodes):
            for dest in range(env.numNodes):
                if source!=dest:
                    for action in range(len(env.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                        middlepoint = -1
                        # First we need to desallocate the current demand before we explore all it's possible actions
                        # Check if there is a middlepoint to desallocate from src-middlepoint-dst
                        if str(source)+':'+str(dest) in env.sp_middlepoints:
                            middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                            env.decrease_links_utilization_sp(source, middlepoint, source, dest)
                            env.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                            del env.sp_middlepoints[str(source)+':'+str(dest)] 
                        # Else, there is no middlepoint and we desallocate the entire src,dst
                        else: 
                            # Remove the bandwidth allocated from the src to the destination
                            env.decrease_links_utilization_sp(source, dest, source, dest)

                        evalState = self.get_value_sp(env, source, dest, action)
                        if evalState > nextVal:
                            nextVal = evalState
                            next_state = (action, source, dest)
                        
                        # Allocate back the demand whose actions we explored
                        # If the current demand had a middlepoint, we allocate src-middlepoint-dst
                        if middlepoint>=0:
                            # First we allocate until the middlepoint
                            env.allocate_to_destination_sp(source, middlepoint, source, dest)
                            # Then we allocate from the middlepoint to the destination
                            env.allocate_to_destination_sp(middlepoint, dest, source, dest)
                            # We store that the pair source,destination has a middlepoint
                            env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                        else:
                            # Then we allocate from the middlepoint to the destination
                            env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state

    def explore_neighbourhood_DRL_sp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        # We iterate over the top critical demands
        for elem in env.list_eligible_demands:
            source = elem[0]
            dest = elem[1]
            for action in range(len(env.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                middlepoint = -1
                # First we need to desallocate the current demand before we explore all it's possible actions
                # Check if there is a middlepoint to desallocate from src-middlepoint-dst
                if str(source)+':'+str(dest) in env.sp_middlepoints:
                    middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                    env.decrease_links_utilization_sp(source, middlepoint, source, dest)
                    env.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                    del env.sp_middlepoints[str(source)+':'+str(dest)] 
                # Else, there is no middlepoint and we desallocate the entire src,dst
                else: 
                    # Remove the bandwidth allocated from the src to the destination
                    env.decrease_links_utilization_sp(source, dest, source, dest)

                evalState = self.get_value_sp(env, source, dest, action)
                if evalState > nextVal:
                    nextVal = evalState
                    next_state = (action, source, dest)
                
                # Allocate back the demand whose actions we explored
                # If the current demand had a middlepoint, we allocate src-middlepoint-dst
                if middlepoint>=0:
                    # First we allocate until the middlepoint
                    env.allocate_to_destination_sp(source, middlepoint, source, dest)
                    # Then we allocate from the middlepoint to the destination
                    env.allocate_to_destination_sp(middlepoint, dest, source, dest)
                    # We store that the pair source,destination has a middlepoint
                    env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                else:
                    # Then we allocate from the middlepoint to the destination
                    env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state

    def explore_neighbourhood_DRL_sp_kp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        # We iterate over the top critical demands
        for elem in env.list_eligible_demands:
            source = elem[0]
            dest = elem[1]
            for a in range(NUM_ACTIONS):
                action = -1
                # First we need to desallocate the current demand before we explore all it's possible actions
                # Check if there is a middlepoint to desallocate from src-middlepoint-dst
                if str(source) + ':' + str(dest) in env.sp_pathk:
                    action = env.sp_pathk[str(source) + ':' + str(dest)]
                    env.decrease_links_utilization_sp(source, dest, source, dest)
                    del env.sp_pathk[str(source) + ':' + str(dest)]
                    # Else, there is no middlepoint and we desallocate the entire src,dst
                else:
                    # Remove the bandwidth allocated from the src to the destination
                    env.decrease_links_utilization_sp(source, dest, source, dest)

                evalState = self.get_value_sp_kp(env, source, dest, a)
                if evalState > nextVal:
                    nextVal = evalState
                    next_state = (a, source, dest)

                # Allocate back the demand whose actions we explored
                # If the current demand had a middlepoint, we allocate src-middlepoint-dst
                if action >= 0:
                    # First we allocate until the middlepoint
                    env.allocate_to_destination_sp(source, dest, source, dest, action)
                    # We store that the pair source,destination has a middlepoint
                    env.sp_pathk[str(source) + ':' + str(dest)] = action
                else:
                    # Then we allocate from the middlepoint to the destination
                    env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state

def play_sp_hill_climbing_games(tm_id):
    # Here we use sp in hill climbing to select the middlepoint and to evaluate
    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_hill_climb.seed(SEED)
    env_hill_climb.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands)

    currentVal = env_hill_climb.reset_hill_sp(tm_id)
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1:
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_sp(env_hill_climb)
        # If the difference between the two edges is super small but non-zero, we break (this is because of precision reasons)
        if nextVal<=currentVal or (abs((-1)*nextVal-(-1)*currentVal)<1e-4):
            break
        
        # Before we apply the new action, we need to remove the current allocation of the chosen demand
        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp(source, middlepoint, source, dest)
            env_hill_climb.decrease_links_utilization_sp(middlepoint, dest, source, dest)
            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 
        # If there is no middlepoint assigned to the src,dst pair
        else:
            # Remove the bandwidth allocated from the src to the destination using sp
            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        
        # We apply the new chosen action to the selected demand
        currentVal = env_hill_climb.step_hill_sp(action, source, dest)
    end = tt.time()
    return currentVal*(-1), end-start

def play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing, list_of_demands_to_change, timesteps, time_start_DRL):
    # Here we use sp in hill climbing to select the middlepoint and to evaluate
    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_hill_climb.seed(SEED)
    env_hill_climb.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands)

    currentVal = env_hill_climb.reset_DRL_hill_sp(tm_id, best_routing, list_of_demands_to_change)
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1:
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_DRL_sp(env_hill_climb)
        # If the difference between the two edges is super small but non-zero, we break (this is because of precision reasons)
        if nextVal<=currentVal or (abs((-1)*nextVal-(-1)*currentVal)<1e-4):
            break
        
        # Before we apply the new action, we need to remove the current allocation of the chosen demand
        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp(source, middlepoint, source, dest)
            env_hill_climb.decrease_links_utilization_sp(middlepoint, dest, source, dest)
            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 
        # If there is no middlepoint assigned to the src,dst pair
        else:
            # Remove the bandwidth allocated from the src to the destination using sp
            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        
        # We apply the new chosen action to the selected demand
        currentVal = env_hill_climb.step_hill_sp(action, source, dest)
        timer = tt.time()
        timesteps.append((timer-time_start_DRL, currentVal*(-1)))
    end = tt.time()
    return currentVal*(-1), end-start


def play_DRL_GNN_sp_hill_climbing_games_kp(tm_id, best_routing, list_of_demands_to_change, timesteps, time_start_DRL):
    # Here we use sp in hill climbing to select the middlepoint and to evaluate
    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT)
    env_hill_climb.seed(SEED)
    env_hill_climb.use_K_path = KP
    env_hill_climb.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT,
                                        NUM_ACTIONS, percentage_demands)

    currentVal = env_hill_climb.reset_DRL_hill_sp(tm_id, best_routing, list_of_demands_to_change)
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1:
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_DRL_sp_kp(env_hill_climb)
        # If the difference between the two edges is super small but non-zero, we break (this is because of precision reasons)
        if nextVal <= currentVal or (abs((-1) * nextVal - (-1) * currentVal) < 1e-4):
            break

        # Before we apply the new action, we need to remove the current allocation of the chosen demand
        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]

        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)

        # We apply the new chosen action to the selected demand
        currentVal = env_hill_climb.step_hill_sp(action, source, dest)
        timer = tt.time()
        timesteps.append((timer - time_start_DRL, currentVal * (-1)))
    end = tt.time()
    return currentVal * (-1), end - start

class SAPAgent:
    def __init__(self, env):
        self.K = env.K

    def act(self, env, demand, n1, n2):
        pathList = env.allPaths[str(n1) +':'+ str(n2)]
        path = 0
        allocated = 0 # Indicates 1 if we allocated the demand, 0 otherwise
        while allocated==0 and path < len(pathList) and path<self.K:
            currentPath = pathList[path]
            can_allocate = 1 # Indicates 1 if we can allocate the demand, 0 otherwise
            i = 0
            j = 1

            # 1. Iterate over pairs of nodes and check if we can allocate the demand
            while j < len(currentPath):
                link_capacity = env.links_bw[currentPath[i]][currentPath[j]]
                if (env.edge_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] + demand)/link_capacity > 1:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate==1:
                return path
            path = path + 1

        return -1

def play_sap_games(tm_id):
    env_sap = gym.make(ENV_SAP_AGENT)
    env_sap.seed(SEED)
    env_sap.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS)

    demand, source, destination = env_sap.reset(tm_id)
    sap_Agent = SAPAgent(env_sap)

    rewardAddTest = 0
    start = tt.time()
    while 1:
        action = sap_Agent.act(env_sap, demand, source, destination)

        done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_sap.step(action, demand, source, destination)
        if done:
            break
    end = tt.time()
    return maxLinkUti[2], end-start

def play_middRout_games(tm_id, env_middRout, agent):
    demand, source, destination = env_middRout.reset(tm_id)
    rewardAddTest = 0
    while 1:
        # Change to agent.pred_action_node_distrib_sp to choose the middlepoint using only the SPs
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout, source, destination)
        action = np.argmax(action_dist)
        
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout.step(action, demand, source, destination)
        rewardAddTest += reward
        if done:
            break
    return rewardAddTest, maxLinkUti[2], minLinkUti, utiStd


if __name__ == "__main__":

    hyper_parameter = {
        'feature_size': 20,
        't': 4,
        'readout_units': 20,
        'episode': 20,
        'lr': 0.0002,
        'gamma': 0.99,
        'alpha': 0.2,
        'batch_size': 55,
        'buffer_size': 10000,
        'update_freq': 100,
        'update_times': 10,
        'max_a_dim': 20,
        'avg_a_dim': 20
    }

    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-t', help='DEFO demands TM file id', type=str, required=True, nargs='+')
    parser.add_argument('-g', help='graph topology name', type=str, required=True, nargs='+')
    parser.add_argument('-m', help='model id whose weights to load', type=str, required=True, nargs='+')
    parser.add_argument('-o', help='Where to store the pckl file', type=str, required=True, nargs='+')
    parser.add_argument('-d', help='differentiation string', type=str, required=True, nargs='+')
    parser.add_argument('-f', help='general dataset folder name', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='specific dataset folder name', type=str, required=True, nargs='+')
    args = parser.parse_args()

    drl_eval_res_folder = args.o[0]
    tm_id = int(args.t[0])
    model_id = args.m[0]
    differentiation_str = args.d[0]
    graph_topology_name = args.g[0]
    general_dataset_folder = args.f[0]
    specific_dataset_folder = args.f2[0]

    timesteps = list()
    results = np.zeros(17)

    K_path = KP
    K = NUM_ACTIONS
    ########### The following lines of code is to evaluate a DRL SP-based agent
    env_DRL_SP = gym.make(ENV_MIDDROUT_AGENT_SP)
    env_DRL_SP.seed(SEED)
    env_DRL_SP.use_K_path = K_path
    env_DRL_SP.generate_environment(general_dataset_folder, graph_topology_name, EPISODE_LENGTH_MIDDROUT, K, percentage_demands)
    # Set to True f we want to take the top X% of the 5 most loaded links
    env_DRL_SP.top_K_critical_demands = True

    DRL_SP_Agent = SACD(hyper_parameter)
    if K_path:
        DRL_SP_Agent.K_path = K
        DRL_SP_Agent.target_entropy = 0.5 * (-np.log(1 / K))
    model_dir = "./models" + differentiation_str
    #model_id = 'final'
    DRL_SP_Agent.actor.load_state_dict(torch.load(model_dir + f"/actor_{model_id}.pt"))
    DRL_SP_Agent.actor.eval()
    # Restore variables on creation if a checkpoint exists.
    print("Restored DRL_SP model ", f"/actor_{model_id}.pt")

    ################################################

    # We can also use simulated annealing but it is going to take a while
    max_link_uti_sim_annealing, optim_cost_SA = 1,1 #play_sp_simulated_annealing_games(tm_id)
    
    max_link_uti_sp_hill_climb, optim_cost_HILL = 1,1 #play_sp_hill_climbing_games(tm_id)
    
    max_link_uti_SAP, optim_cost_SAP = 1, 1 #play_sap_games(tm_id)
    
    max_link_uti_DRL_SP, optim_cost_DRL_GNN, OSPF_init, best_routing, list_of_demands_to_change, time_start_DRL = play_middRout_games_sp(tm_id, env_DRL_SP, DRL_SP_Agent, timesteps)
    
    max_link_uti_DRL_SP_HILL, optim_cost_DRL_HILL = play_DRL_GNN_sp_hill_climbing_games_kp(tm_id, best_routing, list_of_demands_to_change, timesteps, time_start_DRL)

    new_timesteps = list()
    for elem in timesteps:
        new_timesteps.append((elem[0], elem[1], time_start_DRL, max_link_uti_DRL_SP))

    print("MAX UTI abans i despres d'optimitzar: ", OSPF_init, max_link_uti_DRL_SP, max_link_uti_DRL_SP_HILL, tm_id)

    results[3] = max_link_uti_DRL_SP_HILL 
    results[4] = max_link_uti_sim_annealing
    results[6] = len(env_DRL_SP.defoDatasetAPI.Gbase.edges()) # We store the number of edges to order the figures
    results[7] = max_link_uti_sp_hill_climb
    results[8] = max_link_uti_SAP
    results[9] = max_link_uti_DRL_SP
    results[11] = OSPF_init
    results[12] = optim_cost_SA
    results[13] = optim_cost_SAP
    results[14] = optim_cost_DRL_GNN
    results[15] = optim_cost_HILL
    results[16] = optim_cost_DRL_GNN+optim_cost_DRL_HILL

    path_to_pckl_rewards = drl_eval_res_folder + differentiation_str+ '/'+ graph_topology_name + '/'
    if not os.path.exists(path_to_pckl_rewards):
        os.makedirs(path_to_pckl_rewards)

    with open(path_to_pckl_rewards + graph_topology_name +'.' + str(tm_id) + ".pckl", 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
    with open(path_to_pckl_rewards + graph_topology_name +'.' + str(tm_id) + ".timesteps", 'w') as fp:
        json.dump(new_timesteps, fp)