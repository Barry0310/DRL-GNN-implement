import torch.nn.init

from actor_critic import PPOAC
import gym
import gym_graph
#from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import os
import gc


"""
def train(env, agent):

    writer = SummaryWriter('./runs/2')
    for e in range(agent.episode):
        print('#episode', e+1)
        done = False
        state, demand = env.reset(topology=2)  # Line 3
        print('max link utilization:', env.max_util)
        while not done:
            actor_input, critic_input = input_transform(env, demand, state)
            act_dist, c_val = agent.predict(actor_input, critic_input)  # Line 5, 6
            a, pa = agent.choose_action(act_dist)  # Line 7
            next_state, done, next_demand, reward = env.step(a)  # Line 8
            agent.store_result(pa, c_val, a, demand, done, reward)  # Line 9
            state = next_state
            demand = next_demand
        _, critic_input = input_transform(env, demand, state)
        _, c_val = agent.predict([], critic_input)  # Line 10
        agent.store_result(critic_value=c_val)
        advantages, returns, action_probs, c_vals, entropy = agent.compute_gae()  # Line 11
        actor_loss = agent.compute_actor_loss(advantages, action_probs)  # Line 12
        critic_loss = agent.compute_critic_loss(returns, c_vals)  # Line 13
        total_loss = actor_loss + critic_loss - entropy  # Line 14
        agent.compute_gradients(total_loss)  # Line 15, 16, 17
        writer.add_scalar('loss', total_loss.item(), e + 1)
        print('max link utilization:', env.max_util)
    print('Training finish')
    writer.close()
"""



if __name__ == '__main__':

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    SEED = 9
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(1)
    experiment_letter = "_B_NEW"
    take_critic_demands = True  # True if we want to take the demands from the most critical links, True if we want to take the largest
    percentage_demands = 15  # Percentage of demands that will be used in the optimization
    str_perctg_demands = str(percentage_demands)
    percentage_demands /= 100

    max_iters = 150
    EVALUATION_EPISODES = 20  # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM
    PPO_EPOCHS = 8
    num_samples_top1 = int(np.ceil(percentage_demands * 380)) * 5
    num_samples_top2 = int(np.ceil(percentage_demands * 506)) * 4
    num_samples_top3 = int(np.ceil(percentage_demands * 272)) * 6

    num_samples_top = [num_samples_top1, num_samples_top2, num_samples_top3]

    BUFF_SIZE = num_samples_top1 + num_samples_top2 + num_samples_top3

    differentiation_str = "Enero_3top_" + str_perctg_demands + experiment_letter
    checkpoint_dir = "./models" + differentiation_str

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    ENV_NAME = 'GraphEnv-v16'

    training_tm_ids = set(range(100))

    hyper_parameter = {
        'feature_size': 20,
        't': 5,
        'readout_units': 20,
        'episode': 20,
        'lr': 0.0002,
        'lr_decay_rate': 0.96,
        'lr_decay_step': 60,
        'mini_batch': 55,
        'gae_gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_value': 0.5,
        'entropy_beta': 0.01,
        'entropy_step': 60,
        'l2_regular': 0.0001
    }

    dataset_root_folder = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"
    dataset_folder_name1 = "NEW_BtAsiaPac"
    dataset_folder_name2 = "NEW_Garr199905"
    dataset_folder_name3 = "NEW_Goodnet"

    dataset_folder_name1 = dataset_root_folder + dataset_folder_name1
    dataset_folder_name2 = dataset_root_folder + dataset_folder_name2
    dataset_folder_name3 = dataset_root_folder + dataset_folder_name3

    env_training1 = gym.make(ENV_NAME)
    env_training1.seed(SEED)
    env_training1.generate_environment(dataset_folder_name1 + "/TRAIN", "BtAsiaPac", 0, 100, percentage_demands)
    env_training1.top_K_critical_demands = take_critic_demands

    env_training2 = gym.make(ENV_NAME)
    env_training2.seed(SEED)
    env_training2.generate_environment(dataset_folder_name2 + "/TRAIN", "Garr199905", 0, 100, percentage_demands)
    env_training2.top_K_critical_demands = take_critic_demands

    env_training3 = gym.make(ENV_NAME)
    env_training2.seed(SEED)
    env_training2.generate_environment(dataset_folder_name3 + "/TRAIN", "Goodnet", 0, 100, percentage_demands)
    env_training2.top_K_critical_demands = take_critic_demands

    env_training = [env_training1, env_training2, env_training3]

    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name1 + "/EVALUATE", "BtAsiaPac", 0, 100, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name2 + "/EVALUATE", "Garr199905", 0, 100, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name3 + "/EVALUATE", "Goodnet", 0, 100, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands

    AC_policy = PPOAC(hyper_parameter)
    for iters in range(150):
        for e in range(hyper_parameter['episode']):

            print(f"Episode {iters*hyper_parameter['episode']+e}")

            critic_features = []
            tensors = []
            actions = []
            values = []
            masks = []
            rewards = []
            actions_probs = []

            total_num_samples = 0

            for topo in range(len(env_training)):
                print(f"topo {topo+1}")
                number_samples_reached = False
                total_num_samples += num_samples_top[topo]
                while not number_samples_reached:
                    tm_id = random.sample(training_tm_ids, 1)[0]
                    demand, src, dst = env_training[topo].reset(tm_id=tm_id)
                    while True:
                        action_dist, tensor = AC_policy.predict(env_training[topo], src, dst)

                        critic_feature = AC_policy.critic_get_graph_features(env_training[topo])
                        value = AC_policy.critic(critic_feature)[0]

                        action = np.random.choice(len(action_dist), p=action_dist.detach().numpy())
                        action_one_hot = np.eye(len(action_dist))[action]
                        reward, done, _, new_demand, new_src, new_dst, _, _, _ = env_training[topo].step(action,
                                                                                                         demand,
                                                                                                         src,
                                                                                                         dst)
                        mask = not done

                        tensors.append(tensor)
                        critic_features.append(critic_feature)
                        actions.append(action_one_hot)
                        values.append(value)
                        masks.append(mask)
                        rewards.append(reward)
                        actions_probs.append(action_dist)

                        demand = new_demand
                        source = new_src
                        destination = new_dst

                        if len(tensors) == total_num_samples:
                            number_samples_reached = True
                            break

                        if done:
                            break





