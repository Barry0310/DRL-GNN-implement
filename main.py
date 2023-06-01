import torch
from SACD import SACD
import gym
import gym_graph
import random
import numpy as np
import os
import gc
import time

if __name__ == '__main__':

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    SEED = 9
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    experiment_letter = "_B_SAC"
    take_critic_demands = True  # True if we want to take the demands from the most critical links, True if we want to take the largest
    percentage_demands = 15  # Percentage of demands that will be used in the optimization
    K_path = True
    K = 20
    str_perctg_demands = str(percentage_demands)
    percentage_demands /= 100

    max_step = 100
    EVALUATION_EPISODES = 20  # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM

    differentiation_str = "Enero_3top_" + str_perctg_demands + experiment_letter
    model_dir = "./models" + differentiation_str

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "w")

    ENV_NAME = 'GraphEnv-v16'

    training_tm_ids = set(range(100))

    hyper_parameter = {
        'feature_size': 20,
        't': 4,
        'readout_units': 20,
        'a_lr': 0.0003,
        'c_lr': 0.0015,
        'gamma': 0.9,
        'alpha': 0.1,
        'batch_size': 128,
        'buffer_size': 200000,
        'buffer_threshold': 10000,
        'update_freq': 8,
        'avg_a_dim': 1
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
    env_training1.use_K_path = K_path
    env_training1.generate_environment(dataset_folder_name1 + "/TRAIN", "BtAsiaPac", 0, K, percentage_demands)
    env_training1.top_K_critical_demands = take_critic_demands

    env_training2 = gym.make(ENV_NAME)
    env_training2.seed(SEED)
    env_training2.use_K_path = K_path
    env_training2.generate_environment(dataset_folder_name2 + "/TRAIN", "Garr199905", 0, K, percentage_demands)
    env_training2.top_K_critical_demands = take_critic_demands

    env_training3 = gym.make(ENV_NAME)
    env_training3.seed(SEED)
    env_training3.use_K_path = K_path
    env_training3.generate_environment(dataset_folder_name3 + "/TRAIN", "Goodnet", 0, K, percentage_demands)
    env_training3.top_K_critical_demands = take_critic_demands

    env_training = [env_training1, env_training2, env_training3]

    env_eval1 = gym.make(ENV_NAME)
    env_eval1.seed(SEED)
    env_eval1.use_K_path = K_path
    env_eval1.generate_environment(dataset_folder_name1 + "/EVALUATE", "BtAsiaPac", 0, K, percentage_demands)
    env_eval1.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval2.seed(SEED)
    env_eval2.use_K_path = K_path
    env_eval2.generate_environment(dataset_folder_name2 + "/EVALUATE", "Garr199905", 0, K, percentage_demands)
    env_eval2.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval3.seed(SEED)
    env_eval3.use_K_path = K_path
    env_eval3.generate_environment(dataset_folder_name3 + "/EVALUATE", "Goodnet", 0, K, percentage_demands)
    env_eval3.top_K_critical_demands = take_critic_demands

    env_eval = [env_eval1, env_eval2, env_eval3]

    max_a_dim = 0
    a_dim = []
    if not K_path:
        for env in env_eval:
            for action_space in env.src_dst_k_middlepoints.items():
                a_dim.append(len(action_space[1]))
                max_a_dim = max(max_a_dim, len(action_space[1]))
        hyper_parameter['max_a_dim'] = max_a_dim
        hyper_parameter['avg_a_dim'] = round(sum(a_dim)/len(a_dim))

    counter_store_model = 0
    total_step = 0
    actor_loss, critic_loss = 0, 0
    max_reward = -100000
    AC_policy = SACD(hyper_parameter)
    if K_path:
        AC_policy.K_path = K
        AC_policy.target_entropy = 0.3 * (-np.log(1 / K))
    for iters in range(100):

        for e in range(20):

            print(f"Episode {iters*20+e}")

            for topo in range(len(env_training)):
                tm_id = random.sample(training_tm_ids, 1)[0]
                timer_a = time.time()
                step = 0
                while step < max_step:
                    demand, src, dst = env_training[topo].reset(tm_id=tm_id)
                    while True:
                        action_dist, tensor = AC_policy.predict(env_training[topo], src, dst, demand)

                        if total_step < hyper_parameter['buffer_threshold']:
                            action = np.random.choice(K)
                        else:
                            action = np.random.choice(len(action_dist), p=action_dist.cpu().detach().numpy())
                        reward, done, _, demand, src, dst, _, _, _ = env_training[topo].step(action, demand, src, dst)

                        AC_policy.add_exp(env_training[topo], tensor, src, dst, demand, action, reward, done)

                        total_step += 1
                        step += 1

                        if total_step >= hyper_parameter['buffer_threshold'] and total_step % hyper_parameter['update_freq']==0:
                            actor_loss, critic_loss = AC_policy.train()

                        if done:
                            break
                        if step >= max_step:
                            break
                timer_b = time.time()
                print("topo ", topo, timer_b - timer_a, "sec")
            print(total_step)

            if actor_loss != None and critic_loss != None:
                fileLogs.write("a," + str(actor_loss) + ",\n")
                fileLogs.write("c," + str(critic_loss) + ",\n")
                fileLogs.flush()

            rewards_test = np.zeros(EVALUATION_EPISODES * 3)
            error_links = np.zeros(EVALUATION_EPISODES * 3)
            max_link_utis = np.zeros(EVALUATION_EPISODES * 3)
            min_link_utis = np.zeros(EVALUATION_EPISODES * 3)
            uti_stds = np.zeros(EVALUATION_EPISODES * 3)

            for topo in range(len(env_eval)):
                for tm_id in range(EVALUATION_EPISODES):
                    demand, src, dst = env_eval[topo].reset(tm_id=tm_id)
                    total_reward = 0
                    mlu_reward = 0
                    uti_reward = 0
                    posi = EVALUATION_EPISODES * topo + tm_id
                    while True:
                        with torch.no_grad():
                            action_dist, _ = AC_policy.predict(env_eval[topo], src, dst, demand)
                        action = torch.argmax(action_dist)

                        reward, done, error_eval_links, demand, src, dst, max_link_uti, min_link_uti, uti_std = \
                            env_eval[topo].step(action, demand, src, dst)

                        total_reward += reward
                        mlu_reward += error_eval_links
                        uti_reward += min_link_uti
                        if done:
                            break
                    rewards_test[posi] = total_reward
                    error_links[posi] = mlu_reward
                    max_link_utis[posi] = max_link_uti[2]
                    min_link_utis[posi] = uti_reward
                    uti_stds[posi] = uti_std

            eval_mean_reward = np.mean(rewards_test)
            fileLogs.write(";," + str(np.mean(uti_stds)) + ",\n")
            fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
            fileLogs.write("<," + str(np.amax(max_link_utis)) + ",\n")
            fileLogs.write(">," + str(np.mean(min_link_utis)) + ",\n")
            fileLogs.write("ENTR," + str(AC_policy.alpha) + ",\n")
            fileLogs.write("REW," + str(eval_mean_reward) + ",\n")
            fileLogs.write("lr," + str(0.0002) + ",\n")

            if eval_mean_reward > max_reward:
                max_reward = eval_mean_reward
                fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(counter_store_model) + ",\n")
                torch.save(AC_policy.actor.state_dict(), model_dir + '/' + f'actor_{counter_store_model}.pt')
                torch.save(AC_policy.critic.state_dict(), model_dir + '/' + f'critic_{counter_store_model}.pt')
                counter_store_model += 1

            fileLogs.flush()

            gc.collect()
    fileLogs.close()
    torch.save(AC_policy.actor.state_dict(), model_dir + '/' + f'actor_final.pt')
    torch.save(AC_policy.critic.state_dict(), model_dir + '/' + f'critic_final.pt')




