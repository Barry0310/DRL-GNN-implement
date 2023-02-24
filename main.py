import torch
from actor_critic import PPOAC
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
    experiment_letter = "_B_PATH_LINK_TEST"
    take_critic_demands = True  # True if we want to take the demands from the most critical links, True if we want to take the largest
    percentage_demands = 15  # Percentage of demands that will be used in the optimization
    str_perctg_demands = str(percentage_demands)
    percentage_demands /= 100

    max_iters = 150
    EVALUATION_EPISODES = 20  # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM

    num_samples_top1 = int(np.ceil(percentage_demands * 380)) * 5
    num_samples_top2 = int(np.ceil(percentage_demands * 506)) * 4
    num_samples_top3 = int(np.ceil(percentage_demands * 272)) * 6

    num_samples_top = [num_samples_top1, num_samples_top2, num_samples_top3]

    differentiation_str = "Enero_3top_" + str_perctg_demands + experiment_letter
    model_dir = "./models" + differentiation_str

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "w")

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
        'buffer_size': num_samples_top1 + num_samples_top2 + num_samples_top3,
        'update_times': 8
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
    env_training3.seed(SEED)
    env_training3.generate_environment(dataset_folder_name3 + "/TRAIN", "Goodnet", 0, 100, percentage_demands)
    env_training3.top_K_critical_demands = take_critic_demands

    env_training = [env_training1, env_training2, env_training3]

    env_eval1 = gym.make(ENV_NAME)
    env_eval1.seed(SEED)
    env_eval1.generate_environment(dataset_folder_name1 + "/EVALUATE", "BtAsiaPac", 0, 100, percentage_demands)
    env_eval1.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval2.seed(SEED)
    env_eval2.generate_environment(dataset_folder_name2 + "/EVALUATE", "Garr199905", 0, 100, percentage_demands)
    env_eval2.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval3.seed(SEED)
    env_eval3.generate_environment(dataset_folder_name3 + "/EVALUATE", "Goodnet", 0, 100, percentage_demands)
    env_eval3.top_K_critical_demands = take_critic_demands

    env_eval = [env_eval1, env_eval2, env_eval3]

    counter_store_model = 0
    max_reward = -1000
    AC_policy = PPOAC(hyper_parameter)
    for iters in range(100):

        if iters * hyper_parameter['episode'] >= hyper_parameter['entropy_step']:
            AC_policy.entropy_beta = hyper_parameter['entropy_beta'] / 10
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

            timer_a = time.time()
            AC_policy.actor.train()
            AC_policy.critic.train()

            for topo in range(len(env_training)):
                print(f"topo {topo+1}")
                number_samples_reached = False
                total_num_samples += num_samples_top[topo]
                tm_id = random.sample(training_tm_ids, 1)[0]
                while not number_samples_reached:
                    demand, src, dst = env_training[topo].reset(tm_id=tm_id)
                    while True:
                        action_dist, tensor = AC_policy.predict(env_training[topo], src, dst)

                        critic_feature = AC_policy.critic_get_graph_features(env_training[topo])
                        value = AC_policy.critic(critic_feature)[0]

                        action = np.random.choice(len(action_dist), p=action_dist.cpu().detach().numpy())
                        action_one_hot = torch.nn.functional.one_hot(torch.tensor(action), num_classes=len(action_dist))
                        reward, done, _, demand, src, dst, _, _, _ = env_training[topo].step(action, demand, src, dst)
                        mask = not done

                        tensors.append(tensor)
                        critic_features.append(critic_feature)
                        actions.append(action_one_hot)
                        values.append(value.cpu().detach())
                        masks.append(mask)
                        rewards.append(reward)
                        actions_probs.append(action_dist)

                        if len(tensors) == total_num_samples:
                            number_samples_reached = True
                            break

                        if done:
                            break

            critic_feature = AC_policy.critic_get_graph_features(env_training[-1])
            value = AC_policy.critic(critic_feature)[0]
            values.append(value.cpu().detach())
            timer_b = time.time()
            print("collect_data", timer_b - timer_a, "sec")

            timer_a = time.time()
            returns, advantages = AC_policy.compute_gae(values, masks, rewards)
            actor_loss, critic_loss = AC_policy.update(actions, actions_probs, tensors, critic_features, returns,
                                                       advantages)
            if AC_policy.scheduler.get_last_lr()[0] > 0.0001:
                AC_policy.scheduler.step()
            timer_b = time.time()
            print("update", timer_b - timer_a, "sec")

            fileLogs.write("a," + str(actor_loss.cpu().detach().numpy()) + ",\n")
            fileLogs.write("c," + str(critic_loss.cpu().detach().numpy()) + ",\n")
            fileLogs.flush()

            rewards_test = np.zeros(EVALUATION_EPISODES * 3)
            error_links = np.zeros(EVALUATION_EPISODES * 3)
            max_link_utis = np.zeros(EVALUATION_EPISODES * 3)
            min_link_utis = np.zeros(EVALUATION_EPISODES * 3)
            uti_stds = np.zeros(EVALUATION_EPISODES * 3)

            AC_policy.actor.eval()
            AC_policy.critic.eval()

            timer_a = time.time()
            for topo in range(len(env_eval)):
                for tm_id in range(EVALUATION_EPISODES):
                    demand, src, dst = env_eval[topo].reset(tm_id=tm_id)
                    total_reward = 0
                    posi = EVALUATION_EPISODES * topo + tm_id
                    while True:
                        action_dist, _ = AC_policy.predict(env_eval[topo], src, dst)
                        action = torch.argmax(action_dist)

                        reward, done, error_eval_links, demand, src, dst, max_link_uti, min_link_uti, uti_std = \
                            env_eval[topo].step(action, demand, src, dst)

                        total_reward += reward
                        if done:
                            break
                    rewards_test[posi] = total_reward
                    error_links[posi] = error_eval_links
                    max_link_utis[posi] = max_link_uti[2]
                    min_link_utis[posi] = min_link_uti
                    uti_stds[posi] = uti_std

            timer_b = time.time()
            print("eval", timer_b - timer_a, "sec")
            eval_mean_reward = np.mean(rewards_test)
            fileLogs.write(";," + str(np.mean(uti_stds)) + ",\n")
            fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
            fileLogs.write("<," + str(np.amax(max_link_utis)) + ",\n")
            fileLogs.write(">," + str(np.amax(min_link_utis)) + ",\n")
            fileLogs.write("ENTR," + str(AC_policy.entropy_beta) + ",\n")
            fileLogs.write("REW," + str(eval_mean_reward) + ",\n")
            fileLogs.write("lr," + str(AC_policy.scheduler.get_last_lr()[0]) + ",\n")

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




