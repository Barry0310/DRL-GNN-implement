from actor_critic import AC
import gym
import gym_env


def input_transform(env, demand):
    """
    mark actions of demand on current state individually
    """
    neighbor_edges = env.neighbor_edges
    edges_dict = env_training.edges_dict
    pair = []
    for i in neighbor_edges:
        for j in neighbor_edges[i]:
            pair.append([edges_dict[i]] * 3)
            pair.append([edges_dict[j]] * 3)
    data = []
    for i in env.action_space[(demand[0], demand[1])]:
        data.append(
            {
                'link_state': env.mark_action(i),
                'pair': pair
            }
        )

    return data


def train(env, agent):
    """
    train loop
    """
    done = False
    for e in range(agent.episode):
        state, demand = env.reset(topology=0, demand_list=[(0, 2, 100)])
        input = input_transform(env, demand)
        while not done:
            pass


if __name__ == '__main__':
    ENV_NAME = 'GraphEnv-v1'
    env_training = gym.make(ENV_NAME)
    hyper_parameter = {
        'feature_size': 20,
        't': 5,
        'readout_units': 20,
        'episode': 1,
        'lr': 0.0002,
        'gae_gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_value': 0.5,
    }
    AC_policy = AC(hyper_parameter)
    train(env_training, AC_policy)
