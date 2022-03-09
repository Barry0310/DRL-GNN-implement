from actor_critic import AC
import gym
import gym_env


def input_transform(env, demand, state):
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
    actor_data = []
    if demand:
        for i in env.action_space[edges_dict[(demand[0], demand[1])]]:
            actor_data.append(
                {
                    'link_state': env.mark_action(i),
                    'pair': pair
                }
            )
    critic_data = {
        'link_state': state,
        'pair': pair
    }

    return actor_data, critic_data


def train(env, agent):
    """
    train loop
    """
    for e in range(agent.episode):
        done = False
        state, demand = env.reset(topology=0, demand_list=[(0, 2, 100)])  # Line 3
        agent.buffer_clear()
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
        advantages, returns, action_probs, c_vals = agent.compute_gae()  # Line 11
        actor_loss = agent.compute_actor_loss(advantages, action_probs)  # Line 12
        critic_loss = agent.compute_critic_loss(returns, c_vals)  # Line 13
        total_loss = actor_loss + critic_loss  # Line 14
        agent.compute_gradients(total_loss)  # Line 15, 16, 17


if __name__ == '__main__':
    ENV_NAME = 'GraphEnv-v1'
    env_training = gym.make(ENV_NAME)
    hyper_parameter = {
        'feature_size': 3,
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
