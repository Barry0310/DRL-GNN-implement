from actor_critic import AC


if __name__ == '__main__':
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
