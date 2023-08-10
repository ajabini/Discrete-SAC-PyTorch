import gym
from Brain import Disc_DIAYN_Agent
from Common import Play, Logger, get_params, get_params_DIAYN
import numpy as np
from tqdm import tqdm


def concat_state_latent(s, z_, n_skill):
  if len(s.shape) == 3:
    # s is an RGB image
    s = s.reshape(-1)
  z_one_hot = np.zeros(n_skill)
  z_one_hot[z_] = 1
  return np.concatenate([s, z_one_hot])




if __name__ == "__main__":


    params = get_params_DIAYN()


    test_env = gym.make(params["env_name"])
    test_env = gym.wrappers.ResizeObservation(test_env, (84, 84))
    test_env = gym.wrappers.GrayScaleObservation(test_env)
    test_env = gym.wrappers.FrameStack(test_env, 4)

    state_shape = test_env.observation_space.shape
    n_actions = test_env.action_space.n
    # action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"state_shape": state_shape,
                   "n_actions": n_actions})
    print("params:", params)
    test_env.close()
    del test_env, state_shape, n_actions


    env = gym.make(params["env_name"])
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(params["seed"])
    env.action_space.seed(params["seed"])
    env.observation_space.seed(params["seed"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"]) # Starting with uniform distribution
    agent = Disc_DIAYN_Agent(p_z, **params)

    logger = Logger_DIAYN(agent, **params)

    if params["do_train"]:
        if not params["train_from_scratch"]:
            pass
        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch...")

        logger.on()
        max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset().__getitem__(slice(0,None,1))
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            for step in tqdm(range(1, 1+max_n_steps)):

                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state.__getitem__(slice(0,None,1)), z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                log_q_zs = agent.train()
                if log_q_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(log_q_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break


            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                       env.np_random.get_state(),
                       env.observation_space.np_random.get_state(),
                       env.action_space.np_random.get_state(),
                       *agent.get_rng_states(),
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()



