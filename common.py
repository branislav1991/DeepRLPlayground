import numpy as np
import torch


def select_action_policy(state, policy_network, env):
    """Selects action following a sample of the policy network.

    Returns:
        Selected action.
    """
    with torch.no_grad():
        probs = policy_network(state)
        m = torch.distributions.Categorical(logits=probs)
        action = m.sample()

        action = action.cpu().numpy().item()
        return action


def select_action_dqn(state, dqn, env, exp_threshold):
    """Selects action either randomly or following policy given by dqn.
    """
    if np.random.rand() < exp_threshold:
        action = np.random.randint(env.action_space.n)
    else:
        dqn.eval()
        with torch.no_grad():
            action = torch.argmax(dqn.forward(state)).item()

    return action