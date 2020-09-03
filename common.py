import numpy as np
import torch


def select_action_policy(state, policy_network, value_network, env):
    """Selects action following a sample of the policy network.

    Returns:
        (selected action, log probability of action, value).
    """
    with torch.no_grad():
        val = value_network.forward(state).squeeze(0)
        probs = policy_network(state).squeeze(0)
        m = torch.distributions.Categorical(logits=probs)
        action = m.sample()

        log_p = m.log_prob(action)
        action = action.cpu().numpy().item()
        return action, log_p, val


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