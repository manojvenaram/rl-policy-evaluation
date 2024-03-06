# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
## States
The environment has 7 states:

1.    Two Terminal States: G: The goal state & H: A hole state.
2.    Five Transition states / Non-terminal States including S: The starting state.
## Actions
The agent can take two actions:

1.    R: Move right.
2.    L: Move left.
## Transition Probabilities
The transition probabilities for each action are as follows:

1.    50% chance that the agent moves in the intended direction.
2.    33.33% chance that the agent stays in its current state.
3.    16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.
## Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.
## Graphical Representation
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/gra.png)
## POLICY EVALUATION FUNCTION
~~~
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V
~~~
## OUTPUT:
### Policy 1:
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/1.png)
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/2.png)
### Policy 2:
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/3.png)
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/4.png)
### Comparison:
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/5.png)
### Conclusion:
![](https://github.com/RanjithD18/rl-policy-evaluation/blob/main/6.png)

## RESULT:
Thus, a Python program is developed to evaluate the given policy.
