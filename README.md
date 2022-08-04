# Reinforcement-Learning

STEPS:

1-Initialize replay memory capacity.\n
2-Initialize the policy network with random weights.
3-Clone the policy network, and call it the target network.
4-For each episode:
  4.1-Initialize the starting state.
  4.2-For each time step:
       1-Select an action.
            *Via exploration or exploitation
       2-Execute selected action in an emulator.
       3-Observe reward and next state.
       4-Store experience in replay memory.
       5-Sample random batch from replay memory.
       6-Preprocess states from batch.
       7-Pass batch of preprocessed states to policy network.
       8-Calculate loss between output Q-values and target Q-values.
              *Requires a pass to the target network for the next state
       9-Gradient descent updates weights in the policy network to minimize loss.
              *After  time steps, weights in the target network are updated to the weights in the policy network.
        
        
