# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

# def player(prev_play, opponent_history=[]):
#     opponent_history.append(prev_play)

#     guess = "R"
#     if len(opponent_history) > 2:
#         guess = opponent_history[-2]

#     return guess

# Just beat the last choice
# def player(prev_play, opponent_history=[]):
#     opponent_history.append(prev_play)

#     choices = ["R", "P", "S"]
#     opponent_last_move = opponent_history[-1]
#     if opponent_last_move not in choices:
#         guess = "R"
#     else:
#         # Get move that would beat opponent's last move
#         opponent_last_move_index = choices.index(opponent_last_move)
#         guess_index = (opponent_last_move_index - 1) % 3 
#         guess = choices[guess_index]

#     return guess


# Given a move, returns the move that beats it
def get_winning_move(opponent_move):
    choices = ['R', 'P', 'S']
    opponent_last_move_index = choices.index(opponent_move)
    guess_index = (opponent_last_move_index + 1) % 3 
    winning_move = choices[guess_index]
    return winning_move

# DQN table approach
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

def build_model(state_move_count, learning_rate=0.001, hidden_layers=8) -> keras.Model:
    input_size = state_move_count * 6
    model =  keras.Sequential([
        keras.layers.Dense(hidden_layers, activation='relu', input_shape=(input_size,)),
            # keras.layers.Dense(hidden_layers, activation='relu'),
            keras.layers.Dense(3)  # Outputs Q-values for [R, P, S]
        ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# --- Sequential Replay Buffer ---
class SequentialReplayBuffer:
    def __init__(self, sequence_length=5):
        self.buffer = []
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def get_recent(self):
        if len(self.buffer) < self.sequence_length:
            return None
        sequence = self.buffer[-self.sequence_length:]
        states, actions, rewards, next_states = zip(*sequence)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def size(self):
        return len(self.buffer)

class RPSBot:
    def __init__(self, moves_per_state=3, max_training_state_count = 5, epsilon = 0.2, epsilon_decay = 0.0, gamma=0.8) -> None:
        self.my_history = []
        self.opponent_history = []
        self.actions = ['R', 'P', 'S']
        self.moves_per_state = moves_per_state
        self.max_training_state_count = max_training_state_count
        self.model = build_model(moves_per_state)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.buffer = SequentialReplayBuffer(sequence_length=max_training_state_count)
        self.prev_action = None
        self.prev_state = None

    def encode_state(self, opponent_history, my_history):
        my_history_trimmed = my_history[-self.moves_per_state:]
        opponent_history_trimmed = opponent_history[-self.moves_per_state:]
        
        key = {
            'R' : [1,0,0],
            'P' : [0,1,0],
            'S' : [0,0,1],
               }

        my_history_formatted = [key.get(entry, [0,0,0]) for entry in my_history_trimmed]
        opponent_history_formatted = [key.get(entry, [0,0,0]) for entry in opponent_history_trimmed]

        missing_my_entries = self.moves_per_state - len(my_history_trimmed)
        for _ in range(missing_my_entries):
            my_history_formatted.insert(0, [0,0,0])
        
        missing_opponent_entries = self.moves_per_state - len(opponent_history_trimmed)
        for _ in range(missing_opponent_entries):
            opponent_history_formatted.insert(0, [0,0,0])
        
        my_history_encoded = np.hstack(my_history_formatted)
        opponent_history_encoded = np.hstack(opponent_history_formatted)
        encoded = np.concatenate([my_history_encoded, opponent_history_encoded])
        return encoded.astype(np.float32)
        
    def choose_action(self, state_vector, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        q_values = self.model.predict(np.array([state_vector]), verbose=False)  # Shape: (1,3)
        return self.actions[np.argmax(q_values[0])]

    def get_reward(self, my_move, opponent_move):
        # Standard RPS reward: win=+1, tie=0, loss=-1
        if my_move == opponent_move:
            return 0
        winning_move = get_winning_move(opponent_move)
        if my_move == winning_move:
            return 1
        else:
            return -1
    
    def update(self, opponent_history, my_history):
        current_state = self.encode_state(opponent_history, my_history)
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.get_reward(self.prev_action, opponent_history[-1])
            self.buffer.add(self.prev_state, self.actions.index(self.prev_action), reward, current_state)
            recent_buffer = self.buffer.get_recent()
            if recent_buffer is not None:
                states, actions, rewards, next_states = recent_buffer
                # Predict Q-values for states and next_states in the sequence
                q_vals = self.model.predict(states, verbose=False)      # shape: (seq_len, 3)
                next_q_vals = self.model.predict(next_states, verbose=False)
                # Update targets for each transition in the sequence using one-step returns
                for i in range(len(states)):
                    target = rewards[i] + self.gamma * np.max(next_q_vals[i])
                    q_vals[i][actions[i]] = target
                # Train the network on the entire contiguous batch
                self.model.train_on_batch(states, q_vals)
                self.epsilon *= (1-self.epsilon_decay)
        # Choose the next action based on the current state
        action = self.choose_action(current_state, self.epsilon)
        self.prev_state = current_state
        self.prev_action = action
        return action

bot = None
def player(prev_play, opponent_history=[], my_history=[], epsilon=0.7, epsilon_decay=0.01, gamma=0.9, moves_per_state=3,max_training_state_count=5):
    global bot
    if prev_play == '':
        # Initialise bot and history
        bot = RPSBot(moves_per_state=moves_per_state,max_training_state_count=max_training_state_count, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma)
        opponent_history=[]
        my_history=[]
    opponent_history.append(prev_play)
    action = bot.update(opponent_history, my_history)
    my_history.append(action)
    return action


# state_counts = {}
# moves = ['R', 'P', 'S']

# def player(prev_play, opponent_history=[]):
#     global state_counts
#     if prev_play == "":
#         state_counts = {}
#         opponent_history.clear()
#     else:
#         opponent_history.append(prev_play)
    
#     n = 3

#     hist = opponent_history

#     guess = "R"
#     if len(hist) > n:
#         pattern = join(hist[-n:])

#         if join(hist[-(n + 1):]) in state_counts.keys():
#             state_counts[join(hist[-(n + 1):])] += 1
#         else:
#             state_counts[join(hist[-(n + 1):])] = 1

#         options = [pattern + x for x in moves]

#         for i in options:
#             if not i in state_counts.keys():
#                 state_counts[i] = 0

#         most_common_next_state = max(options, key=lambda key: state_counts[key])

#         guess = get_winning_move(most_common_next_state[-1])

#     return guess


# def join(moves):
#     return "".join(moves)
