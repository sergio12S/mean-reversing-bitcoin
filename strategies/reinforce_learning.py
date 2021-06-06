from tqdm import tqdm
import matplotlib.pyplot as plt
from random import sample
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
import pandas as pd


class ModelAgent():
    def __init__(
        self,
        name,
        type_model,
        window_size,
        architecture,
        l2_reg,
        learning_rate
    ):
        self.name = name
        self.type_model = type_model
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.online_network = self.build_model(
            input_dim=window_size,
            output_dim=3, trainable=True)

        self.target_network = self.build_model(
            input_dim=window_size,
            output_dim=3, trainable=False)

        self._update_target()

    def trader_model1(self, input_dim, output_dim, trainable=True):
        layers = []
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=input_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=output_dim,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def build_model(self, input_dim, output_dim, trainable):
        if self.type_model == 'lstm':
            model = self.lstm_model(
                input_dim=input_dim,
                output_dim=output_dim,
                trainable=trainable)
            return model
        if self.type_model in ('deep_net', 'forward_net'):
            model = self.deep_net_model(
                input_dim=input_dim,
                output_dim=output_dim,
                trainable=trainable)
            return model
        if self.type_model == 'trader_0':
            model = self.trader_model0(
                input_dim=input_dim,
                output_dim=output_dim,
                trainable=trainable
            )
            return model
        if self.type_model == 'trader_1':
            model = self.trader_model1(
                input_dim=input_dim,
                output_dim=output_dim,
                trainable=trainable
            )
            return model

    def save_model(self, memory, name):
        self.online_network.save_weights(
            f"models/{self.name}_ai_trader_{name}.h5")

        with open(
            f'models/{self.name}_memory_{name}.pickle',
                'wb') as p:
            pickle.dump(memory, p, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        self.online_network.load_weights(
            f"models/{self.name}_ai_trader_{name}.h5")
        self._update_target()
        with open(
            f'models/{self.name}_memory_{name}.pickle',
                'rb') as p:
            memory = pickle.load(p)
        return memory

    def _update_target(self):
        '''
        Update waits in target model
        '''
        self.target_network.set_weights(self.online_network.get_weights())


class Environment():
    def __init__(self, type_model):
        self.type_model = type_model
        self.state = []

    def state_create(self, data, window_size, iterator):
        if self.type_model == 'trader_1':
            return data[iterator]

    def state_get(self):
        return self.state


class Experience():
    def __init__(self):
        self.memory = []

    def get_memory(self):
        return self.memory

    def set_memory(self, x):
        self.memory = x

    def add_memory(self, x):
        self.memory.append(x)

    def size_memory(self):
        return len(self.memory)

    def minibatch(self, batch_size, randomly=True, que=500):
        if batch_size > len(self.memory):
            return
        if len(self.memory) > que:
            self.memory = self.memory[-que:]
        if randomly:
            minibatch = sample(self.memory, batch_size)
            return minibatch
        if not randomly:
            minibatch = self.memory[-batch_size:]
            return minibatch


class Reward():
    def __init__(self, window=2):
        self.wait_window = 0
        self.window = window
        self.position = {
            'time_open': None,
            'time_close': None,
            'status': 'close',
            'signal': None,
            'reward': None,
            'open_price': None,
            'close_price': None
        }
        self.balance = []
        self.total_profit = 0
        self.epochs_profit = []

    def get_epochs_profit(self):
        return self.epochs_profit

    def add_epochs_profit(self, data):
        self.epochs_profit.append(data)

    def get_result(self):
        return self.total_profit

    def check_status_position(self):
        return self.position['status']

    def reset_balance(self):
        self.balance = []
        self.total_profit = 0
        self.position = {
            'time_open': None,
            'time_close': None,
            'status': 'close',
            'signal': None,
            'reward': None,
            'open_price': None,
            'close_price': None
        }

    def reset_position(self):
        self.position = {
            'time_open': None,
            'time_close': None,
            'status': 'close',
            'signal': None,
            'reward': None,
            'open_price': None,
            'close_price': None
        }

    def _open_long(self, data):
        self.position['time_open'] = data['time']
        self.position['status'] = 'long'
        self.position['open_price'] = data['open_price']
        self.position['close_price'] = data['close_price']
        self.position['reward'] = data['reward']

        self.position['signal'] = 1

    def _open_short(self, data):
        self.position['time_open'] = data['time']
        self.position['status'] = 'short'
        self.position['open_price'] = data['open_price']
        self.position['close_price'] = data['close_price']
        self.position['signal'] = -1
        self.position['reward'] = data['reward']

    def _reward_long(self, data):
        self.wait_window = 0

        self.position['status'] = 'close'
        self.position['time_close'] = data['time']
        self.position['reward'] = (
            self.position['close_price'] - self.position['open_price']) \
            * self.position['signal']
        self.balance.append(dict(self.position))
        self.total_profit += self.position['reward']
        reward = max(self.position['reward'], 0)
        self.reset_position()
        return reward

    def _reward_short(self, data):
        self.wait_window = 0

        self.position['status'] = 'close'
        self.position['time_close'] = data['time']
        self.position['reward'] = (
            self.position['close_price'] - self.position['open_price']) \
            * self.position['signal']
        self.balance.append(dict(self.position))
        self.total_profit += self.position['reward']
        reward = max(self.position['reward'], 0)
        self.reset_position()
        return reward

    def _reward_hold(self):
        ''' cash position '''
        return 10

    def _current_long(self):
        if self.position['status'] == 'long':
            return self.position['open_price']

    def _current_short(self):
        if self.position['status'] == 'short':
            return self.position['open_price']

    def _managment_take(self,
                        action, status_position,
                        data):
        # Cash position
        if action == 0 and status_position == 'close':
            return 5
        # Open long position
        if action == 1 and status_position == 'close':
            self._open_long(data=data)
            return 5
        # Open short position
        if action == 2 and status_position == 'close':
            self._open_short(data=data)
            return 5

        # Hold and Close the long position
        if status_position == 'long':
            # self.wait_window += 1
            # if self.wait_window < self.window:
            #     return 5
            # if self.wait_window == self.window:
            return self._reward_long(data=data)

        # Hold and Close the short position
        if status_position == 'short':
            # self.wait_window += 1
            # if self.wait_window < self.window:
            #     return 5
            # if self.wait_window == self.window:
            return self._reward_short(data=data)

    def plot_profit(self, name='', show=True, iter=1):
        data = list(map(lambda x: x['reward'], self.balance))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.cumsum(data), 'o-')
        ax.set_title('Balance')
        if show:
            plt.show()
            return
        plt.savefig(f"results/{name} + '_' + {iter}+ _.png")


class Agent():

    def __init__(self,
                 name,
                 window,
                 type_model,
                 data,
                 window_size,
                 batch_size,
                 action_space,
                 episodes,
                 memory_size,
                 X_var,
                 Y_var,
                 architecture
                 ):
        '''
        action space: buy, sell, hold
        '''
        self.name = name
        self.type_model = type_model
        self.data = data
        self.X_var = X_var
        self.Y_var = Y_var
        self.window_size = window_size
        self.action_space = action_space
        self.batch_size = batch_size
        self.architecture = architecture
        self.memory = Experience()
        self.env = Environment(type_model=type_model)
        self.reward = Reward(window=window)
        self.model = ModelAgent(
            name=self.name,
            type_model=self.type_model,
            window_size=self.window_size,
            architecture=self.architecture,
            l2_reg=1e-6,
            learning_rate=0.0001
        )
        self.idx = tf.range(batch_size)
        self.count = 0
        self.episodes = episodes
        self.memory_size = memory_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_history = []

    def plot_epsilon(self):
        plt.plot(self.epsilon_history)

    def _load_model(self, name):
        self.model.online_network.load_weights(
            f"models/{self.name}_ai_trader_{name}.h5"
        )
        self.model._update_target()
        with open(
            f'models/{self.name}_memory_{name}.pickle',
                'rb') as p:
            self.memory.set_memory(pickle.load(p)
                                   )

    def _update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
            self.epsilon_history.append(self.epsilon)
        # if self.total_profit < 0:
        #     self.epsilon *= 1+(1-self.epsilon_decay)

    def epsilon_greedy_policy(self, state):
        '''
        Epsilon greedy police
        Greedy policy: gets predicted Q-values from the DQN and finds the index
        of the action that corresponds to the highest Q-value
        '''
        self.count += 1
        random_value = np.random.rand()
        if random_value <= self.epsilon:
            actions = np.random.choice(self.action_space)
            return actions
        else:
            # Define q-parameters
            q = self.model.online_network.predict(
                state.reshape(1, self.window_size))
            actions = np.argmax(q, axis=1).squeeze()
            return actions

    def experience_replay(self):
        """ vectorized implementation; 30x speed up compared with for loop """
        if self.batch_size > self.memory.size_memory():
            return
        batch = self.memory.minibatch(
            batch_size=self.batch_size, randomly=True, que=self.memory_size)
        batch = map(np.array, zip(*batch))

        self._dqn(batch=batch)
        if self.count % 10 == 0:
            self.model._update_target()

    def _dqn(self, batch):
        states, actions, rewards, next_states, not_done = batch

        states = states.reshape(-1, self.window_size)
        next_states = next_states.reshape(-1, self.window_size)

        q_target = self.model.target_network.predict_on_batch(next_states)
        q_online = self.model.online_network.predict_on_batch(next_states)

        best_actions = tf.cast(tf.argmax(q_online, axis=1), tf.int32)

        stack = tf.stack((self.idx, best_actions), axis=1)
        target_q_values = tf.gather_nd(q_target, stack)

        targets = rewards + not_done * self.gamma * target_q_values
        q_online[(self.idx, actions)] = targets

        self.model.online_network.fit(
            x=states,
            y=q_online,
            epochs=10,
            verbose=0
        )

    def train(self, load_model=False):
        if load_model:
            self._load_model('best')

        data_samples = len(self.data) - 1
        for episode in range(1, self.episodes + 1):
            print("Episode: {}/{}".format(episode, self.episodes))
            self.reward.reset_balance()
            state = self.env.state_create(
                data=self.data[self.X_var].values,
                window_size=self.window_size,
                iterator=0)
            for t in tqdm(range(data_samples)):
                action = self.epsilon_greedy_policy(state)
                # print(action)
                next_state = self.env.state_create(
                    data=self.data[self.X_var].values,
                    iterator=t+1,
                    window_size=self.window_size)

                status_position = self.reward.check_status_position()
                reward = self.reward._managment_take(
                    action=action,
                    status_position=status_position,
                    data=self.data.iloc[t, :]
                )
                # print(action, reward)
                if t == data_samples - 1:
                    done = True
                    result_model = {
                        'epoch': episode,
                        'total_profit': self.reward.get_result()
                    }
                    print(result_model)
                    # Save best model
                    list_result = list(map(lambda i: i['total_profit'],
                                           self.reward.get_epochs_profit()))
                    best_result = 0
                    if len(list_result):
                        best_result = max(list_result)
                    self.reward.add_epochs_profit(result_model)
                    if self.reward.get_result() > best_result:
                        # Save best model
                        self.model.save_model(
                            memory=self.memory.get_memory(),
                            name='best'
                        )
                        # Save best results
                        pd.DataFrame(self.reward.balance).to_csv(
                            f'results/{self.name}.csv')

                        print('Best result: {}'.format(
                            self.reward.get_result()))
                        print(self.name)
                        # self.reward.plot_profit()
                else:
                    done = False
                self.memory.add_memory(
                    (state, action, reward, next_state, done))
                state = next_state
                # Update epsilon
                self._update_epsilon()
                # Update agents' experience
                self.experience_replay()
            if episode % 1 == 0:
                self.reward.plot_profit(
                    name=self.name, show=False, iter=episode)
