import collections
import numpy as np
import pandas as pd


class MaternalHealthEnvironment:
    """Class for generating 3-state maternal health environment."""

    def __init__(self, n_arms: int, start_seed: int, horizon: int, model_data_file: str, stream_map: np.ndarray) -> None:
        """Initialize the MaternalHealthEnvironment class.

        Args:
            n_arms: number of arms in the RMAB
            start_seed: starting random seed for experiments
            horizon: length of simulation
            model_data_file: file with data to populate group_map
            stream_map: unused by this class
        """
        self.n_arms = n_arms
        self.start_seed = start_seed
        self.horizon = horizon
        self.model_data_file = model_data_file
        self.stream_map = stream_map

        # Create random stream and set seed
        self.random_stream = np.random.RandomState()
        self.set_seed(self.start_seed)

        # Parse model data
        ret_data = self.parse_model_data(model_data_file, n_arms)
        self.group_probs = ret_data[0]
        self.group_map = ret_data[1]
        self.group_rewards = ret_data[2]

        self.n_states = 3
        self.n_actions = 2

        # Initialize transition probabilities
        self.transition_probs = np.zeros((self.n_arms, self.n_states, self.n_actions, self.n_states))
        self.get_transition_probabilities()

        # Initialize state
        self.current_states = np.zeros(n_arms, dtype=np.int32)
        self.reset_states()

        # Define rewards
        self.rewards = np.zeros((n_arms, self.n_states), dtype=float)
        self.set_reward_definition()

    def set_seed(self, seed: int) -> None:
        """Set the random seed."""
        self.random_stream.seed(seed)

    @staticmethod
    def parse_model_data(filename, n_arms):
        """Parse maternal health model data."""
        df = pd.read_csv(filename)
        prob_cols = ['p000', 'p010', 'p102', 'p110', 'p202', 'p212']
        reward_cols = ['r0', 'r1', 'r2']
        expected_cols = ['Group', 'frac'] + reward_cols + prob_cols

        assert not set(expected_cols) - set(df.columns.values)
        assert abs(df['frac'].sum() - 1) < 1e-4

        group_probs = [{} for group in range(df.shape[0])]
        group_rewards = [{} for group in range(df.shape[0])]
        group_mappings = np.zeros(n_arms, dtype=np.int32)
        group_definitions = []
        prev_mapping_ind = 0
        for row in range(df.shape[0]):
            group = int(df.iloc[row]['Group'])

            for param in prob_cols:
                group_probs[group][param] = df.iloc[row][param]

            for param in reward_cols:
                group_rewards[group][param] = df.iloc[row][param]

            frac = df.iloc[row]['frac']
            next_mapping_ind = int(n_arms * frac)
            group_mappings[prev_mapping_ind:prev_mapping_ind + next_mapping_ind] = group
            prev_mapping_ind += next_mapping_ind

        return group_probs, group_mappings, group_rewards

    def sample_prob(self, prob):
        """Sample transition probabilities with some noise."""
        tiny_eps = 1e-3
        std = 0.2
        scale = min(abs(prob), abs(1 - prob))
        sigma = std * scale
        sampled_prob = min(1 - tiny_eps, self.random_stream.normal(prob, sigma))
        sampled_prob = max(tiny_eps, sampled_prob)
        return sampled_prob

    def get_transition_probabilities(self) -> None:
        """Load parameters from dict."""
        for n in range(self.n_arms):
            group = self.group_map[n]
            tup_list = [(0, 0, 0), (0, 1, 0), (1, 0, 2), (1, 1, 0), (2, 0, 2), (2, 1, 2)]
            for tup in tup_list:
                s, a, sp = tup
                prob_str = 'p%i%i%i' % (s, a, sp)
                prob_mean = self.group_probs[group][prob_str]
                prob = self.sample_prob(prob_mean)
                self.transition_probs[n, s, a, sp] = prob
                self.transition_probs[n, s, a, 1] = 1 - prob

    def reset_states(self) -> np.ndarray:
        """Reset the state of all arms in a uniform random manner."""
        self.current_states = self.random_stream.choice(a=np.arange(self.n_states), size=self.n_arms, replace=True)
        return self.current_states

    def set_reward_definition(self) -> None:
        """Set the reward definition."""
        for n in range(self.n_arms):
            group = self.group_map[n]
            for s in range(self.n_states):
                reward_str = 'r%i' % s
                self.rewards[n, s] = self.group_rewards[group][reward_str]

    @staticmethod
    def env_key() -> str:
        """Return the name of the environment."""
        return 'Maternal Health Environment'

    @staticmethod
    def environment_params() -> collections.OrderedDict[str, str]:
        """Return an OrderedDict of environment hyperparameters and their types."""
        params = collections.OrderedDict()
        return params