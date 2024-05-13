import abc 
import collections
import functools
import itertools
import math
import numpy as np
import pandas as pd
import util
class RMABEnvironment(abc.ABC):
  """Abstract base class for RMAB environments."""

  @abc.abstractmethod
  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               ) -> None:
    """Initialize the abstract RMABEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to instantiate a model
        e.g., probabilities, rewards, and group maps
      stream_map: defines when each arm enters the simulation
    """
    self.n_arms = n_arms
    self.start_seed = start_seed
    self.horizon = horizon
    self.model_data_file = model_data_file
    self.stream_map = stream_map

    # create random stream and set seed
    self.random_stream = np.random.RandomState()
    self.set_seed(self.start_seed)

    self.current_states = None
    self.transition_probs = None
    self.rewards = None

    self.group_map = None

  def set_seed(self, seed: int) -> None:
    """Set the random seed."""
    self.random_stream.seed(seed)

  def step(self, actions: np.ndarray) -> np.ndarray:
    """Evolve all arm states by a single step.

    Args:
      actions: n_arms-length array of actions, encoded as integers

    Returns:
      next_states: n_arms-length array of arm states after being evolved
      rewards: n_arms-length array of rewards

    """
    next_states = np.zeros(self.n_arms, dtype=int)
    for n in range(self.n_arms):
      state_ind = self.current_states[n]
      next_state_one_hot = self.random_stream.multinomial(
          n=1,
          pvals=self.transition_probs[n, state_ind, actions[n]],
      )
      next_state_ind = np.argmax(next_state_one_hot)
      next_states[n] = next_state_ind

    rewards = self.get_rewards_for_states(self.current_states)
    self.current_states = np.copy(next_states)

    return next_states, rewards

  def get_rewards_for_states(self, current_states: np.ndarray) -> np.ndarray:
    """Return rewards for the current arm states.

    Args:
      current_states: n_arms-length array of integer arm states

    Returns:
      rewards: reward value for each arm, based on its state

    """
    return np.array(
        [self.rewards[n, current_states[n]] for n in range(self.n_arms)])

  @staticmethod
  def parse_model_data(filename, n_arms):
    """Default parser for RMABEnvironments, only expecting group definitions."""
    df = pd.read_csv(filename)
    expected_cols = ['Group', 'frac']

    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_mappings = np.zeros(n_arms, dtype=np.int32)
    prev_mapping_ind = 0
    for row in range(df.shape[0]):

      group = df.iloc[row]['Group']

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

    return group_mappings

  def get_reward_definition(self) -> np.ndarray:
    """Return a copy of the reward definition."""
    return np.copy(self.rewards)

  def get_states(self) -> np.ndarray:
    """Return the current arm states."""
    return np.copy(self.current_states)

  def get_info(self) -> dict[str, object]:
    """Return other relevant environment information."""
    return {
        'group_map': self.group_map,
        'active_arms_helper': self.create_active_arms_helper(),
    }

  def create_active_arms_helper(self) -> util.GetActiveArms:
    self.active_arms_helper = util.GetActiveArmsDefault()
    return self.active_arms_helper

  def get_active_arms(self, states):
    """Method to get active arms of this environment."""
    return self.active_arms_helper.get_active_arms(states)

  @abc.abstractmethod
  def reset_states(self) -> np.ndarray:
    """Set all arm states to some initial value, and return the states."""
    pass

  @abc.abstractmethod
  def get_transition_probabilities(self) -> None:
    """Set the arm transition probabilities."""
    pass

  @abc.abstractmethod
  def set_reward_definition(self) -> None:
    """Set all arm reward definitions."""
    pass

  @staticmethod
  @abc.abstractmethod
  def env_key() -> str:
    """Return the name of the environment."""
    raise NotImplementedError

  @staticmethod
  @abc.abstractmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    raise NotImplementedError
class MaternalHealthEnvironment(RMABEnvironment):
  """Class for generating 3-state maternal health environment."""

  env_key_name = 'Maternal Health Environment'

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               ) -> None:
    """Initialize the MaternalHealthEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_map
      stream_map: unused by this class
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    ret_data = MaternalHealthEnvironment.parse_model_data(
        model_data_file, 
        n_arms
    )
    self.group_probs = ret_data[0]
    self.group_map = ret_data[1]
    self.group_rewards = ret_data[2]

    self.n_states = 3
    self.n_actions = 2

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    self.current_states = np.zeros(n_arms, dtype=np.int32)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((n_arms, self.n_states), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return MaternalHealthEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    return params


  @staticmethod
  def parse_model_data(filename, n_arms):
    """Parse maternal health model data."""
    df = pd.read_csv(filename)
    prob_cols = [
        'p000',
        'p010',
        'p102',
        'p110',
        'p202',
        'p212',
    ]
    reward_cols = [
        'r0',
        'r1',
        'r2',
    ]
    expected_cols = ['Group', 'frac'] + reward_cols + prob_cols

    # print(df['frac'].sum())
    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_probs = [{} for group in range(df.shape[0])]
    group_rewards = [{} for group in range(df.shape[0])]
    group_mappings = np.zeros(n_arms, dtype=np.int32)
    group_definitions = []
    prev_mapping_ind = 0
    for row in range(df.shape[0]):
      group = int(df.iloc[row]['Group'])

      # assign probability parameters
      for param in prob_cols:
        group_probs[group][param] = df.iloc[row][param]

      # assign reward parameters
      for param in reward_cols:
        group_rewards[group][param] = df.iloc[row][param]

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

    return group_probs, group_mappings, group_rewards

  def sample_prob(self, prob):
    """Sample transition probabilities with some noise."""
    tiny_eps = 1e-3
    std = 0.2

    # group = self.group_map[arm]
    # prob = self.group_probs[group][prob_name]

    # want to scale deviations based on their distance from 0 or 1
    scale = min(abs(prob), abs(1-prob))

    # introduce small variation between arms in each group
    sigma = std*scale
    sampled_prob = min(1-tiny_eps, self.random_stream.normal(prob, sigma))
    sampled_prob = max(tiny_eps, sampled_prob)
    # self.arm_probs[arm][prob_name] = sampled_prob

    return sampled_prob


  def get_transition_probabilities(self) -> None:
    """Load parameters from dict."""

    for n in range(self.n_arms):

      group = self.group_map[n]
      
      tup_list = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 2),
        (1, 1, 0),
        (2, 0, 2),
        (2, 1, 2),
      ]
      for tup in tup_list:
        s, a, sp = tup
        prob_str = 'p%i%i%i'%(s, a, sp)
        prob_mean = self.group_probs[group][prob_str]
        prob = self.sample_prob(prob_mean)
        self.transition_probs[n, s, a, sp] = prob
        self.transition_probs[n, s, a, 1] = 1 - prob


  def reset_states(self) -> np.ndarray:
    """Reset the state of all arms in a uniform random manner."""
    self.current_states = self.random_stream.choice(
        a=np.arange(self.n_states), size=self.n_arms, replace=True)
    return self.current_states

  def set_reward_definition(self) -> None:
    """Set the reward definition."""
    for n in range(self.n_arms):
      group = self.group_map[n]
      for s in range(self.n_states):
        reward_str = 'r%i' % s
        self.rewards[n, s] = self.group_rewards[group][reward_str]

class Config():
  """Class for getting references to and instantiating RMABEnvironments."""

  # Define policy class mapping
  # If defining a new class, add its name to class_list
  class_list = [
     
      MaternalHealthEnvironment,
      
  ]
  env_class_map = {cl.env_key(): cl for cl in class_list}

  @staticmethod
  def get_env_class(env_name):
    """Return a class reference corresponding the input environment string."""
    try:
      env_class = Config.env_class_map[env_name]
    except KeyError:
      err_string = 'Did not recognize environment "%s". Please add it to '
      err_string += 'environments.Config.class_list.'
      print(err_string % env_name)
      exit()
    return env_class


