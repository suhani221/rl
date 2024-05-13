import argparse
import os
import pickle
import environments
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from policies import EquitableLPPolicyMMR
from policies import NoActionPolicy
from policies import RandomPolicy
from policies import WhittleIndexPolicy
from policies import RoundRobinPolicy
from policies import MyopicPolicy
import util

parser = argparse.ArgumentParser(
    description='Plot data from experiments. Must first run combine_results.py'
)
parser.add_argument('--expname', type=str, help='Name of experiment')
parser.add_argument(
    '--model', type=str, help='Name of model data file for experiments'
)
parser.add_argument(
    '--params', type=str, help='Name of parameter file for experiments'
)
parser.add_argument(
    '--image_type',
    type=str,
    default='png',
    choices=['png', 'pdf', 'jpg'],
    help='Plot image file type',
)
parser.add_argument(
    '--csv',
    action='store_true',
    help='If true, will write csv summaries of data in plots',
)
args = parser.parse_args()
param_dict = util.parse_parameters(args.params)
IMAGE_TYPE = args.image_type
csv = args.csv
horizon = param_dict['horizon_list'][0]
REWARDS_AGGREGATION_METHOD = ('From Start', 'Total', horizon)

policy_name_conversion = {
    NoActionPolicy.policy_key(): 'No Act',
    RandomPolicy.policy_key(): 'Rand',
    RoundRobinPolicy.policy_key(): 'RR',
    WhittleIndexPolicy.policy_key(): 'Opt',
    MyopicPolicy.policy_key(): 'Myopic',
    EquitableLPPolicyMMR.policy_key(): 'MMR',
   
}

policy_colors = {
    NoActionPolicy.policy_key(): '#000000',
    RandomPolicy.policy_key(): 'y',
    WhittleIndexPolicy.policy_key(): '#7CAE00',
    EquitableLPPolicyMMR.policy_key(): 'b',
    RoundRobinPolicy.policy_key(): 'r',
    MyopicPolicy.policy_key(): 'g',
}

def reward_accounting(
    reward_list, state_list, accounting_method, ind_to_tup_dict
):
  """Implement reward accounting methods.

  Args:
    reward_list: array of rewards of dimension n_trials x n_arms x horizon
    state_list: array of states of dimension n_trials x n_arms x horizon
    accounting_method: a 3-tuple, as follows:
      0: {'From Start', 'Timepoints'}
        'From Start' counts from the beginning of the simulation
        'Timepoints' counts from the point each arm became active in simulation
      1: {'Final', 'Total'}
        'Final' reports the final state/reward of the accoutning period
        'Total' reports the sum of rewards over the accounting period
      2: int that defines the length of the accounting period
    ind_to_tup_dict: dictionary mapping state indexes to tuples

  Returns:
    rewards aggregated according to accounting_method
  """
  rewards_accounted = None
  if accounting_method[:2] == ('From Start', 'Final'):
    rewards_accounted = reward_list[:, :, -1]  # final step rewards
  elif accounting_method[:2] == ('From Start', 'Total'):
    rewards_accounted = reward_list.sum(axis=-1)  # sum over horizon rewards
  else:
    print('Accounting method not recognized')
    exit()

  return rewards_accounted


def group_reward_difference_bar_chart_general(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    budget_fracs,
    img_dir,
    expname,
    policy_name_list=None,
    environment_class=None,
):
    small_size = 12
    medium_size = 18
    bigger_size = 22

    plt.rc('font', size=medium_size)
    plt.rc('axes', titlesize=medium_size)
    plt.rc('axes', labelsize=medium_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=medium_size)
    plt.rc('legend', fontsize=small_size)
    plt.rc('figure', titlesize=bigger_size)

    n_policies = len(policy_name_list)
    policy_name_list_pretty = [policy_name_conversion[pname] for pname in policy_name_list]

    _, group_mappings, _ = environment_class.parse_model_data(args.model, n_arms)
    group_list = sorted(list(set(group_mappings)))
    n_groups = len(group_list)

    # Calculate total number of rows needed for subplots
    total_rows = n_policies * len(budget_fracs)
    fig, axs = plt.subplots(total_rows, 1, figsize=(8, 4 * total_rows))

    row = 0
    for i, policy_name in enumerate(policy_name_list):
        for b, budget_frac in enumerate(budget_fracs):
            budget = int(n_arms * budget_frac)
            file_name = input_file_name_template % (
                expname,
                n_arms,
                budget_frac,
                environment_name,
                n_trials,
                seed,
                horizon,
            )

            with open(file_name, 'rb') as fh:
                problem_pickle = pickle.load(fh)

            data = problem_pickle['data']
            reward_list = data[policy_name]['rewards']
            state_list = data[policy_name]['states']

            reward_list_accounted = reward_accounting(
                reward_list,
                state_list,
                REWARDS_AGGREGATION_METHOD,
                None,
            )

            rewards_means = np.zeros(n_groups + 1)
            rewards_errors = np.zeros(n_groups + 1)

            for group in group_list:
                group_inds = group_mappings == group
                group_size = group_inds.sum()
                rewards_group = (reward_list_accounted[:, group_inds].sum(axis=-1) / group_size)
                rewards_means[group] = rewards_group.mean()
                rewards_errors[group] = rewards_group.std() / np.sqrt(n_trials)

            arm_sum = reward_list_accounted.sum(axis=-1)
            rewards_means[n_groups] = arm_sum.mean() / n_arms
            rewards_errors[n_groups] = arm_sum.std() / n_arms / np.sqrt(n_trials)

            axs[row].bar(
                np.arange(n_groups + 1),
                rewards_means,
                yerr=rewards_errors,
                align='center',
                alpha=0.8,
                ecolor='black',
                capsize=2,
                color=policy_colors[policy_name],
                label=f'{policy_name_list_pretty[i]} (Budget {budget_frac})'
            )

            axs[row].set_xticks(np.arange(n_groups + 1))
            axs[row].set_xticklabels(list(map(str, group_list)) + ['Total'])
            axs[row].set_ylabel('Mean %s Reward' % REWARDS_AGGREGATION_METHOD[1])
            axs[row].legend(loc='upper right')
            axs[row].grid(True)
            

            row += 1

    # # Additional plots for total rewards for each budget
    # fig_totals, axs_totals = plt.subplots(1, len(budget_fracs), figsize=(12, 4))
    # for b, budget_frac in enumerate(budget_fracs):
    #     total_rewards = [reward_list_accounted.sum(axis=-1).mean() / n_arms for reward_list_accounted in [data[policy_name_list[i]]['rewards'] for i in range(n_policies)]]
    #     total_errors = [np.std(reward_list_accounted.sum(axis=-1)) / np.sqrt(n_trials) / n_arms for reward_list_accounted in [data[policy_name_list[i]]['rewards'] for i in range(n_policies)]]

    #     axs_totals[b].bar(
    #         np.arange(n_policies),
    #         total_rewards,
    #         yerr=total_errors,
    #         align='center',
    #         alpha=0.8,
    #         ecolor='black',
    #         capsize=2,
    #         color=policy_colors[policy_name],
    #         label=f'{policy_name_list_pretty[i]} (Budget {budget_frac})'
    #     )

    #     bars = axs_totals[b].bar(
    #         np.arange(n_policies),
    #         total_rewards,
    #         yerr=total_errors,
    #         align='center',
    #         alpha=0.8,
    #         ecolor='black',
    #         capsize=2,
    #         color=[policy_colors[policy_name_list[i]] for i in range(n_policies)]
    #     )
    #     for i, bar in enumerate(bars):
    #         bar.set_label(policy_name_list_pretty[i])

    #     axs_totals[b].set_xticks(np.arange(n_policies))
    #     axs_totals[b].set_xticklabels([policy_name_list_pretty[i] for i in range(n_policies)], rotation=45)
    #     axs_totals[b].set_title(f'Total Rewards for Budget {budget_frac:.2f}')
    #     axs_totals[b].set_ylabel('Mean Reward')
    #     axs_totals[b].legend(loc='upper right')
    #     axs_totals[b].grid(True)

    #     axs_totals[b].legend(loc='upper left', bbox_to_anchor=(1, 1))
    #     axs_totals[b].grid(True)
      

    # plt.tight_layout()
    # save_str_totals = '%s/total_rewards_per_budget_%s.%s'
    # plt.savefig(
    #     save_str_totals % (
    #         img_dir,
    #         REWARDS_AGGREGATION_METHOD,
    #         IMAGE_TYPE,
    #     ),
    #     dpi=300,
    # )
    # plt.close(fig_totals)


def health_difference_bar_chart_general(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    budget_fracs,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Health difference bar chart."""

  small_size = 10
  medium_size = 14
  bigger_size = 18

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  rewards_means = np.zeros((n_policies))
  rewards_errors = np.zeros((n_policies))

  for budget_frac in budget_fracs:
    budget = int(n_arms * budget_frac)
    for i, policy_name in enumerate(policy_name_list):
      file_name = input_file_name_template % (
          expname,
          n_arms,
          budget_frac,
          environment_name,
          n_trials,
          seed,
          horizon,
      )

      problem_pickle = {}
      with open(file_name, 'rb') as fh:
        problem_pickle = pickle.load(fh)

      # environment_parameters = problem_pickle['simulation_parameters'][
      #     'environment_parameters'
      # ]

      env_info = problem_pickle['simulation_parameters']['environment_info']

      data = problem_pickle['data']

      reward_list = data[policy_name][
          'rewards'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
      state_list = data[policy_name][
          'states'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)

      reward_list_accounted = reward_accounting(
          reward_list,
          state_list,
          REWARDS_AGGREGATION_METHOD,
          None,
      )

      reward_arm_sum = reward_list_accounted.sum(axis=-1)  # sum over arms
      rewards_means[i] = reward_arm_sum.mean()
      rewards_errors[i] = reward_arm_sum.std() / np.sqrt(n_trials)

      plt.bar(
          i,
          rewards_means[i],
          yerr=rewards_errors[i],
          align='center',
          alpha=0.8,
          ecolor='black',
          capsize=10,
          color=policy_colors[policy_name],
          label=policy_name_list_pretty[i],
      )

    plt.xticks(np.arange(n_policies), policy_name_list_pretty, rotation=35)
    plt.ylabel('%s Reward' % (REWARDS_AGGREGATION_METHOD[1]))
    # plt.xlabel('Policy')
    time_delta = horizon
    if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
      time_delta = REWARDS_AGGREGATION_METHOD[2]

    plt.title(
        r'Reward (%s months)' % (time_delta,)
        + '\n'
        + '(%s Arms, %s Budget)' % (n_arms, budget)
    )
    plt.ylim([
        rewards_means.min() - 0.05 * n_arms,
        rewards_means.max() + 0.05 * n_arms,
    ])

    sub_dir_name = 'key_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.3)

    # plt.tight_layout()
    save_str = '%s/health_bar_chart_seed-%s_ntrials-'
    save_str += '%s_narms%s_b%.2f_all_%s.%s'
    plt.savefig(
        save_str
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        ),
        dpi=200,
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/health_bar_chart_seed-%s_ntrials-%s_narms%s_b%.2f_'
      fname += '%s.csv'
      fname_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'means',
      )
      fname_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'errors',
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      columns = ['Value']
      df_means = pd.DataFrame(
          rewards_means, index=pretty_policy_name_list, columns=columns
      )
      df_errors = pd.DataFrame(
          rewards_errors, index=pretty_policy_name_list, columns=columns
      )

      df_means.to_csv(fname_means)
      df_errors.to_csv(fname_errors)

if __name__ == '__main__':
    environment_name_in = param_dict['environment_name']
    environment_class_in = environments.Config.get_env_class(environment_name_in)
    env_more_params_in = environment_class_in.environment_params().keys()

    combined_filename_template_in = util.Config.get_combined_filename_template(
        environment_class_in, args.expname
    )
    combined_filename_template_in += '.pkl'

    batch_dir_path_in = util.Config.get_img_output_path(args.expname)

    n_arms_list_in = param_dict['n_arms_list']
    budget_frac_list_in = param_dict['budget_frac_list']
    n_trials_in = param_dict['n_trials']
    horizon_list_in = param_dict['horizon_list']
    base_seed_in = param_dict['base_seed']
    n_arms_in = n_arms_list_in[0]
    horizon_in = horizon_list_in[0]

    group_reward_difference_bar_chart_general(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        budget_frac_list_in,
        batch_dir_path_in,
        policy_name_list=param_dict['policy_list'],
        expname=args.expname,
        environment_class=environment_class_in,
    )

    health_difference_bar_chart_general(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )