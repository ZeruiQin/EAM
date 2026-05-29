# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a, UI_TARS_M3A, RAC_Agent, EAM_Agent
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface
import config

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  env_adb = os.environ.get('ADB_PATH')
  if env_adb and os.path.isfile(env_adb):
    return env_adb
  android_home = os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT')
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  if android_home:
    potential_paths.append(os.path.join(android_home, 'platform-tools', 'adb'))
    potential_paths.append(os.path.join(android_home, 'platform-tools', 'adb.exe'))
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    True,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')
_EAM_GRAPH_DIR = flags.DEFINE_string(
    'eam_graph_dir',
    'artifacts/graph_env',
    'Directory containing exported EAM app graph folders.',
)
_EAM_MODEL_PATH = flags.DEFINE_string(
    'eam_model_path',
    '',
    'Trained EAM reward-model checkpoint used for MCTS path retrieval.',
)
_EAM_APP_NAME = flags.DEFINE_string(
    'eam_app_name',
    '',
    'Exported app graph name, e.g. contacts or simple-calendar-pro.',
)
_EAM_TOP_K = flags.DEFINE_integer('eam_top_k', 5, 'Number of MCTS paths to send to the planner.')
_EAM_MAX_DEPTH = flags.DEFINE_integer('eam_max_depth', 60, 'Maximum MCTS path depth.')
_EAM_MAX_ITERATIONS = flags.DEFINE_integer('eam_max_iterations', 50, 'MCTS search iterations.')
_EAM_EXPLORATION_CONSTANT = flags.DEFINE_float(
    'eam_exploration_constant',
    3.0,
    'MCTS UCT exploration constant.',
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    True,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('openai/gpt-4o'))
  elif _AGENT_NAME.value == 'm3a_qwen':
    agent = m3a.M3A(env, infer.QwenWrapper('qwen2.5-vl-3b-instruct'))
  elif _AGENT_NAME.value == 't3a_qwen':
    agent = t3a.T3A(env, infer.QwenWrapper('qwen2.5-vl-7b'))
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)
  elif _AGENT_NAME.value == 'm3a_uitars':
    agent = UI_TARS_M3A.SRoA(env, infer.UiTarsWrapper('ui-tars'))
  elif _AGENT_NAME.value == 'm3a_uitars_7b':
    agent = UI_TARS_M3A.SRoA(env, infer.Gpt4Wrapper('bytedance/ui-tars-1.5-7b'))
  elif _AGENT_NAME.value == 'RAC':
    # agent = RAC_Agent.RAC(env, cloud_llm=infer.Gpt4Wrapper('anthropic/claude-sonnet-4'),
    #                     local_llm=infer.QwenWrapper('qwen2.5-vl-3b-instruct'), uri=config.Neo4j_URI,
    #                     auth=config.Neo4j_AUTH)
    agent = RAC_Agent.RAC(env, cloud_llm=infer.Gpt4Wrapper('anthropic/claude-sonnet-4'),
                          local_llm=infer.Gpt4Wrapper('anthropic/claude-sonnet-4'), uri=config.Neo4j_URI,
                          auth=config.Neo4j_AUTH)
  elif _AGENT_NAME.value == 'eam_mcts':
    if not _EAM_MODEL_PATH.value:
      raise ValueError('--eam_model_path is required for --agent_name=eam_mcts')
    agent = EAM_Agent.EAMAgent(
        env,
        cloud_llm=infer.Gpt4Wrapper(config.LLM_MODEL),
        local_llm=infer.Gpt4Wrapper(config.LLM_MODEL),
        uri=config.Neo4j_URI,
        auth=config.Neo4j_AUTH,
        graph_dir=_EAM_GRAPH_DIR.value,
        model_path=_EAM_MODEL_PATH.value,
        app_name=_EAM_APP_NAME.value or None,
        top_k=_EAM_TOP_K.value,
        max_depth=_EAM_MAX_DEPTH.value,
        max_iterations=_EAM_MAX_ITERATIONS.value,
        exploration_constant=_EAM_EXPLORATION_CONSTANT.value,
    )

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  # suite = suite_utils.create_suite(
  #     task_registry.get_registry(family=_SUITE_FAMILY.value),
  #     n_task_combinations=n_task_combinations,
  #     seed=2,
  #     tasks=_TASKS.value,
  #     use_identical_params=_FIXED_TASK_SEED.value,
  # )
  # for name, instances in suite.items():
  #   print(name)
  #   for i, instance in enumerate(instances):
  #       print(instance.goal)
  suite.suite_family = _SUITE_FAMILY.value

  agent = _get_agent(env, _SUITE_FAMILY.value)

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
