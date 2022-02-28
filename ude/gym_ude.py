#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################
"""A class to wrap UDE Environment to interface it as OpenAI gym environment"""
import gym
import numpy as np
from typing import Union, Tuple, Dict, List, Any, Optional
from ude.environment.ude_environment import UDEEnvironment
from ude.environment.constants import UDEResetMode

from gym.spaces.space import Space

GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class UDEToGymWrapper(gym.Env):
    """
    UDEToGymWrapper class
    - Custom Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 ude_env: UDEEnvironment,
                 agent_name: Optional[str] = None):
        """
        Initialize UDEToGymWrapper class

        Args:
            ude_env (UDEEnvironment): UDE environment instance
        """
        self._ude_env = ude_env
        observation_space = self._ude_env.observation_space or {}
        self._agent_name = list(observation_space.keys())[0] if observation_space else agent_name
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        # Turn off auto-reset in UDE Environment as auto-reset
        # will make self.game_over status to not synchronized to
        # actual environment game_over status.
        self._ude_env.reset_mode = UDEResetMode.MANUAL

    def step(self, action: Union[List[Any], Any]) -> GymStepResult:
        """
        Performs one single-agent step in the environment with given action,
        and return GymStepResult, which contains
        observation, reward, done, and info

        Args:
            action (Union[List[Any], Any]): action for the agent

        Returns:
            GymStepResult: observation, reward, done, and info
        """
        # Execute one time step within the environment
        action_dict = {self._agent_name: action}
        obs, rewards, dones, _, info = self._ude_env.step(action_dict=action_dict)
        done = list(dones.values())[0]
        if done:
            self.game_over = True
        # Return first value of obs, rewards, dones
        return (list(obs.values())[0],
                list(rewards.values())[0],
                list(dones.values())[0],
                info)

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """
        Reset the Environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            Union[List[np.ndarray], np.ndarray]: first observation in new episode.
        """
        self.game_over = False
        # Reset the state of the environment to an initial state
        obs, info = self._ude_env.reset()
        # Return first value of obs
        return list(obs.values())[0]

    def render(self, mode: str = 'human', close: bool = False) -> None:
        """
        Render the environment.
        * Note: Currently not implemented.

        Args:
            mode (str): not used.
            close (bool):  not used.
        """
        # Render the environment to the screen
        # logger.warning("Could not render environment")
        return

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        self._ude_env.close()

    def seed(self, seed: Any = None) -> None:
        """
        Sets the seed for this env's random number generator(s).
        * Note: Currently not implemented.

        Args:
            seed (Any): not used
        """
        # logger.warning("Could not seed environment")
        return

    @property
    def action_space(self) -> Space:
        """
        Return the action space.

        Returns:
            Any: action space
        """
        action_space = self._ude_env.action_space or {}
        return list(action_space.values())[0] if action_space else Space()

    @property
    def observation_space(self) -> Any:
        """
        Return the observation space.

        Returns:
            Any: observation space
        """
        observation_space = self._ude_env.observation_space or {}
        return list(observation_space.values())[0] if observation_space else Space()
