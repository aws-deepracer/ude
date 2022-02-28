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
"""A class for UDE Environment."""
from typing import Union, Any, Dict
from threading import RLock

from gym import Space

from ude.ude_typing import MultiAgentDict, UDEStepResult, UDEResetResult, AgentID
from ude.side_channels.ude_side_channel import AbstractSideChannel
from ude.environment.interfaces import (
    UDEEnvironmentInterface,
    UDEEnvironmentAdapterInterface
)
from ude.environment.adapters.remote_environment_adapter import RemoteEnvironmentAdapter
from ude.environment.constants import UDEResetMode


class UDEEnvironment(UDEEnvironmentInterface):
    """
    UDE Environment to expose the environment interface.
    """
    def __init__(self, ude_env_adapter: UDEEnvironmentAdapterInterface,
                 reset_mode: Union[str, UDEResetMode] = UDEResetMode.MANUAL,
                 game_over_cond: Union[all, any] = any):
        """
        Initialized UDE Environment.

        Args:
            ude_env_adapter (UDEEnvironmentAdapterInterface): Actual environment wrapped as an adapter.
            reset_mode (Union[str, UDEResetMode]): Reset mode (MANUAL| AUTO) whether to reset automatically or not on game over.
            game_over_cond (Union[all, any]): Game over condition.
              - all: all agents need to be in done status to be done status.
              - any: if one of agents is in done status then it is in done status.
        """
        super().__init__()
        self._ude_env_adapter = ude_env_adapter
        # Automatically reset the environment when game-over condition is met.
        # This configuration only impact local UDEEnvironment, and
        # in remote UDEEnvironment case, this will be ignored.
        # This will be useful when ownership of reset call is not
        # clearly defined (ex. multi-agents training/evaluation).
        self._reset_mode = UDEResetMode(reset_mode)
        self._game_over_cond = game_over_cond
        # Guarantee critical section for interfaces (step/reset/close)
        self._lock = RLock()

    @property
    def reset_mode(self) -> UDEResetMode:
        """
        Return the flag whether environment will automatically reset on game over.
        Returns:
            UDEResetMode: The enum (MANUAL|AUTO) whether to reset automatically or not on game over.
        """
        return self._reset_mode

    @reset_mode.setter
    def reset_mode(self, value: Union[str, UDEResetMode]):
        """
        Setter for the flag whether environment will automatically reset on game over.
        Args:
            value (Union[str, UDEResetMode]): Reset mode (MANUAL| AUTO)
        """
        self._reset_mode = UDEResetMode(value)

    @property
    def game_over_cond(self) -> Any:
        """
        Return the game over condition method

        Returns:
            Any: the game over condition method
        """
        return self._game_over_cond

    @game_over_cond.setter
    def game_over_cond(self, value: Any) -> None:
        """
        Set the game over condition method.

        Args:
            value (Any): the game over condition method.
        """
        self._game_over_cond = value

    @property
    def env(self) -> UDEEnvironmentAdapterInterface:
        """
        Return the actual environment instance.

        Returns:
            UDEEnvironmentAdapterInterface: the actual environment instance.
        """
        return self._ude_env_adapter

    @property
    def side_channel(self) -> AbstractSideChannel:
        """
        Returns side channel to send and receive data to/from actual environment

        Returns:
            AbstractSideChannel: the instance of side channel.
        """
        return self.env.side_channel

    @property
    def is_remote(self) -> bool:
        """
        The flag whether the environment is remote.

        Returns:
            bool: True if remote, False otherwise.
        """

        return isinstance(self.env, RemoteEnvironmentAdapter)

    @property
    def is_local(self) -> bool:
        """
        The flag whether the environment is local.

        Returns:
            bool: True if local, False otherwise.
        """
        return not self.is_remote

    def step(self, action_dict: MultiAgentDict) -> UDEStepResult:
        """
        Performs one multi-agent step with given action, and retrieve
        observation(s), reward(s), done(s), action(s) taken,
        and info (if there is any).

        Args:
            action_dict (MultiAgentDict): the action(s) for the agent(s) with agent_name as key.

        Returns:
            UDEStepResult: observations, rewards, dones, last_actions, info
        """
        with self._lock:
            obs, reward, done, last_action, info = self.env.step(action_dict=action_dict)
            if self.is_local:
                if self.reset_mode == UDEResetMode.AUTO and self._game_over_cond(done.values()):
                    obs, info = self.reset()
            return obs, reward, done, last_action, info

    def reset(self) -> UDEResetResult:
        """
        Reset the environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            UDEResetResult: first observation and info in new episode.
        """
        with self._lock:
            return self.env.reset()

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        with self._lock:
            self.env.close()

    @property
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        with self._lock:
            return self.env.observation_space

    @property
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        with self._lock:
            return self.env.action_space
