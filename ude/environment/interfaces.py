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
"""Classes for UDE environment interfaces."""
import abc
from typing import Dict

from ude.ude_typing import MultiAgentDict, UDEStepResult, UDEResetResult, AgentID
from ude.side_channels.ude_side_channel import AbstractSideChannel

from gym.spaces.space import Space

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta('ABC', (object,), {})


class UDEEnvironmentInterface(ABC):
    """
    UDE Environment Interface
    """
    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> UDEResetResult:
        """
        Reset the environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            UDEResetResult: first observation and info in new episode.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        raise NotImplementedError()

    """
    SideChannel Related Functions
    """
    @property
    @abc.abstractmethod
    def side_channel(self) -> AbstractSideChannel:
        """
        Returns side channel to send to and receive data from environment

        Returns:
            AbstractSideChannel: the instance of side channel.
        """
        raise NotImplementedError()


class UDEEnvironmentAdapterInterface(UDEEnvironmentInterface, metaclass=abc.ABCMeta):
    """
    UDEEnvironmentAdapterInterface
    """
