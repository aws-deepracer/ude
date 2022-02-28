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
"""A class for Remove Environment Adapter."""
from typing import Optional, Dict, List, Tuple, Any, Union

from ude.environment.interfaces import UDEEnvironmentAdapterInterface
from ude.communication.ude_client import UDEClient
from ude.ude_typing import MultiAgentDict, UDEStepResult, UDEResetResult, AgentID
from ude.side_channels.ude_side_channel import AbstractSideChannel

from gym.spaces.space import Space
from grpc import Compression, ChannelCredentials


class RemoteEnvironmentAdapter(UDEEnvironmentAdapterInterface):
    """
    RemoteEnvironmentAdapter class to relay the message between UDE Server and UDE Client.
    - The instance of RemoteEnvironmentAdapter class resides in UDE Client.
    """
    def __init__(self,
                 address: str,
                 port: Optional[int] = None,
                 options: Optional[List[Tuple[str, Any]]] = None,
                 compression: Compression = Compression.NoCompression,
                 credentials: Optional[Union[str, bytes, ChannelCredentials]] = None,
                 auth_key: Optional[str] = None,
                 timeout: float = 10.0,
                 max_retry_attempts: int = 5):
        """
        Initialize RemoteEnvironmentAdapter.

        Args:
            address (str): address of UDE Server
            port (Optional[int]): the port of UDE Server (default: 3003)
            options (Optional[List[Tuple[str, Any]]]): An optional list of key-value pairs
                                                        (:term:`channel_arguments` in gRPC runtime)
                                                        to configure the channel.
            compression (Compression) = channel compression type (default: NoCompression)
            credentials: Optional[Union[str, bytes, ChannelCredentials]]: grpc.ChannelCredentials, the path to
                certificate file or bytes of the certificate to use with an SSL-enabled Channel.
            auth_key (Optional[str]): channel authentication key (only applied when credentials are provided).
            timeout (float): the time-out of grpc.io call
            max_retry_attempts (int): maximum number of retry
        """
        super().__init__()
        self._client = UDEClient(address=address,
                                 port=port,
                                 options=options,
                                 compression=compression,
                                 credentials=credentials,
                                 auth_key=auth_key,
                                 timeout=timeout,
                                 max_retry_attempts=max_retry_attempts)

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
        return self._client.step(action_dict=action_dict)

    def reset(self) -> UDEResetResult:
        """
        Reset the environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            UDEResetResult: first observation and info in new episode.
        """
        return self._client.reset()

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        self._client.close()

    @property
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        return self._client.observation_space

    @property
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        return self._client.action_space

    @property
    def side_channel(self) -> AbstractSideChannel:
        """
        Returns side channel to send and receive data from UDE Server

        Returns:
            AbstractSideChannel: the instance of side channel.
        """
        return self._client
