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
"""A class for UDE Server."""
from copy import deepcopy
import grpc
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import socket
from sys import platform
from threading import Event, RLock
import threading
from typing import Union, Dict, Optional, List, Tuple, Any, Iterable
import traceback
import sys

from ude.serialization_context import UDESerializationContext
from ude.communication.grpc_auth_interceptor import AuthInterceptor

from ude.ude_objects.ude_pb2_grpc import (
    UDEProtoServicer,
    add_UDEProtoServicer_to_server
)
from ude.ude_objects.ude_message_pb2 import (
    UDEMessageProto,
    UDEMessageHeaderProto
)
from ude.ude_objects.ude_empty_message_pb2 import UDEEmptyMessageProto
from ude.ude_objects.ude_side_channel_message_pb2 import (
    UDESideChannelMessageProto,
)
from ude.ude_objects.ude_data_message_pb2 import UDEDataMessageProto

from ude.side_channels.ude_side_channel import (
    SideChannelObserverInterface,
    AbstractSideChannel
)
import ude.communication.constants as const
from ude.ude_typing import MultiAgentDict, UDEStepResult, UDEResetResult, SideChannelData, AgentID
from ude.exception import (UDEClientError, UDEClientException,
                           UDEServerException,
                           UDECommunicationException,
                           UDEEnvException)
from ude.environment.ude_environment import UDEEnvironment

from grpc import Compression, ServerCredentials
from gym.spaces.space import Space


class UDEServicerImplementation(UDEProtoServicer):
    """
    UDEServicer Implementation for GRPC Servicer.
    """

    def __init__(self, server: object, timeout_wait: float = 60.0):
        """
        Initialize UDEServicer Implementation for GRPC Servicer.

        Args:
            server (UDEServer): UDE Server instance
            timeout_wait (float): Maximum wait time in seconds for step request from UDE Client.
        """
        self._server = server
        self._timeout_wait = timeout_wait
        self._context = UDESerializationContext.get_context()

    def step(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle step request from UDE Client.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with UDEMessageProto message containing
                             serialized return values from step function.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.DATA)

            step_data = self._context.deserialize(request.dataMsg.data)
            step_event = self._server.step(step_data)

            if step_event.wait(timeout=self._timeout_wait):
                step_info = self._server.get_step_info()
                if step_info is None:
                    raise UDEServerException(message='server closed',
                                             error_code=500)
                serialized_obj = bytes(self._context.serialize(step_info).to_buffer())
                msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      dataMsg=UDEDataMessageProto(data=serialized_obj))
            else:
                raise UDEServerException(message='step function timed out',
                                         error_code=500)
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def reset(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle reset request from UDE Client.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with UDEMessageProto message containing
                             serialized observation from reset function.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.EMPTY)
            try:
                reset_info = self._server.reset()
            except Exception as ex:
                traceback.print_exc()
                reset_info = UDEEnvException(ex)

            if reset_info is None:
                raise UDEServerException(message='server closed',
                                         error_code=500)
            serialized_obj = self._context.serialize(reset_info).to_buffer()
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  dataMsg=UDEDataMessageProto(data=bytes(serialized_obj)))
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def close(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle close request from UDE Client.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with empty UDEMessageProto message with 200 status.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.EMPTY)
            self._server.close()
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  emptyMsg=UDEEmptyMessageProto())
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def observation_space(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle observation space request from UDE Client.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with UDEMessageProto message containing
                             serialized observation from reset function.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.EMPTY)
            try:
                observation_space = self._server.observation_space
            except Exception as ex:
                traceback.print_exc()
                observation_space = UDEEnvException(ex)

            if observation_space is None:
                raise UDEServerException(message='server closed',
                                         error_code=500)
            serialized_obj = self._context.serialize(observation_space).to_buffer()
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  dataMsg=UDEDataMessageProto(data=bytes(serialized_obj)))
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def action_space(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle action space request from UDE Client.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with UDEMessageProto message containing
                             serialized observation from reset function.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.EMPTY)
            try:
                action_space = self._server.action_space
            except Exception as ex:
                traceback.print_exc()
                action_space = UDEEnvException(ex)

            if action_space is None:
                raise UDEServerException(message='server closed',
                                         error_code=500)
            serialized_obj = self._context.serialize(action_space).to_buffer()
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  dataMsg=UDEDataMessageProto(data=bytes(serialized_obj)))
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def side_channel_send(self, request: UDEMessageProto, context: object) -> UDEMessageProto:
        """
        Handle side channel request from UDE Client to relay the message to environment.

        Args:
            request (UDEMessageProto): the request message from UDE Client
            context (object): the context object used by GRPC Servicer.

        Returns:
            UDEMessageProto: Response with empty UDEMessageProto message with 200 status.
        """
        try:
            self.validate_msg(request, const.UDEMessageType.SIDE_CHANNEL)

            channel_msg = request.sideChannelMsg
            key = channel_msg.key
            val = getattr(channel_msg, channel_msg.WhichOneof('data')).val
            store_local = channel_msg.store_local

            self._server.send(key=key, value=val, store_local=store_local)
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  emptyMsg=UDEEmptyMessageProto())
        except Exception as e:
            traceback.print_exc()
            self._server.shutdown()
            raise e
        return msg

    def side_channel_stream(self, request: object, context: object) -> UDEMessageProto:
        """
        Stream the new side channel message from environment to relay to all the UDE clients.

        Args:
            request (object): Not used.
            context (object): Not used.

        Returns:
            UDEMessageProto: UDEMessageProto containing side channel message from the environment.
        """
        msg_queue = Queue()
        self._server.add_channel_queue(msg_queue)
        while True:
            msg = msg_queue.get()
            if msg.header.status != 200:
                return msg
            yield msg

    @staticmethod
    def validate_msg(msg: UDEMessageProto, expected_msg_type: Union[const.UDEMessageType, str]) -> bool:
        """
        Validate the received message from UDEClients.

        Args:
            msg (UDEMessageProto): the message to validate
            expected_msg_type (Union[const.UDEMessageType, str]): expected type of the message

        Returns:
            bool: True if successfully validated, otherwise UDEClientError/Exception is raised.
        """
        if isinstance(expected_msg_type, const.UDEMessageType):
            expected_msg_type = expected_msg_type.value

        if msg.header.status != 200:
            if msg.header.status < 500:
                raise UDEClientError(message=msg.header.message,
                                     error_code=msg.header.status)
            else:
                raise UDEClientException(message=msg.header.message,
                                         error_code=msg.header.status)
        if msg.WhichOneof('msg') != expected_msg_type:
            err_msg = "Received unexpected message type (expected: {} received: {})".format(expected_msg_type,
                                                                                            msg.WhichOneof('msg'))
            raise UDEClientException(message=err_msg,
                                     error_code=500)
        return True


class UDEServer(SideChannelObserverInterface):
    """
    UDEServer class to support remote UDEEnvironment
    """

    def __init__(self,
                 ude_env: UDEEnvironment,
                 step_invoke_type: const.UDEStepInvokeType = const.UDEStepInvokeType.WAIT_FOREVER,
                 step_invoke_period: Union[int, float] = 120.0,
                 num_agents: int = 1,
                 port: Optional[int] = None,
                 options: Optional[List[Tuple[str, Any]]] = None,
                 compression: Compression = Compression.NoCompression,
                 credentials: Optional[Union[ServerCredentials, Iterable[str], Iterable[bytes]]] = None,
                 auth_key: Optional[str] = None,
                 timeout_wait: Union[int, float] = 60.0,
                 max_workers: int = sys.maxsize,
                 **kwargs):
        """
        Initialize UDE Server.

        Args:
            ude_env (UDEEnvironment): Actual UDE Environment instance.
            step_invoke_type (const.UDEStepInvokeType):  step invoke type (WAIT_FOREVER vs PERIODIC)
            step_invoke_period (Union[int, float]): step invoke period (used only with PERIODIC step_invoke_type)
            num_agents (int): number of agents in the environment. (default: 1)
            port (Optional[int]): Port to use for UDE Server (default: 3003)
            options (Optional[List[Tuple[str, Any]]]): An optional list of key-value pairs
                                                        (:term:`channel_arguments` in gRPC runtime)
                                                        to configure the channel.
            compression (Compression) = channel compression type (default: NoCompression)
            credentials (Optional[Union[ServerCredentials, Iterable[str], Iterable[bytes]]]): grpc.ServerCredentials,
                the path to certificate private key and body/chain file, or bytes of the certificate private
                key and body/chain to use with an SSL-enabled Channel.
            auth_key (Optional[str]): channel authentication key (only applied when credentials are provided).
            timeout_wait (Union[int, float]): the maximum wait time to respond step request to UDE clients.
            max_workers (int): the maximum number of grpc.io server threads (This must be larger than num_agents.).
            kwargs: Arbitrary keyword arguments for grpc.server
        """
        self._side_channel = ude_env.side_channel
        self._side_channel.register(self)
        self._ude_env = ude_env

        self._action_dict = dict()
        self._step_info = None

        self._port = port or const.UDE_COMM_DEFAULT_PORT

        option_list = options or []
        custom_options = {item[0]: item[1] for item in option_list}
        options = {'grpc.max_send_message_length': const.GRPC_MAX_MESSAGE_LENGTH,
                   'grpc.max_receive_message_length': const.GRPC_MAX_MESSAGE_LENGTH}
        options.update(custom_options)
        self._options = [(item[0], item[1]) for item in options.items()]
        self._compression = compression
        self._credentials = UDEServer.to_server_credentials(credentials=credentials)
        self._auth_key = auth_key
        self._kwargs = kwargs

        self._timeout_wait = timeout_wait
        self._server = None
        self._env_to_remote = None
        self._is_open = False
        self._channel_queue_set = set()
        self._num_agent = num_agents
        if max_workers < num_agents:
            raise ValueError("max_workers ({}) must be greater than num_agents ({}).".format(max_workers, num_agents))
        self._max_workers = max_workers

        self._shutdown_lock = RLock()
        self._shutdown_event = Event()

        self._step_info_ready_event = Event()
        self._step_info_ready_event_lock = RLock()

        self._step_invoke_type = step_invoke_type
        # self._step_invoke_period is only used with UDEStepInvokeType.PERIODIC
        self._step_invoke_period = step_invoke_period
        self._invoke_step_event = Event()
        self._invoke_step_event_lock = RLock()
        self._invoke_step_thread = None
        self._should_stop_invoke_step_thread = False

    @staticmethod
    def to_server_credentials(credentials: Optional[Union[ServerCredentials,
                                                          Iterable[str],
                                                          Iterable[bytes]]]) -> ServerCredentials:
        """
        Convert given argument to grpc server credentials.

        Args:
            credentials (Optional[Union[ServerCredentials, Iterable[str], Iterable[bytes]]]): grpc.ServerCredentials,
                the path to certificate private key and body/chain file, or bytes of the certificate private
                key and body/chain to use with an SSL-enabled Channel.

        Returns:
            ServerCredentials: converted server credential.
        """
        if credentials and not isinstance(credentials, ServerCredentials):
            private_key, cert = credentials
            if os.path.isfile(private_key):
                with open(private_key, 'rb') as f:
                    private_key = f.read()
            if os.path.isfile(cert):
                with open(cert, 'rb') as f:
                    cert = f.read()
            credentials = grpc.ssl_server_credentials(((private_key, cert), ))
        return credentials

    @property
    def env(self) -> UDEEnvironment:
        """
        Returns UDE environment instance

        Returns:
            UDEEnvironment: UDE environment instance
        """
        return self._ude_env

    @property
    def side_channel(self) -> AbstractSideChannel:
        """
        Returns side channel to send and receive data to/from actual environment

        Returns:
            AbstractSideChannel: the instance of side channel.
        """
        return self._side_channel

    @property
    def step_invoke_type(self) -> const.UDEStepInvokeType:
        """
        Return step invoke type [WAIT_FOREVER | PERIODIC]

        Returns:
            const.UDEStepInvokeType: step invoke type
        """
        return self._step_invoke_type

    @step_invoke_type.setter
    def step_invoke_type(self, value: const.UDEStepInvokeType) -> None:
        """
        Sets step invoke type [WAIT_FOREVER | PERIODIC]

        Args:
            value (const.UDEStepInvokeType): step invoke type
        """
        self._step_invoke_type = value

    @property
    def step_invoke_period(self) -> Union[None, float, int]:
        """
        Return step invoke period [None | >0]

        Returns:
            Union[None, float, int]: step invoke period in seconds.
        """
        return self._step_invoke_period

    @step_invoke_period.setter
    def step_invoke_period(self, value: Union[None, float, int]) -> None:
        """
        Sets step invoke period [None | >0]

        Args:
            value (Union[None, float, int]): step invoke period in seconds.
        """
        self._step_invoke_period = value

    @property
    def timeout_wait(self) -> float:
        """
        Return the maximum timeout in seconds for step function waiting.

        Returns:
            float: the maximum timeout in seconds for step function waiting.
        """
        return self._timeout_wait

    @timeout_wait.setter
    def timeout_wait(self, value: float) -> None:
        """
        Sets the maximum timeout in seconds for step function waiting.

        Args:
            value (float): the maximum timeout in seconds for step function waiting.
        """
        self._timeout_wait = value

    @property
    def port(self) -> int:
        """
        Return the port number

        Returns:
            int: the port number

        """
        return self._port

    @property
    def options(self) -> List[Tuple[str, Any]]:
        """
        Returns the copy of grpc options.

        Returns:
            List[Tuple[str, Any]]: the copy of grpc options.
        """
        return deepcopy(self._options)

    @property
    def compression(self) -> Compression:
        """
        Returns the channel compression type of UDE Server.

        Returns:
            Compression: the grpc.Compression type.
        """
        return self._compression

    @property
    def credentials(self) -> ServerCredentials:
        """
        Returns the grpc.ServerCredentials.

        Returns:
            ServerCredentials: the grpc.ServerCredentials.
        """
        return self._credentials

    @property
    def auth_key(self) -> str:
        """
        Return the authentication key

        Returns:
            str: the channel authentication key
        """
        return self._auth_key

    @property
    def is_open(self) -> bool:
        """
        Return whether the server is open or not.

        Returns:
            bool: The flag whether the server is open or not.
        """
        return self._is_open

    @property
    def num_agent(self) -> int:
        """
        Return the number of agents configured.

        Returns:
            int: the number of agents.
        """
        return self._num_agent

    @num_agent.setter
    def num_agent(self, value: int) -> None:
        """
        Sets the number of agents.

        Args:
            value (int): the number of agents.
        """
        self._num_agent = value

    def on_received(self, side_channel: AbstractSideChannel,
                    key: str,
                    value: SideChannelData) -> None:
        """
        Relay the side channel message from environment to all UDE clients connected.

        Args:
            side_channel (AbstractSideChannel): the side channel to environment.
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
        """
        data_msg = const.BUILTIN_TYPE_TO_SIDE_CHANNEL_DATA_MSG[type(value)](val=value)
        valarg = {const.BUILTIN_TYPE_TO_SIDE_CHANNEL_MSG_ARG[type(value)]: data_msg}

        side_channel_msg = UDESideChannelMessageProto(key=key,
                                                      **valarg)

        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              sideChannelMsg=side_channel_msg)
        if not self._is_open:
            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500,
                                                               message='server closed'),
                                  emptyMsg=UDEEmptyMessageProto())
        for queue in self._channel_queue_set:
            queue.put(msg)

    def add_channel_queue(self, channel_queue: Queue) -> None:
        """
        When UDE Client is connected, add side channel queue,
        so UDE Server can relay the side channel message from environment
        to all the UDE clients connected.

        Args:
            channel_queue (Queue): UDE Client's side channel queue.
        """
        self._channel_queue_set.add(channel_queue)

    def _invoke_step(self) -> None:
        """
        Execute in thread to invoke the step function of environment.
        - In case of step_invoke_type, Periodic, when time-up, we need
          mechanism to invoke environment step function regardless all
          the agents' actions are received from UDE Clients or not.
        - The best way to manage is execute this in separate thread
          with waiting for event.
        """
        try:
            while self._is_open:
                with self._invoke_step_event_lock:
                    if self._should_stop_invoke_step_thread:
                        # if stop is requested, then exit the thread
                        break
                step_invoke_period = self._step_invoke_period
                if self._step_invoke_type == const.UDEStepInvokeType.WAIT_FOREVER:
                    # If StepInvokeType is WAIT_FOREVER then event should wait indefinitely.
                    step_invoke_period = None
                self._invoke_step_event.wait(timeout=step_invoke_period)
                if not self._is_open:
                    # if server is closed then exit the thread
                    break
                with self._invoke_step_event_lock:
                    if self._should_stop_invoke_step_thread:
                        # if stop is requested, then exit the thread
                        break
                    # Setup new event to wait for step invoke event.
                    # This way, if the event set by other threads, holding old event,
                    # won't have impact on new step invoke waiting.
                    self._invoke_step_event = Event()

                try:
                    self._step_info = self._ude_env.step(self._action_dict)
                except Exception as ex:
                    traceback.print_exc()
                    self._step_info = UDEEnvException(ex)

                self._action_dict = {}
                with self._step_info_ready_event_lock:
                    self._step_info_ready_event.set()
                    self._step_info_ready_event = Event()
        except Exception:
            traceback.print_exc()
            self.shutdown()

    def step(self, action_dict: MultiAgentDict) -> Event:
        """
        Stack the step request from UDE Client. The agent action from request message
        will be stacked, and when all agents' actions are received, it will invoke
        the environment step with actions collected, and set the event when
        next observation(s), reward(s), done(s), agent action(s) are ready.

        Args:
            action_dict (MultiAgentDict): the dict containing the agent name as key and action(s) as value.

        Returns:
            threading.Event: step info ready event, when set, all data needs to be returned as the response
                             for step request, are ready.

        """
        with self._invoke_step_event_lock:
            if self._invoke_step_thread is None:
                self._should_stop_invoke_step_thread = False
                self._invoke_step_thread = threading.Thread(target=self._invoke_step)
                self._invoke_step_thread.start()
            invoke_step_event = self._invoke_step_event

        with self._step_info_ready_event_lock:
            ret_event = self._step_info_ready_event

        self._action_dict.update(action_dict)
        if len(self._action_dict) >= self._num_agent:
            invoke_step_event.set()
        return ret_event

    def get_step_info(self) -> UDEStepResult:
        """
        Return step return values.

        Returns:
            UDEStepResult: observations, rewards, dones, last_actions, info
        """
        return self._step_info

    def reset(self) -> UDEResetResult:
        """
        Reset the environment.

        Returns:
            UDEResetResult: first observation and info in new episode.
        """
        with self._invoke_step_event_lock:
            # Clean up invoke_step thread.
            # - Killing the thread and re-starting new thread is expensive,
            #   but we are doing this only at reset, so performance degradation
            #   should be small. Also this way, it is much cleaner to handle
            #   multi-agents' reset case.
            self._should_stop_invoke_step_thread = True
            self._invoke_step_event.set()
            self._invoke_step_event = Event()
        if self._invoke_step_thread is not None:
            self._invoke_step_thread.join()
            self._invoke_step_thread = None
        return self._ude_env.reset()

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        - For now, we also shutdown the server.
        TODO: Consider whether this is right behavior.
        """
        self._ude_env.close()
        # TODO: should we shutdown server?
        self.shutdown()

    @property
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        return self._ude_env.observation_space

    @property
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        return self._ude_env.action_space

    def send(self, key: str, value: SideChannelData, store_local: bool = False) -> None:
        """
        Send key and value pair to environment as a side channel message.

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
            store_local (bool, optional): The flag whether to store locally or not.
        """
        self._side_channel.send(key=key, value=value, store_local=store_local)
        if store_local:
            # Sync to all other clients
            self.on_received(side_channel=self._side_channel, key=key, value=value)

    def start(self) -> 'UDEServer':
        """
        Start the UDE Server.

        Returns:
            UDEServer: self
        """
        self.check_port(self._port)

        try:
            # Establish communication grpc
            interceptors = None
            if self._credentials and self._auth_key:
                interceptors = (AuthInterceptor(key=self._auth_key),)
            self._server = grpc.server(ThreadPoolExecutor(max_workers=self._max_workers),
                                       interceptors=interceptors,
                                       options=self._options,
                                       compression=self._compression,
                                       **self._kwargs)
            self._env_to_remote = UDEServicerImplementation(server=self,
                                                            timeout_wait=self._timeout_wait)
            add_UDEProtoServicer_to_server(
                self._env_to_remote, self._server
            )
            # Using unspecified address, which means that grpc is communicating on all IPs
            # This is so that the docker container can connect.
            if self._credentials:
                self._server.add_secure_port("[::]:" + str(self._port), self._credentials)
            else:
                self._server.add_insecure_port("[::]:" + str(self._port))
            self._server.start()
            self._is_open = True
            return self
        except Exception:
            raise UDECommunicationException(self._port)

    @staticmethod
    def check_port(port: int) -> None:
        """
        Attempts to bind to the requested communicator port, checking if it is already in use.
        - If it is in use then raises UDECommunicationException.

        Args:
            port (int): the port to check whether it is already in  use.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if platform == "linux" or platform == "linux2":
            # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
            # SO_REUSEADDR frees the port right after closing the environment
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logging.info("platform: {}".format(platform))
        try:
            s.bind(("localhost", port))
        except socket.error:
            raise UDECommunicationException(port)
        finally:
            s.close()

    def shutdown(self) -> None:
        """
        Sends a shut down signal to the environment, and closes the grpc connection.
        - Also shut down the server.
        """
        with self._shutdown_lock:
            if self._is_open:
                self._is_open = False
                if self._invoke_step_thread and self._invoke_step_thread != threading.current_thread():
                    self._should_stop_invoke_step_thread = True
                    self._invoke_step_event.set()
                    self._invoke_step_thread.join()
                self._invoke_step_thread = None

                self._step_info_ready_event.set()

                self._side_channel.unregister(self)
                # Clean up side channel stream queues
                for queue in self._channel_queue_set:
                    msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500),
                                          emptyMsg=UDEEmptyMessageProto())
                    queue.put(msg)
                self._channel_queue_set = set()
                self._server.stop(False)
                self._shutdown_event.set()

    def spin(self) -> None:
        """
        Spin wait till server shuts down.
        """
        while not self._shutdown_event.wait(1):
            continue
