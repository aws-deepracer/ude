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
"""A class for UDE Client."""
from copy import deepcopy
from enum import Enum
import grpc
import os
import logging
import threading
from typing import Optional, Union, Any, Dict, cast, List, Tuple

from ude.serialization_context import UDESerializationContext
from ude.communication.grpc_auth import GrpcAuth
import ude.ude_objects.ude_pb2_grpc as rpc
from ude.ude_objects.ude_empty_message_pb2 import UDEEmptyMessageProto
from ude.ude_objects.ude_message_pb2 import (
    UDEMessageProto,
    UDEMessageHeaderProto
)
from ude.ude_objects.ude_side_channel_message_pb2 import (
    UDESideChannelMessageProto,
)
from ude.ude_objects.ude_data_message_pb2 import (
    UDEDataMessageProto
)

import ude.communication.constants as const
from ude.exception import UDEException, UDEServerError, UDEServerException, UDEEnvException
from ude.ude_typing import MultiAgentDict, UDEStepResult, UDEResetResult, SideChannelData, AgentID
from ude.side_channels.ude_side_channel import AbstractSideChannel

from gym.spaces.space import Space

from grpc import Compression, ChannelCredentials


class RpcFuncNames(Enum):
    """
    RPC Function names supported by UDE Client
    """
    SIDE_CHANNEL_STREAM = 'side_channel_stream'
    SIDE_CHANNEL_SEND = 'side_channel_send'
    STEP = 'step'
    RESET = 'reset'
    CLOSE = 'close'
    OBSERVATION_SPACE = 'observation_space'
    ACTION_SPACE = 'action_space'


class UDEClient(AbstractSideChannel):
    """
    UDEClient class to support remote UDEEnvironment
    """

    def __init__(self, address: str, port: Optional[int] = None,
                 options: Optional[List[Tuple[str, Any]]] = None,
                 compression: Compression = Compression.NoCompression,
                 credentials: Optional[Union[str, bytes, ChannelCredentials]] = None,
                 auth_key: Optional[str] = None,
                 timeout: float = 10.0,
                 max_retry_attempts: int = 5):
        """
        Initialize UDEClient

        Args:
            address (str): address of UDE Server
            port (Optional[int]): the port of UDE Server (default: 3003)
            options (Optional[List[Tuple[str, Any]]]): An optional list of key-value pairs
                                                        (:term:`channel_arguments` in gRPC runtime)
                                                        to configure the channel.
            compression (Compression): channel compression type (default: NoCompression)
            credentials: Optional[Union[str, bytes, ChannelCredentials]]: grpc.ChannelCredentials, the path to
                certificate file or bytes of the certificate to use with an SSL-enabled Channel.
            auth_key (Optional[str]): channel authentication key (only applied when credentials are provided).
            timeout (float): the time-out of grpc.io call
            max_retry_attempts (int): maximum number of retry
        """
        super().__init__()

        self._context = UDESerializationContext.get_context()

        self._timeout = timeout
        self._max_retry_attempts = int(max_retry_attempts)
        self._address = address
        self._port = port or const.UDE_COMM_DEFAULT_PORT
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()

        option_list = options or []
        custom_options = {item[0]: item[1] for item in option_list}
        options = {'grpc.max_send_message_length': const.GRPC_MAX_MESSAGE_LENGTH,
                   'grpc.max_receive_message_length': const.GRPC_MAX_MESSAGE_LENGTH}
        options.update(custom_options)
        self._options = [(item[0], item[1]) for item in options.items()]
        self._compression = compression
        self._credentials = UDEClient.to_channel_credentials(credentials=credentials)
        self._auth_key = auth_key

        self._should_stop_receiver_thread = False
        self._channel = None
        self._conn = None
        self._receiver_thread = None
        self._connect()

    @staticmethod
    def to_channel_credentials(credentials: Optional[Union[str, bytes, ChannelCredentials]]) -> ChannelCredentials:
        """
        Convert given argument to grpc channel credentials.

        Args:
            credentials: Optional[Union[str, bytes, ChannelCredentials]]: grpc.ChannelCredentials, the path to
                certificate file or bytes of the certificate to use with an SSL-enabled Channel.

        Returns:
            ChannelCredentials: converted channel credential.
        """
        if credentials and not isinstance(credentials, ChannelCredentials):
            if os.path.isfile(credentials):
                with open(credentials, 'rb') as f:
                    credentials = f.read()
            credentials = grpc.ssl_channel_credentials(credentials)
        return credentials

    @property
    def port(self) -> int:
        """
        Returns the port value that UDE Server is listening.

        Returns:
            int: the port value that UDE Server is listening.
        """
        return self._port

    @property
    def address(self) -> str:
        """
        Returns the address of UDE Server.

        Returns:
            str: the address of UDE Server.
        """
        return self._address

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
        Return the channel compression type of UDE Server.

        Returns:
            Compression: grpc.Compression type.
        """
        return self._compression

    @property
    def credentials(self) -> ChannelCredentials:
        """
        Return the grpc.ChannelCredentials.

        Returns:
            ChannelCredentials: the grpc.ChannelCredentials.
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
    def timeout(self) -> float:
        """
        Returns the timeout in seconds for the response of rpc call.

        Returns:
            float: the timeout in seconds for the response of rpc call.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: Union[float, None]) -> None:
        """
        Sets the timeout in seconds for the response of rcp call.

        Args:
            value (Union[float, None]): new timeout in seconds for the response of rcp call.
        """
        self._timeout = value

    @property
    def max_retry_attempts(self) -> int:
        """
        Returns the max retry attempts.

        Returns:
            int: the max retry attempts.
        """
        return self._max_retry_attempts

    @max_retry_attempts.setter
    def max_retry_attempts(self, value: int) -> None:
        """
        Sets the new max retry attempts value.

        Args:
            value (int): the new max retry attempts value.
        """
        self._max_retry_attempts = int(value)

    def on_message_received(self) -> None:
        """
        This function is executed in different thread to handle side_channel message received
        from UDEServer.
        """
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        try_count = 0
        max_retry_attempts = self.max_retry_attempts
        try:
            while True:
                with self._lock:
                    # Store connection object to check equality later.
                    conn_obj = self._conn
                try:
                    # Below line will wait for new messages from the server!
                    # * This is stream communicator, so we should wait indefinitely
                    #   till we get a new message. Therefore, we need to pass None for timeout.
                    # * self._conn.side_chnnel_stream returns iterator that on subsequent next call
                    #   it waits till next message and yield next message, so _call_with_retry
                    #   is only helpful on handling transient error during initiation of stream connection.
                    #   Thus, we need to restart for-loop when grpc stream transiently fails.
                    for response in self._call_with_retry(RpcFuncNames.SIDE_CHANNEL_STREAM,
                                                          msg=msg,
                                                          timeout=None,
                                                          max_retry_attempts=max_retry_attempts):
                        try:
                            # Reset try-count once we successfully receive one message.
                            try_count = 0
                            self.validate_msg(msg=response,
                                              expected_msg_type=const.UDEMessageType.SIDE_CHANNEL)
                        except UDEException as ex:
                            raise ex
                        channel_msg = response.sideChannelMsg
                        key = channel_msg.key
                        value = getattr(channel_msg, channel_msg.WhichOneof('data')).val
                        self.store(key=key, value=value)
                        self.notify(key=key, value=value)
                        if self._should_stop_receiver_thread:
                            return
                except grpc.RpcError as ex:
                    if self._should_stop_receiver_thread:
                        return
                    with self._lock:
                        if conn_obj is not self._conn:
                            # Connection object reset by other function.
                            # Just continue to use new connection object
                            continue

                    try_count += 1
                    if try_count > max_retry_attempts:
                        raise ex
                    log_msg_format = "[UDEClient] Failed on side_channel_stream, Retry count: {0}/{1}: {2}"
                    logging.info(log_msg_format.format(str(try_count),
                                                       str(max_retry_attempts),
                                                       ex))
                    self._reset_channel()

                if self._should_stop_receiver_thread:
                    return
        finally:
            self._close()

    def _connect(self) -> None:
        """
        Connect to UDE Server.
        """
        with self._lock:
            if self._channel is None:
                self._reset_channel()
                self._should_stop_receiver_thread = False
                self._receiver_thread = threading.Thread(target=self.on_message_received)
                self._receiver_thread.start()

    def _reset_channel(self) -> None:
        """
        Resets grpc.io channel.
        """
        with self._lock:
            try:
                if self._channel is not None:
                    logging.info("[UDEClient] Resetting grpc.io channel...")
                    self._channel.close()
            except Exception as ex:
                logging.info("[UDEClient] Ignoring: Exception raised from channel.close during reset_channel: {}".format(ex))
            finally:
                if self._credentials and self._auth_key:
                    logging.debug("[UDEClient] Connecting secure channel with credentials and auth_key...")
                    metadata_call_credentials = grpc.metadata_call_credentials(GrpcAuth(key=self._auth_key))
                    self._channel = grpc.secure_channel(self._address + ':' + str(self._port),
                                                        credentials=grpc.composite_channel_credentials(
                                                            self._credentials,
                                                            metadata_call_credentials),
                                                        options=self._options,
                                                        compression=self._compression)
                elif self._credentials:
                    logging.debug("[UDEClient] Connecting secure channel with credentials...")
                    self._channel = grpc.secure_channel(self._address + ':' + str(self._port),
                                                        credentials=self._credentials,
                                                        options=self._options,
                                                        compression=self._compression)
                else:
                    logging.debug("[UDEClient] Connecting insecure channel...")
                    self._channel = grpc.insecure_channel(self._address + ':' + str(self._port),
                                                          options=self._options,
                                                          compression=self._compression)
                self._conn = rpc.UDEProtoStub(self._channel)
                logging.debug("[UDEClient] Connection established!")

    def _call_with_retry(self,
                         rpc_func_name: RpcFuncNames,
                         msg: Any,
                         timeout: Union[float, None],
                         max_retry_attempts: int) -> Any:
        """
        Call RPC function with retries.

        Args:
            rpc_func_name (RpcFuncNames): RPC function name Enum type.
            msg (Any): message for RPC function.
            timeout (Union[float, None]): timeout in seconds (None waits indefinitely).
            max_retry_attempts (int): maximum retry attempts.

        Returns:
            Any: the response from RPC function call.
        """
        try_count = 0
        while True:
            try:
                with self._lock:
                    conn_obj = self._conn
                    rpc_func = getattr(self._conn, rpc_func_name.value)
                return rpc_func(msg, timeout=timeout)
            except grpc.RpcError as ex:
                with self._lock:
                    if conn_obj is not self._conn:
                        # Connection object reset by other function.
                        # Just continue to use new connection object
                        continue
                try_count += 1
                if try_count > max_retry_attempts:
                    logging.error("[UDEClient] RPC call failed with {} retries".format(str(max_retry_attempts)))
                    raise ex
                logging.info("[UDEClient] RPC call failed, Retry count: {0}/{1}: {2}".format(str(try_count),
                                                                                             str(max_retry_attempts),
                                                                                             ex))
                self._reset_channel()

    def _send(self, key: str, value: SideChannelData, store_local: bool = False) -> None:
        """
        Send key and value pair to UDEServer as a side channel message.

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
            store_local (bool, optional): The flag whether to store locally or not.
        """
        try:
            data_msg = const.BUILTIN_TYPE_TO_SIDE_CHANNEL_DATA_MSG[type(value)](val=value)
            valarg = {const.BUILTIN_TYPE_TO_SIDE_CHANNEL_MSG_ARG[type(value)]: data_msg}
        except KeyError:
            raise TypeError("Not supported type: {}".format(type(value)))

        try:
            self._connect()
            side_channel_msg = UDESideChannelMessageProto(key=key,
                                                          store_local=store_local,
                                                          **valarg)

            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
            self._call_with_retry(rpc_func_name=RpcFuncNames.SIDE_CHANNEL_SEND,
                                  msg=msg,
                                  timeout=self.timeout,
                                  max_retry_attempts=self.max_retry_attempts)
        except Exception as ex:
            self._close()
            raise ex

    def _close(self) -> None:
        """
        Clean up resources.
        """
        if self._close_lock.acquire(False):
            try:
                with self._lock:
                    if self._channel is not None:
                        self._should_stop_receiver_thread = True
                        self._channel.close()
                        if self._receiver_thread and self._receiver_thread != threading.current_thread():
                            self._receiver_thread.join()
                    self._channel = None
                    self._conn = None
                    self._receiver_thread = None
            finally:
                self._close_lock.release()

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
        if not isinstance(action_dict, dict):
            raise ValueError("action_dict must be dict type with agent name (str) as key and action (Any) as value!")

        try:
            self._connect()
            serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
            action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                         dataMsg=UDEDataMessageProto(data=serialized_obj))
            response = self._call_with_retry(rpc_func_name=RpcFuncNames.STEP,
                                             msg=action_msg,
                                             timeout=self.timeout,
                                             max_retry_attempts=self.max_retry_attempts)
            self.validate_msg(msg=response,
                              expected_msg_type=const.UDEMessageType.DATA)
            step_data = self._context.deserialize(response.dataMsg.data)
            if isinstance(step_data, UDEEnvException):
                raise step_data
            obs, reward, done, last_action, info = step_data
            return obs, reward, done, last_action, info
        except UDEEnvException:
            raise
        except Exception as ex:
            self._close()
            raise ex

    def reset(self) -> UDEResetResult:
        """
        Reset the Environment and start new episode.
        Also, returns the first observation for new episode started.

        Returns:
            UDEResetResult: first observation and info in new episode.
        """
        try:
            self._connect()

            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  emptyMsg=UDEEmptyMessageProto())
            response = self._call_with_retry(rpc_func_name=RpcFuncNames.RESET,
                                             msg=msg,
                                             timeout=self.timeout,
                                             max_retry_attempts=self.max_retry_attempts)
            self.validate_msg(msg=response,
                              expected_msg_type=const.UDEMessageType.DATA)
            reset_data = self._context.deserialize(response.dataMsg.data)
            if isinstance(reset_data, UDEEnvException):
                raise reset_data
            obs, info = reset_data
            return obs, info
        except UDEEnvException:
            raise
        except Exception as ex:
            self._close()
            raise ex

    def close(self) -> None:
        """
        Close the environment, and environment will be no longer available to be used.
        """
        self._connect()

        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        try:
            self._call_with_retry(rpc_func_name=RpcFuncNames.CLOSE,
                                  msg=msg,
                                  timeout=self.timeout,
                                  max_retry_attempts=self.max_retry_attempts)
        except Exception as ex:
            raise ex
        finally:
            self._close()

    @property
    def observation_space(self) -> Dict[AgentID, Space]:
        """
        Returns the observation spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the observation spaces of agents in env.
        """
        try:
            self._connect()

            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  emptyMsg=UDEEmptyMessageProto())
            response = self._call_with_retry(rpc_func_name=RpcFuncNames.OBSERVATION_SPACE,
                                             msg=msg,
                                             timeout=self.timeout,
                                             max_retry_attempts=self.max_retry_attempts)
            self.validate_msg(msg=response,
                              expected_msg_type=const.UDEMessageType.DATA)
            deserialized_obj = self._context.deserialize(response.dataMsg.data)
            if isinstance(deserialized_obj, UDEEnvException):
                raise deserialized_obj
            return cast(Dict[AgentID, Space], deserialized_obj)
        except UDEEnvException:
            raise
        except Exception as ex:
            self._close()
            raise ex

    @property
    def action_space(self) -> Dict[AgentID, Space]:
        """
        Returns the action spaces of agents in env.

        Returns:
            Dict[AgentID, Space]: the action spaces of agents in env.
        """
        try:
            self._connect()

            msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  emptyMsg=UDEEmptyMessageProto())
            response = self._call_with_retry(rpc_func_name=RpcFuncNames.ACTION_SPACE,
                                             msg=msg,
                                             timeout=self.timeout,
                                             max_retry_attempts=self.max_retry_attempts)
            self.validate_msg(msg=response,
                              expected_msg_type=const.UDEMessageType.DATA)
            deserialized_obj = self._context.deserialize(response.dataMsg.data)
            if isinstance(deserialized_obj, UDEEnvException):
                raise deserialized_obj
            return cast(Dict[AgentID, Space], deserialized_obj)
        except UDEEnvException:
            raise
        except Exception as ex:
            self._close()
            raise ex

    @staticmethod
    def validate_msg(msg: UDEMessageProto, expected_msg_type: Union[const.UDEMessageType, str]) -> bool:
        """
        Validate the received message from UDEServer.

        Args:
            msg (UDEMessageProto): the message to validate
            expected_msg_type (Union[const.UDEMessageType, str]): expected type of the message

        Returns:
            bool: True if successfully validated, otherwise UDEServerError/Exception is raised.
        """
        if isinstance(expected_msg_type, const.UDEMessageType):
            expected_msg_type = expected_msg_type.value

        if msg.header.status != 200:
            if msg.header.status < 500:
                raise UDEServerError(message=msg.header.message,
                                     error_code=msg.header.status)
            else:
                raise UDEServerException(message=msg.header.message,
                                         error_code=msg.header.status)
        if msg.WhichOneof('msg') != expected_msg_type:
            err_msg = "Received unexpected message type (expected: {} received: {})".format(expected_msg_type,
                                                                                            msg.WhichOneof('msg'))
            raise UDEServerException(message=err_msg,
                                     error_code=500)
        return True
