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
from unittest import TestCase
from unittest.mock import patch, MagicMock, ANY, PropertyMock

from ude.ude_objects.ude_message_pb2 import (
    UDEMessageProto,
    UDEMessageHeaderProto
)
from ude.ude_objects.ude_side_channel_message_pb2 import (
    UDESideChannelMessageProto,
)
from ude.ude_objects.ude_side_channel_message_pb2 import (
    UDEBoolDataProto, UDEIntDataProto,
    UDEFloatDataProto, UDEFloatListDataProto,
    UDEStringDataProto, UDEBytesDataProto
)
from ude.ude_objects.ude_data_message_pb2 import (
    UDEDataMessageProto
)
from ude.ude_objects.ude_empty_message_pb2 import UDEEmptyMessageProto

from ude.communication.ude_server import UDEServicerImplementation, UDEServer

from ude.communication.constants import (
    UDEMessageType, UDE_COMM_DEFAULT_PORT,
    UDEStepInvokeType
)
from ude.exception import (UDEServerException,
                           UDEClientError, UDEClientException,
                           UDECommunicationException,
                           UDEEnvException)
from ude.serialization_context import UDESerializationContext
from ude.communication.constants import GRPC_MAX_MESSAGE_LENGTH

import threading
import socket
import time
import grpc

from gym.spaces.space import Space


@patch("ude.communication.ude_server.UDEServer")
class UDEServicerImplementationTest(TestCase):
    def setUp(self) -> None:
        self._context = UDESerializationContext.get_context()

    def test_step(self, ude_server_mock):
        # action_msg expected from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        # Mock environment step
        step_event = threading.Event()
        ude_server_mock.return_value.step.return_value = step_event
        step_event.set()

        next_state = {"agent": "next_state"}
        done = {"agent": False}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        # Mock return value from environment step
        ude_server_mock.return_value.get_step_info.return_value = step_dict

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(step_dict).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))

        msg = servicer.step(action_msg, None)

        assert msg == expected_msg

    def test_step_wrong_type_from_client(self, ude_server_mock):
        # wrong action_msg from UDE Client
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock environment step
        step_event = threading.Event()
        ude_server_mock.return_value.step.return_value = step_event
        step_event.set()

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just ignore wrong type msg from the client
        with self.assertRaises(UDEClientException):
            _ = servicer.step(action_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_step_step_event_timeout(self, ude_server_mock):
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        # Mock environment step
        step_event = threading.Event()
        ude_server_mock.return_value.step.return_value = step_event

        # wait only 0.1 sec to expedite the test
        servicer = UDEServicerImplementation(ude_server_mock(), timeout_wait=0.1)

        # TODO: Consider raising Exception is right choice of action.
        # Maybe just ignore the timeout?
        with self.assertRaises(UDEServerException):
            _ = servicer.step(action_msg, None)
        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_step_empty_step_info(self, ude_server_mock):
        # action_msg expected from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        # Mock environment step
        step_event = threading.Event()
        ude_server_mock.return_value.step.return_value = step_event
        step_event.set()

        # Mock return value from environment step
        ude_server_mock.return_value.get_step_info.return_value = None

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just send back empty step msg...
        with self.assertRaises(UDEServerException):
            _ = servicer.step(action_msg, None)
        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_reset(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        next_state = {"agent": "next_state"}
        # Mock return value from environment step
        ude_server_mock.return_value.reset.return_value = next_state

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(next_state).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))

        msg = servicer.reset(client_msg, None)

        assert msg == expected_msg

    def test_reset_env_failure(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        exception = Exception("Some Error")
        ude_server_mock.return_value.reset.side_effect = exception
        env_err = UDEEnvException(exception)

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))
        with patch("traceback.print_exc") as print_exc_mock:
            msg = servicer.reset(client_msg, None)
            print_exc_mock.assert_called_once()
            assert msg == expected_msg

    def test_reset_wrong_type_from_client(self, ude_server_mock):
        # wrong msg from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        next_state = {"agent": "next_state"}
        # Mock return value from environment step
        ude_server_mock.return_value.reset.return_value = next_state

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just ignore wrong type msg from the client
        with self.assertRaises(UDEClientException):
            _ = servicer.reset(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_reset_empty_env_reset(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        ude_server_mock.return_value.reset.return_value = None

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEServerException):
            _ = servicer.reset(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_close(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        expected_msg = client_msg

        servicer = UDEServicerImplementation(ude_server_mock())

        msg = servicer.close(client_msg, None)

        assert msg == expected_msg

    def test_close_wrong_type_from_client(self, ude_server_mock):
        # wrong msg from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just ignore wrong type msg from the client
        with self.assertRaises(UDEClientException):
            _ = servicer.close(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_close_server_fault(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        ude_server_mock.return_value.close.side_effect = UDEServerException()

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEServerException):
            _ = servicer.close(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_observation_space(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        observation_space = {"agent": Space([4, 2])}
        # Mock return value from environment step
        ude_server_mock.return_value.observation_space = observation_space

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(observation_space).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))

        msg = servicer.observation_space(client_msg, None)

        assert msg == expected_msg

    def test_observation_space_env_failure(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        exception = Exception("Some Error")
        type(ude_server_mock.return_value).observation_space = PropertyMock(side_effect=exception)
        env_err = UDEEnvException(exception)

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))

        with patch("traceback.print_exc") as print_exc_mock:
            msg = servicer.observation_space(client_msg, None)
            print_exc_mock.assert_called_once()
            assert msg == expected_msg

    def test_observation_space_wrong_type_from_client(self, ude_server_mock):
        # wrong msg from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        observation_space = {"agent": Space([4, 2])}
        # Mock return value from environment step
        ude_server_mock.return_value.observation_space = observation_space

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just ignore wrong type msg from the client
        with self.assertRaises(UDEClientException):
            _ = servicer.observation_space(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_observation_space_empty_env_observation_space(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        ude_server_mock.return_value.observation_space = None

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEServerException):
            _ = servicer.observation_space(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_action_space(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        action_space = {"agent": Space([4, 2])}
        # Mock return value from environment step
        ude_server_mock.return_value.action_space = action_space

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(action_space).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))

        msg = servicer.action_space(client_msg, None)

        assert msg == expected_msg

    def test_action_space_env_failure(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        exception = Exception("Some Error")
        type(ude_server_mock.return_value).action_space = PropertyMock(side_effect=exception)
        env_err = UDEEnvException(exception)

        servicer = UDEServicerImplementation(ude_server_mock())

        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       dataMsg=UDEDataMessageProto(data=serialized_obj))
        with patch("traceback.print_exc") as print_exc_mock:
            msg = servicer.action_space(client_msg, None)

            print_exc_mock.assert_called_once()
            assert msg == expected_msg

    def test_action_space_wrong_type_from_client(self, ude_server_mock):
        # wrong msg from UDE Client
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        action_space = {"agent": Space([4, 2])}
        # Mock return value from environment step
        ude_server_mock.return_value.action_space = action_space

        servicer = UDEServicerImplementation(ude_server_mock())

        # TODO: Consider raising Exception is right choice of action.
        # Maybe we can just ignore wrong type msg from the client
        with self.assertRaises(UDEClientException):
            _ = servicer.action_space(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_action_space_empty_env_action_space(self, ude_server_mock):
        # empty_msg expected from UDE Client
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        # Mock return value from environment step
        ude_server_mock.return_value.action_space = None

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEServerException):
            _ = servicer.action_space(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_side_channel_send_bool(self, ude_server_mock):
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                               boolVal=UDEBoolDataProto(val=True))
                                     )

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=True,
                                                                  store_local=False)

    def test_side_channel_send_int(self, ude_server_mock):
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                               intVal=UDEIntDataProto(val=42))
                                     )

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=42,
                                                                  store_local=False)

    def test_side_channel_send_float(self, ude_server_mock):
        data = UDEFloatDataProto(val=42.42)
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                               floatVal=data)
                                     )

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=data.val,
                                                                  store_local=False)

    def test_side_channel_send_float_list(self, ude_server_mock):
        float_list_val = [42.42, 43.43]
        data = UDEFloatListDataProto(val=float_list_val)
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatListVal=data)
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=data.val,
                                                                  store_local=False)

    def test_side_channel_send_string(self, ude_server_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val))
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=string_val,
                                                                  store_local=False)

    def test_side_channel_send_bytes(self, ude_server_mock):
        bytes_val = "the answer is 42.".encode('utf-8')
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      bytesVal=UDEBytesDataProto(val=bytes_val))
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=bytes_val,
                                                                  store_local=False)

    def test_side_channel_send_wrong_message(self, ude_server_mock):
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     emptyMsg=UDEEmptyMessageProto())

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEClientException):
            _ = servicer.side_channel_send(client_msg, None)
        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_side_channel_send_store_local(self, ude_server_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=string_val,
                                                                  store_local=True)

    def test_side_channel_send_no_store_local(self, ude_server_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=False)
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        servicer = UDEServicerImplementation(ude_server_mock())

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       emptyMsg=UDEEmptyMessageProto())

        msg = servicer.side_channel_send(client_msg, None)

        assert msg == expected_msg
        ude_server_mock.return_value.send.assert_called_once_with(key="key",
                                                                  value=string_val,
                                                                  store_local=False)

    def test_side_channel_send_server_send_fault(self, ude_server_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        client_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     sideChannelMsg=side_channel_msg)

        ude_server_mock.return_value.send.side_effect = UDEServerException()

        servicer = UDEServicerImplementation(ude_server_mock())

        with self.assertRaises(UDEServerException):
            _ = servicer.side_channel_send(client_msg, None)

        ude_server_mock.return_value.shutdown.assert_called_once()

    def test_side_channel_stream(self, ude_server_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        env_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)

        env_msg2 = UDEMessageProto(header=UDEMessageHeaderProto(status=400),
                                   emptyMsg=UDEEmptyMessageProto())

        from queue import Queue
        msg_queue = Queue()
        msg_queue.put(env_msg)
        msg_queue.put(env_msg2)

        servicer = UDEServicerImplementation(ude_server_mock())

        with patch("ude.communication.ude_server.Queue") as queue_mock:
            queue_mock.return_value = msg_queue
            for msg in list(servicer.side_channel_stream(None, None)):
                assert msg == env_msg or msg == env_msg2

        ude_server_mock.return_value.add_channel_queue.assert_called_with(msg_queue)

    def test_validate_msg_empty(self, ude_server_mock):
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        assert UDEServicerImplementation.validate_msg(msg, UDEMessageType.EMPTY)

    def test_validate_msg_expected_type_string(self, ude_server_mock):
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        assert UDEServicerImplementation.validate_msg(msg, "emptyMsg")

    def test_validate_msg_step(self, ude_server_mock):
        next_state = {"agent": "next_state"}
        serialized_obj = bytes(self._context.serialize(next_state).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        assert UDEServicerImplementation.validate_msg(step_msg, UDEMessageType.DATA)

    def test_validate_msg_side_channel(self, ude_server_mock):
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        assert UDEServicerImplementation.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)

    def test_validate_msg_unexpected_msg_type(self, ude_server_mock):
        # client sent SideChannel message when expected Step msg.
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        with self.assertRaises(UDEClientException):
            UDEServicerImplementation.validate_msg(ude_msg, UDEMessageType.DATA)

    def test_validate_msg_client_fault(self, ude_server_mock):
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=400),
                                  sideChannelMsg=side_channel_msg)
        with self.assertRaises(UDEClientError):
            UDEServicerImplementation.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)

        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500),
                                  sideChannelMsg=side_channel_msg)

        with self.assertRaises(UDEClientException):
            UDEServicerImplementation.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)


@patch("socket.socket")
@patch("ude.communication.ude_server.grpc.server")
class UDEServerTest(TestCase):
    def setUp(self) -> None:
        self.socket_mock = patch("socket.socket")
        self._default_options = [('grpc.max_send_message_length',
                                  GRPC_MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length',
                                  GRPC_MAX_MESSAGE_LENGTH)]

    def test_initialize(self, grpc_server_mock, socket_mock):
        ude_env_mock_obj = MagicMock()
        server = UDEServer(ude_env=ude_env_mock_obj)

        assert server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert server.step_invoke_period == 120.0
        assert server.num_agent == 1
        assert server.port == UDE_COMM_DEFAULT_PORT
        assert server.options == self._default_options
        assert server.compression == grpc.Compression.NoCompression
        assert server.credentials is None
        assert server.timeout_wait == 60.0
        assert server.env == ude_env_mock_obj
        assert server.side_channel == ude_env_mock_obj.side_channel
        assert not server.is_open

    def test_initialize_with_param(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        custom_option = [('test', 42)]
        credentials = MagicMock()
        server = UDEServer(ude_env=ude_env_mock,
                           step_invoke_type=UDEStepInvokeType.PERIODIC,
                           step_invoke_period=42.42,
                           num_agents=42,
                           options=custom_option,
                           compression=grpc.Compression.Gzip,
                           credentials=credentials,
                           port=4242,
                           timeout_wait=42.4242)

        assert server.step_invoke_type == UDEStepInvokeType.PERIODIC
        assert server.step_invoke_period == 42.42
        assert server.num_agent == 42
        assert server.port == 4242
        assert server.options == self._default_options + custom_option
        assert server.compression == grpc.Compression.Gzip
        assert server.credentials == credentials
        assert server.timeout_wait == 42.4242
        assert not server.is_open

    def test_initialize_add_insecure_port_error(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        grpc_server_mock.return_value.add_insecure_port.side_effect = Exception("something went wrong")
        with self.assertRaises(UDECommunicationException):
            _ = UDEServer(ude_env=ude_env_mock).start()

    def test_initialize_server_start_error(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        grpc_server_mock.return_value.start.side_effect = Exception("something went wrong")
        with self.assertRaises(UDECommunicationException):
            _ = UDEServer(ude_env=ude_env_mock).start()

    def test_initialize_check_port_error(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        socket_mock.return_value.bind.side_effect = socket.error("something went wrong")
        with self.assertRaises(UDECommunicationException):
            _ = UDEServer(ude_env=ude_env_mock).start()

    def test_start(self, grpc_server_mock, socket_mock):
        ude_env_mock_obj = MagicMock()
        server = UDEServer(ude_env=ude_env_mock_obj)

        assert server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert server.step_invoke_period == 120.0
        assert server.num_agent == 1
        assert server.port == UDE_COMM_DEFAULT_PORT
        assert server.timeout_wait == 60.0
        assert server.env == ude_env_mock_obj
        assert server.side_channel == ude_env_mock_obj.side_channel
        assert server.options == self._default_options
        assert server.compression == grpc.Compression.NoCompression
        assert server.credentials is None
        assert not server.is_open

        ret_server = server.start()
        assert ret_server == server
        assert server.is_open
        grpc_server_mock.assert_called_once_with(ANY,
                                                 options=self._default_options,
                                                 compression=grpc.Compression.NoCompression)
        grpc_server_mock.return_value.add_insecure_port.assert_called_once()
        grpc_server_mock.return_value.start.assert_called_once()

    def test_start_with_compression(self, grpc_server_mock, socket_mock):
        ude_env_mock_obj = MagicMock()
        server = UDEServer(ude_env=ude_env_mock_obj,
                           compression=grpc.Compression.Gzip)

        assert server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert server.step_invoke_period == 120.0
        assert server.num_agent == 1
        assert server.port == UDE_COMM_DEFAULT_PORT
        assert server.timeout_wait == 60.0
        assert server.env == ude_env_mock_obj
        assert server.side_channel == ude_env_mock_obj.side_channel
        assert server.options == self._default_options
        assert server.compression == grpc.Compression.Gzip
        assert server.credentials is None
        assert not server.is_open

        ret_server = server.start()
        assert ret_server == server
        assert server.is_open
        grpc_server_mock.assert_called_once_with(ANY,
                                                 options=self._default_options,
                                                 compression=grpc.Compression.Gzip)
        grpc_server_mock.return_value.add_insecure_port.assert_called_once()
        grpc_server_mock.return_value.start.assert_called_once()

    def test_start_with_options(self, grpc_server_mock, socket_mock):
        ude_env_mock_obj = MagicMock()
        custom_option = [('test', 42)]
        server = UDEServer(ude_env=ude_env_mock_obj,
                           options=custom_option)

        assert server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert server.step_invoke_period == 120.0
        assert server.num_agent == 1
        assert server.port == UDE_COMM_DEFAULT_PORT
        assert server.timeout_wait == 60.0
        assert server.env == ude_env_mock_obj
        assert server.side_channel == ude_env_mock_obj.side_channel
        assert server.options == self._default_options + custom_option
        assert server.compression == grpc.Compression.NoCompression
        assert server.credentials is None
        assert not server.is_open

        ret_server = server.start()
        assert ret_server == server
        assert server.is_open

        expected_options = self._default_options + custom_option
        grpc_server_mock.assert_called_once_with(ANY,
                                                 options=expected_options,
                                                 compression=grpc.Compression.NoCompression)
        grpc_server_mock.return_value.add_insecure_port.assert_called_once()
        grpc_server_mock.return_value.start.assert_called_once()

    def test_start_with_credentials(self, grpc_server_mock, socket_mock):
        ude_env_mock_obj = MagicMock()
        credentials = MagicMock()

        server = UDEServer(ude_env=ude_env_mock_obj,
                           credentials=credentials)

        assert server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert server.step_invoke_period == 120.0
        assert server.num_agent == 1
        assert server.port == UDE_COMM_DEFAULT_PORT
        assert server.timeout_wait == 60.0
        assert server.env == ude_env_mock_obj
        assert server.side_channel == ude_env_mock_obj.side_channel
        assert server.options == self._default_options
        assert server.compression == grpc.Compression.NoCompression
        assert server.credentials == credentials
        assert not server.is_open

        ret_server = server.start()
        assert ret_server == server
        assert server.is_open

        grpc_server_mock.assert_called_once_with(ANY,
                                                 options=self._default_options ,
                                                 compression=grpc.Compression.NoCompression)
        grpc_server_mock.return_value.add_secure_port.assert_called_once()
        grpc_server_mock.return_value.start.assert_called_once()

    def test_check_port_on_linux(self, grpc_server_mock, socket_mock):
        with patch("ude.communication.ude_server.platform", "linux"):
            UDEServer.check_port(port=UDE_COMM_DEFAULT_PORT)

            socket_mock.return_value.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            socket_mock.return_value.bind.assert_called_once()
            socket_mock.return_value.close.assert_called_once()

    def test_check_port_on_linux2(self, grpc_server_mock, socket_mock):
        with patch("ude.communication.ude_server.platform", "linux2"):
            UDEServer.check_port(port=UDE_COMM_DEFAULT_PORT)

            socket_mock.return_value.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            socket_mock.return_value.bind.assert_called_once()
            socket_mock.return_value.close.assert_called_once()

    def test_check_port_on_non_linux(self, grpc_server_mock, socket_mock):
        with patch("ude.communication.ude_server.platform", "darwin"):
            UDEServer.check_port(port=UDE_COMM_DEFAULT_PORT)

            socket_mock.return_value.setsockopt.assert_not_called()
            socket_mock.return_value.bind.assert_called_once()
            socket_mock.return_value.close.assert_called_once()

    def test_on_received_bool(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = False
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               boolVal=UDEBoolDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_int(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = 42
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               intVal=UDEIntDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_float(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = 42.42
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               floatVal=UDEFloatDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_float_list(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = [42.42, 43.43]
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               floatListVal=UDEFloatListDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_string(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = "the answer is 42."
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               stringVal=UDEStringDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_bytes(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = "the answer is 42.".encode("utf-8")
        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               bytesVal=UDEBytesDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_on_received_server_closed(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        key = "key"
        value = "the answer is 42.".encode("utf-8")

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500,
                                                                    message='server closed'),
                                       emptyMsg=UDEEmptyMessageProto())

        ude_server = UDEServer(ude_env=ude_env_mock).start()
        ude_server._is_open = False

        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)
        ude_server.on_received(side_channel=ude_env_mock.return_value.side_channel.return_value,
                               key=key,
                               value=value)

        channel_queue.put.assert_called_once_with(expected_msg)

    def test_invoke_step(self, grpc_server_mock, socket_mock):
        action_dict = {"agent_0": 1}
        ude_env_mock = MagicMock()
        ude_env_mock.step.return_value = "step_info"

        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread
        ude_server._invoke_step_event.set()
        ude_server._action_dict = action_dict
        step_info_ready_event = ude_server._step_info_ready_event
        invoke_step_thread.start()
        step_info_ready_event.wait()
        ude_server.shutdown()

        ude_env_mock.step.assert_called_once_with(action_dict)
        assert ude_server.get_step_info() == "step_info"
        assert ude_server._action_dict == {}

    def test_invoke_step_wait_forever_must_have_period_none(self, grpc_server_mock, socket_mock):
        action_dict = {"agent_0": 1}
        ude_env_mock = MagicMock()
        ude_env_mock.step.return_value = "step_info"
        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER,
                               step_invoke_period=42.42).start()

        assert ude_server.step_invoke_type == UDEStepInvokeType.WAIT_FOREVER
        assert ude_server.step_invoke_period == 42.42
        assert ude_server.is_open

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread
        invoke_step_event_mock = MagicMock()
        ude_server._invoke_step_event = invoke_step_event_mock
        ude_server._action_dict = action_dict
        step_info_ready_event = ude_server._step_info_ready_event
        invoke_step_thread.start()
        step_info_ready_event.wait()
        ude_server.shutdown()

        ude_env_mock.step.assert_called_once_with(action_dict)
        invoke_step_event_mock.wait.assert_called_once_with(timeout=None)
        assert ude_server.get_step_info() == "step_info"
        assert ude_server._action_dict == {}

    def test_invoke_step_should_exit_when_stop_requested(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread
        ude_server._should_stop_invoke_step_thread = True
        invoke_step_thread.start()
        # Expect the thread to exit within 3 seconds.
        # * Thread should exit right away, but giving 3 seconds as maximum time.
        invoke_step_thread.join(3)

    def test_invoke_step_should_exit_when_server_closed(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread
        ude_server._is_open = False
        invoke_step_thread.start()
        # Expect the thread to exit within 3 seconds.
        # * Thread should exit right away, but giving 3 seconds as maximum time.
        invoke_step_thread.join(3)

    def test_invoke_step_should_exit_when_server_closed_while_invoke_step_event(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread

        invoke_step_thread.start()
        time.sleep(2)
        ude_server._is_open = False
        ude_server._invoke_step_event.set()

        invoke_step_thread.join(3)

    def test_invoke_step_should_exit_when_stop_request_while_invoke_step_event(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        ude_server = UDEServer(ude_env=ude_env_mock,
                               step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

        invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
        ude_server._invoke_step_thread = invoke_step_thread

        invoke_step_thread.start()
        time.sleep(2)
        ude_server._should_stop_invoke_step_thread = True
        ude_server._invoke_step_event.set()

        invoke_step_thread.join(3)

    def test_invoke_step_handle_env_fault(self, grpc_server_mock, socket_mock):
        action_dict = {"agent_0": 1}
        ude_env_mock = MagicMock()
        ude_env_mock.step.side_effect = Exception("Some Error")
        with patch("traceback.print_exc") as print_exc_mock:
            ude_server = UDEServer(ude_env=ude_env_mock,
                                   step_invoke_type=UDEStepInvokeType.WAIT_FOREVER).start()

            invoke_step_thread = threading.Thread(target=ude_server._invoke_step)
            ude_server._invoke_step_thread = invoke_step_thread
            ude_server._invoke_step_event.set()
            ude_server._action_dict = action_dict
            step_info_ready_event = ude_server._step_info_ready_event
            invoke_step_thread.start()
            step_info_ready_event.wait()
            ude_server.shutdown()

            ude_env_mock.step.assert_called_once_with(action_dict)
            assert isinstance(ude_server.get_step_info(), UDEEnvException)
            assert ude_server.get_step_info().message == repr(ude_env_mock.step.side_effect)
            assert ude_server._action_dict == {}
            print_exc_mock.assert_called_once()

    def test_step(self, grpc_server_mock, socket_mock):
        action_dict = {"agent_0": 1}
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock).start()

        invoke_step_event_mock = MagicMock()
        ude_server._invoke_step_event = invoke_step_event_mock

        with patch("threading.Thread") as thread_mock:
            ude_server.step(action_dict=action_dict)

            thread_mock.assert_called_once()
            thread_mock.return_value.start.assert_called_once()
            invoke_step_event_mock.set.assert_called_once()

        assert ude_server._action_dict == action_dict

    def test_step_multi_agent(self, grpc_server_mock, socket_mock):
        agent_1_action_dict = {"agent_0": 1}
        agent_2_action_dict = {"agent_2": 2}
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock,
                               num_agents=2).start()

        assert ude_server.num_agent == 2

        invoke_step_event_mock = MagicMock()
        ude_server._invoke_step_event = invoke_step_event_mock

        with patch("threading.Thread") as thread_mock:
            ude_server.step(action_dict=agent_1_action_dict)

            # Thread should be started on first step
            thread_mock.assert_called_once()
            thread_mock.return_value.start.assert_called_once()

            # Not all actions are received by server, so invoke_step_event shouldn't be raised.
            invoke_step_event_mock.set.assert_not_called()

            ude_server.step(action_dict=agent_2_action_dict)

            # Subsequent call to step shouldn't start new thread
            # so the number of thread called should be same after second step call.
            thread_mock.assert_called_once()
            thread_mock.return_value.start.assert_called_once()

            # invoke_step_event should have been raised on second step call.
            invoke_step_event_mock.set.assert_called_once()
        expected_action_dict = {}
        expected_action_dict.update(agent_1_action_dict)
        expected_action_dict.update(agent_2_action_dict)
        assert ude_server._action_dict == expected_action_dict

    def test_reset(self, grpc_server_mock, socket_mock):
        next_obs = "next_obs"
        ude_env_mock = MagicMock()

        ude_env_mock.reset.return_value = next_obs

        ude_server = UDEServer(ude_env=ude_env_mock).start()

        invoke_step_event_mock = MagicMock()
        ude_server._invoke_step_event = invoke_step_event_mock

        ret_val = ude_server.reset()

        assert ret_val == next_obs

        ude_env_mock.reset.assert_called_once()
        invoke_step_event_mock.set.assert_called_once()

        assert ude_server._should_stop_invoke_step_thread
        assert ude_server._invoke_step_thread is None
        assert ude_server._invoke_step_event != invoke_step_event_mock

    def test_reset_with_invoke_step_thread_running(self, grpc_server_mock, socket_mock):
        next_obs = "next_obs"
        ude_env_mock = MagicMock()

        ude_env_mock.reset.return_value = next_obs

        ude_server = UDEServer(ude_env=ude_env_mock).start()

        invoke_step_event_mock = MagicMock()
        invoke_step_thread_mock = MagicMock()
        ude_server._invoke_step_event = invoke_step_event_mock
        ude_server._invoke_step_thread = invoke_step_thread_mock

        ret_val = ude_server.reset()

        assert ret_val == next_obs

        ude_env_mock.reset.assert_called_once()
        invoke_step_event_mock.set.assert_called_once()
        invoke_step_thread_mock.join.assert_called_once()

        assert ude_server._should_stop_invoke_step_thread
        assert ude_server._invoke_step_thread is None
        assert ude_server._invoke_step_event != invoke_step_event_mock

    def test_close(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock)
        ude_server.start()

        ude_server.close()

        ude_env_mock.close.assert_called_once()

        # check shutdown is called
        assert not ude_server.is_open
        ude_env_mock.side_channel.unregister.assert_called_once_with(ude_server)
        grpc_server_mock.return_value.stop.assert_called_once_with(False)

    def test_close_without_start(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock)

        ude_server.close()

        ude_env_mock.side_channel.unregister.assert_not_called()
        grpc_server_mock.return_value.stop.assert_not_called()

    def test_send(self, grpc_server_mock, socket_mock):
        key = "key"
        value = 42

        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock).start()

        ude_server.send(key=key, value=value, store_local=False)
        ude_env_mock.side_channel.send.assert_called_once_with(key=key, value=value,
                                                               store_local=False)

    def test_send_with_store_local(self, grpc_server_mock, socket_mock):
        key = "key"
        value = 42

        ude_env_mock = MagicMock()

        ude_server = UDEServer(ude_env=ude_env_mock).start()

        # Add a channel
        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)

        ude_server.send(key=key, value=value, store_local=True)
        ude_env_mock.side_channel.send.assert_called_once_with(key=key, value=value,
                                                               store_local=True)

        expected_side_channel_msg = UDESideChannelMessageProto(key=key,
                                                               intVal=UDEIntDataProto(val=value))
        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=expected_side_channel_msg)

        # Channel queue should have been called with this key and value.
        channel_queue.put.assert_called_once_with(expected_msg)

    def test_shutdown(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        ude_server = UDEServer(ude_env=ude_env_mock).start()

        ude_env_mock.side_channel.register.assert_called_once_with(ude_server)

        ude_server.shutdown()

        assert not ude_server.is_open
        ude_env_mock.side_channel.unregister.assert_called_once_with(ude_server)
        grpc_server_mock.return_value.stop.assert_called_once_with(False)

    def test_shutdown_clean_up_channel_stream_queues(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        ude_server = UDEServer(ude_env=ude_env_mock).start()

        channel_queue = MagicMock()
        ude_server.add_channel_queue(channel_queue)

        ude_server.shutdown()

        expected_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500),
                                       emptyMsg=UDEEmptyMessageProto())
        channel_queue.put.assert_called_once_with(expected_msg)

    def test_shutdown_clean_up_invoke_step_thread(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        with patch("threading.current_thread") as current_thread_mock:

            ude_server = UDEServer(ude_env=ude_env_mock).start()

            invoke_step_thread_mock = MagicMock()
            ude_server._invoke_step_thread = invoke_step_thread_mock
            current_thread_mock.return_value = "other thread"

            ude_server.shutdown()

            invoke_step_thread_mock.join.assert_called_once()
            assert ude_server._invoke_step_thread is None

    def test_shutdown_no_clean_up_invoke_step_thread_if_from_thread_itself(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        with patch("threading.current_thread") as current_thread_mock:

            ude_server = UDEServer(ude_env=ude_env_mock).start()

            invoke_step_thread_mock = MagicMock()
            ude_server._invoke_step_thread = invoke_step_thread_mock
            current_thread_mock.return_value = invoke_step_thread_mock

            ude_server.shutdown()

            invoke_step_thread_mock.join.assert_not_called()
            assert ude_server._invoke_step_thread is None

    def test_shutdown_server_already_closed(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        with patch("threading.current_thread") as current_thread_mock:

            ude_server = UDEServer(ude_env=ude_env_mock).start()

            ude_env_mock.side_channel.register.assert_called_once_with(ude_server)

            ude_server._is_open = False

            invoke_step_thread_mock = MagicMock()
            ude_server._invoke_step_thread = invoke_step_thread_mock
            current_thread_mock.return_value = invoke_step_thread_mock

            ude_server.shutdown()

            invoke_step_thread_mock.join.assert_not_called()
            ude_env_mock.side_channel.unregister.assert_not_called()
            grpc_server_mock.return_value.stop.assert_not_called()

    def test_spin(self, grpc_server_mock, socket_mock):
        ude_env_mock = MagicMock()
        ude_server = UDEServer(ude_env=ude_env_mock).start()
        ude_server_spin_thread = threading.Thread(target=ude_server.spin)
        ude_server_spin_thread.start()
        # Sleep for 3 seconds
        time.sleep(3)
        # The spin thread should still be alive!
        assert ude_server_spin_thread.is_alive()
        # Set shut down event
        ude_server._shutdown_event.set()
        # The spin thread should exit and join within 3 seconds.
        ude_server_spin_thread.join(3)
