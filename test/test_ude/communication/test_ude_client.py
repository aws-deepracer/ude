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
from unittest import mock, TestCase
from unittest.mock import patch, MagicMock

import numpy as np
import grpc
import os

from ude.communication.ude_client import UDEClient, RpcFuncNames
from ude import UDE_COMM_DEFAULT_PORT

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
from ude.ude_objects.ude_empty_message_pb2 import UDEEmptyMessageProto
from ude.ude_objects.ude_data_message_pb2 import (
    UDEDataMessageProto
)

from ude.communication.constants import UDEMessageType
from ude.exception import UDEServerException, UDEServerError, UDEException, UDEEnvException
from ude.serialization_context import UDESerializationContext
from ude.communication.constants import GRPC_MAX_MESSAGE_LENGTH

from gym.spaces.discrete import Discrete


@mock.patch("threading.Thread")
@mock.patch("grpc.insecure_channel")
@mock.patch("ude.ude_objects.ude_pb2_grpc.UDEProtoStub")
class UDEClientTest(TestCase):
    def setUp(self) -> None:
        self._context = UDESerializationContext.get_context()
        self._default_options = [('grpc.max_send_message_length',
                                  GRPC_MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length',
                                  GRPC_MAX_MESSAGE_LENGTH)]

    def test_ude_client_creation(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        client = UDEClient(address)
        # Confirm port is configured with default port if not provided.
        assert client.port == UDE_COMM_DEFAULT_PORT
        assert client.address == address
        assert client.timeout == 10.0
        assert client.max_retry_attempts == 5
        assert client.options == self._default_options
        assert client.compression == grpc.Compression.NoCompression
        assert client.credentials is None

        with patch("grpc.secure_channel") as secure_channel_mock, \
             patch("grpc.composite_channel_credentials") as composite_channel_credential_mock, \
             patch("grpc.ssl_channel_credentials") as ssl_channel_credential_mock:
            custom_option = [('test', 42)]
            credentials = "credential"
            auth_key = "my_pass"
            # Confirm address and port are configured as given.
            client2 = UDEClient(address, port=5005,
                                options=custom_option,
                                compression=grpc.Compression.Gzip,
                                credentials=credentials,
                                auth_key=auth_key,
                                timeout=5.0,
                                max_retry_attempts=10)
            assert client2.port == 5005
            assert client2.address == address
            assert client2.compression == grpc.Compression.Gzip
            assert client2.timeout == 5.0
            assert client2.max_retry_attempts == 10
            assert client2.options == self._default_options + custom_option
            assert client2.credentials == ssl_channel_credential_mock.return_value
            assert client2.auth_key == auth_key

            client2.timeout = 7.0
            client2.max_retry_attempts = 6
            assert client2.timeout == 7.0
            assert client2.max_retry_attempts == 6

    def test_ude_client_creation_with_compression(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        client = UDEClient(address,
                           compression=grpc.Compression.Gzip)

        # Confirm port is configured with default port if not provided.
        assert client.port == UDE_COMM_DEFAULT_PORT
        assert client.address == address
        assert client.timeout == 10.0
        assert client.max_retry_attempts == 5
        assert client.compression == grpc.Compression.Gzip
        assert client.credentials is None

        insecure_channel_mock.assert_called_once_with(address + ":" + str(UDE_COMM_DEFAULT_PORT),
                                                      options=self._default_options,
                                                      compression=grpc.Compression.Gzip)

    def test_ude_client_creation_with_options(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        custom_option = [('test', 42)]

        client = UDEClient(address,
                           options=custom_option)

        expected_options = self._default_options + custom_option

        # Confirm port is configured with default port if not provided.
        assert client.port == UDE_COMM_DEFAULT_PORT
        assert client.address == address
        assert client.timeout == 10.0
        assert client.max_retry_attempts == 5
        assert client.compression == grpc.Compression.NoCompression
        assert client.options == expected_options
        assert client.credentials is None

        insecure_channel_mock.assert_called_once_with(address + ":" + str(UDE_COMM_DEFAULT_PORT),
                                                      options=expected_options,
                                                      compression=grpc.Compression.NoCompression)

    def test_ude_client_creation_with_credentials(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        with patch("grpc.secure_channel") as secure_channel_mock, \
             patch("grpc.composite_channel_credentials") as composite_channel_credential_mock, \
             patch("grpc.ssl_channel_credentials") as ssl_channel_credential_mock:
            address = "localhost"
            credentials = "credential"
            client = UDEClient(address,
                               credentials=credentials)

            # Confirm port is configured with default port if not provided.
            assert client.port == UDE_COMM_DEFAULT_PORT
            assert client.address == address
            assert client.timeout == 10.0
            assert client.max_retry_attempts == 5
            assert client.compression == grpc.Compression.NoCompression
            assert client.options == self._default_options
            assert client.credentials == ssl_channel_credential_mock.return_value

            secure_channel_mock.assert_called_once_with(address + ":" + str(UDE_COMM_DEFAULT_PORT),
                                                        credentials=client.credentials,
                                                        options=self._default_options,
                                                        compression=grpc.Compression.NoCompression)

    def test_ude_client_creation_with_credentials_and_auth_key(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        with patch("grpc.secure_channel") as secure_channel_mock, \
             patch("grpc.composite_channel_credentials") as composite_channel_credential_mock, \
             patch("grpc.metadata_call_credentials") as metadata_call_credentials_mock, \
             patch("grpc.ssl_channel_credentials") as ssl_channel_credentials_mock, \
             patch("ude.communication.ude_client.GrpcAuth") as grpc_auth_mock:
            address = "localhost"
            credentials = "credential"
            auth_key = "my_pass"
            client = UDEClient(address,
                               credentials=credentials,
                               auth_key=auth_key)

            # Confirm port is configured with default port if not provided.
            assert client.port == UDE_COMM_DEFAULT_PORT
            assert client.address == address
            assert client.timeout == 10.0
            assert client.max_retry_attempts == 5
            assert client.compression == grpc.Compression.NoCompression
            assert client.options == self._default_options
            assert client.credentials == ssl_channel_credentials_mock.return_value
            assert client.auth_key == auth_key

            assert not isinstance(credentials, grpc.ChannelCredentials)


            grpc_auth_mock.assert_called_once_with(key=auth_key)
            metadata_call_credentials_mock.assert_called_once_with(grpc_auth_mock.return_value)
            composite_channel_credential_mock.assert_called_once_with(client.credentials,
                                                                      metadata_call_credentials_mock.return_value)
            secure_channel_mock.assert_called_once_with(address + ":" + str(UDE_COMM_DEFAULT_PORT),
                                                        credentials=composite_channel_credential_mock.return_value,
                                                        options=self._default_options,
                                                        compression=grpc.Compression.NoCompression)
            ssl_channel_credentials_mock.assert_called_once_with(credentials)

    def test_to_channel_credentials_file_path(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        with patch("grpc.ssl_channel_credentials") as ssl_channel_credentials_mock, \
                patch("builtins.open") as open_mock, \
                patch("os.path.isfile") as is_file_mock:
            credentials = "/file_path"
            is_file_mock.return_value = True
            channel_credential = UDEClient.to_channel_credentials(credentials)
            open_mock.assert_called_once_with(credentials, 'rb')
            open_mock.return_value.__enter__.return_value.read.assert_called_once()
            file_data_mock = open_mock.return_value.__enter__.return_value.read.return_value
            ssl_channel_credentials_mock.assert_called_once_with(file_data_mock)
            assert channel_credential == ssl_channel_credentials_mock.return_value

    def test_to_channel_credentials_bytes(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        with patch("grpc.ssl_channel_credentials") as ssl_channel_credentials_mock, \
             patch("builtins.open") as open_mock, \
             patch("os.path.isfile") as is_file_mock:
            credentials = "bytes_array"
            is_file_mock.return_value = False
            channel_credential = UDEClient.to_channel_credentials(credentials)
            open_mock.assert_not_called()
            open_mock.return_value.read.assert_not_called()
            ssl_channel_credentials_mock.assert_called_once_with(credentials)
            assert channel_credential == ssl_channel_credentials_mock.return_value

    def test_to_channel_credentials_channel_credentials(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        with patch("grpc.ssl_channel_credentials") as ssl_channel_credentials_mock, \
             patch("builtins.open") as open_mock, \
             patch("os.path.isfile") as is_file_mock:
            credentials = grpc.ChannelCredentials(MagicMock())
            is_file_mock.return_value = False
            channel_credential = UDEClient.to_channel_credentials(credentials)

            open_mock.assert_not_called()

            assert credentials == channel_credential

    def test_on_message_received_bool(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):

        ude_message1 = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=UDESideChannelMessageProto(key="key1",
                                                                                 boolVal=UDEBoolDataProto(val=True),
                                                                                 store_local=True)
                                       )

        ude_message2 = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                       sideChannelMsg=UDESideChannelMessageProto(key="key2",
                                                                                 boolVal=UDEBoolDataProto(val=False),
                                                                                 store_local=True)
                                       )
        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message1, ude_message2]
        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called()
        assert client.get("key1")
        assert not client.get("key2")

    def test_on_message_received_int(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                                intVal=UDEIntDataProto(val=42),
                                                                                store_local=True)
                                      )

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        assert client.get("key") == 42

    def test_on_message_received_float(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                                floatVal=UDEFloatDataProto(val=42.42),
                                                                                store_local=True)
                                      )

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        assert np.isclose(client.get("key"), 42.42)

    def test_on_message_received_float_list(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        float_list_val = [42.42, 43.43]
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatListVal=UDEFloatListDataProto(val=float_list_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        for a, b in zip(client.get("key"), float_list_val):
            assert np.isclose(a, b)

    def test_on_message_received_string(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        assert client.get("key") == string_val

    def test_on_message_received_bytes(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        bytes_val = "the answer is 42.".encode('utf-8')
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      bytesVal=UDEBytesDataProto(val=bytes_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        assert client.get("key") == bytes_val

    def test_on_message_received_wrong_message(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        bytes_val = "the answer is 42.".encode('utf-8')
        wrong_type_msg = UDEDataMessageProto(data=bytes_val)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      dataMsg=wrong_type_msg)

        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        client._should_stop_receiver_thread = True
        channel_mock = client._channel
        with self.assertRaises(UDEException):
            client.on_message_received()

        ude_proto_stub_mock.return_value.side_channel_stream.assert_called_once()
        channel_mock.close.assert_called_once()

    def test_on_message_observer_any_key(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address, timeout=1.0, max_retry_attempts=0)
        with MagicMock() as mock:
            client.register(mock)
            client._should_stop_receiver_thread = True
            client.on_message_received()
            mock.on_received.assert_called_once_with(side_channel=client, key="key", value=string_val)

    def test_on_message_observer_specific_key(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        with MagicMock() as mock:
            client.register(observer=mock, key="key")
        client._should_stop_receiver_thread = True
        client.on_message_received()
        mock.on_received.assert_called_once_with(side_channel=client, key="key", value=string_val)

    def test_on_message_specific_key_observer_for_other_key(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.side_channel_stream.return_value = [ude_message]

        address = "localhost"
        client = UDEClient(address)
        with MagicMock() as mock:
            client.register(observer=mock, key="key2")
        client._should_stop_receiver_thread = True
        client.on_message_received()
        # message's key is "key", and observer is registered with "key2"
        # So, on_received shouldn't be called.
        mock.on_received.assert_not_called()

    def test_send_bool(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                                boolVal=UDEBoolDataProto(val=True))
                                      )

        address = "localhost"
        client = UDEClient(address)

        client.send(key="key", value=True)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_int(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                                intVal=UDEIntDataProto(val=42))
                                      )

        address = "localhost"
        client = UDEClient(address)

        client.send("key", 42)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_float(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=UDESideChannelMessageProto(key="key",
                                                                                floatVal=UDEFloatDataProto(val=42.42))
                                      )

        address = "localhost"
        client = UDEClient(address)

        client.send("key", 42.42)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_float_list(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        float_list_val = [42.42, 43.43]
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatListVal=UDEFloatListDataProto(val=float_list_val))
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        address = "localhost"
        client = UDEClient(address)

        client.send("key", float_list_val)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_string(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val))
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        address = "localhost"
        client = UDEClient(address)

        client.send("key", string_val)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_bytes(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        bytes_val = "the answer is 42.".encode('utf-8')
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      bytesVal=UDEBytesDataProto(val=bytes_val))
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        address = "localhost"
        client = UDEClient(address)

        client.send("key", bytes_val)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

    def test_send_wrong_date_type(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        wrong_data = {"hello": "hi"}
        address = "localhost"
        client = UDEClient(address)

        with self.assertRaises(TypeError):
            client.send("key", wrong_data)

    def test_send_store_local(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=True)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        address = "localhost"
        client = UDEClient(address)

        client.send("key", string_val, store_local=True)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

        assert client.get("key") == string_val

    def test_send_no_store_local(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        string_val = "the answer is 42."
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      stringVal=UDEStringDataProto(val=string_val),
                                                      store_local=False)
        ude_message = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                      sideChannelMsg=side_channel_msg)

        address = "localhost"
        client = UDEClient(address)

        client.send("key", string_val, store_local=False)
        ude_proto_stub_mock.return_value.side_channel_send.assert_called_once_with(ude_message,
                                                                                   timeout=client.timeout)

        assert client.get("key") is None

    def test_step(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        next_state = {"agent": "next_state"}
        done = {"agent": False}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        serialized_obj = bytes(self._context.serialize(step_dict).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.step.return_value = step_msg

        address = "localhost"
        client = UDEClient(address)
        ret_next_state, ret_reward, ret_done, ret_last_action, ret_info = client.step(action_dict=action_dict)

        ude_proto_stub_mock.return_value.step.assert_called_once_with(action_msg,
                                                                      timeout=client.timeout)
        assert ret_next_state == next_state
        assert ret_done == done
        assert ret_reward == reward
        assert ret_last_action == action_dict
        assert ret_info == info

    def test_step_env_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        action_dict = {"agent": 1}
        serialized_obj = bytes(self._context.serialize(action_dict).to_buffer())
        action_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                     dataMsg=UDEDataMessageProto(data=serialized_obj))

        exception = Exception("Some Error")
        env_err = UDEEnvException(exception)
        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.step.return_value = step_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEEnvException):
            _ = client.step(action_dict=action_dict)

        ude_proto_stub_mock.return_value.step.assert_called_once_with(action_msg,
                                                                      timeout=client.timeout)

    def test_step_wrong_action_dict_type(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # action_dict must be dict type.
        action_dict = 10

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(ValueError):
            _ = client.step(action_dict=action_dict)

    def test_step_wrong_msg_from_server(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # Server sends SideChannel msg when Step msg is expected.
        action_dict = {"agent": 1}

        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.step.return_value = ude_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEServerException):
            _ = client.step(action_dict=action_dict)

    def test_reset(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        next_state = {"agent": "next_state"}
        serialized_obj = bytes(self._context.serialize(next_state).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.reset.return_value = step_msg

        address = "localhost"
        client = UDEClient(address)
        ret_next_state = client.reset()

        ude_proto_stub_mock.return_value.reset.assert_called_once()
        assert ret_next_state == next_state

    def test_reset_env_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        exception = Exception("Some Error")
        env_err = UDEEnvException(exception)

        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.reset.return_value = step_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEEnvException):
            _ = client.reset()

        ude_proto_stub_mock.return_value.reset.assert_called_once()

    def test_reset_wrong_msg_from_server(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # Server sends SideChannel msg when Step msg is expected.
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.reset.return_value = ude_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEServerException):
            _ = client.reset()

    def test_close(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())

        address = "localhost"
        client = UDEClient(address)
        client.close()

        ude_proto_stub_mock.return_value.close.assert_called_once_with(msg,
                                                                       timeout=client.timeout)

        insecure_channel_mock.return_value.close.assert_called_once()

    def test_close_client_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):

        ude_proto_stub_mock.return_value.close.side_effect = Exception()

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(Exception):
            client.close()
        insecure_channel_mock.return_value.close.assert_called_once()
        assert client._channel is None
        assert client._conn is None
        assert client._receiver_thread is None

    def test_observation_space(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        observation_space = {"agent": Discrete(42)}
        serialized_obj = bytes(self._context.serialize(observation_space).to_buffer())
        observation_space_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                                dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.observation_space.return_value = observation_space_msg

        address = "localhost"
        client = UDEClient(address)
        ret_observation_space = client.observation_space

        assert ret_observation_space == observation_space

    def test_observation_space_env_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        exception = Exception("Some Error")
        env_err = UDEEnvException(exception)
        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        observation_space_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                                dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.observation_space.return_value = observation_space_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEEnvException):
            _ = client.observation_space

    def test_observation_space_wrong_msg_from_server(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # Server sends SideChannel msg when Step msg is expected.
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.observation_space.return_value = ude_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEServerException):
            _ = client.observation_space

    def test_action_space(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        action_space = {"agent": Discrete(42)}
        serialized_obj = bytes(self._context.serialize(action_space).to_buffer())
        action_space_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                           dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.action_space.return_value = action_space_msg

        address = "localhost"
        client = UDEClient(address)
        ret_action_space = client.action_space

        assert ret_action_space["agent"].shape == action_space["agent"].shape
        assert ret_action_space["agent"].dtype == action_space["agent"].dtype

    def test_action_space_env_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        exception = Exception("Some Error")
        env_err = UDEEnvException(exception)
        serialized_obj = bytes(self._context.serialize(env_err).to_buffer())
        action_space_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                           dataMsg=UDEDataMessageProto(data=serialized_obj))
        ude_proto_stub_mock.return_value.action_space.return_value = action_space_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEEnvException):
            _ = client.action_space

    def test_action_space_wrong_msg_from_server(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # Server sends SideChannel msg when Step msg is expected.
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        ude_proto_stub_mock.return_value.action_space.return_value = ude_msg

        address = "localhost"
        client = UDEClient(address)
        with self.assertRaises(UDEServerException):
            _ = client.action_space

    def test_validate_msg_empty(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        assert UDEClient.validate_msg(msg, UDEMessageType.EMPTY)

    def test_validate_msg_expected_type_string(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                              emptyMsg=UDEEmptyMessageProto())
        assert UDEClient.validate_msg(msg, "emptyMsg")

    def test_validate_msg_step(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        next_state = {"agent": "next_state"}
        serialized_obj = bytes(self._context.serialize(next_state).to_buffer())
        step_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                   dataMsg=UDEDataMessageProto(data=serialized_obj))
        assert UDEClient.validate_msg(step_msg, UDEMessageType.DATA)

    def test_validate_msg_side_channel(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        assert UDEClient.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)

    def test_validate_msg_unexpected_msg_type(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        # Server sends SideChannel message when expected Step msg.
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=200),
                                  sideChannelMsg=side_channel_msg)
        with self.assertRaises(UDEServerException):
            UDEClient.validate_msg(ude_msg, UDEMessageType.DATA)

    def test_validate_msg_server_fault(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=400),
                                  sideChannelMsg=side_channel_msg)
        with self.assertRaises(UDEServerError):
            UDEClient.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)

        side_channel_msg = UDESideChannelMessageProto(key="key",
                                                      floatVal=UDEFloatDataProto(val=42.42))
        ude_msg = UDEMessageProto(header=UDEMessageHeaderProto(status=500),
                                  sideChannelMsg=side_channel_msg)

        with self.assertRaises(UDEServerException):
            UDEClient.validate_msg(ude_msg, UDEMessageType.SIDE_CHANNEL)

    def test_connect(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        client = UDEClient(address)

        insecure_channel_mock.assert_called_once()
        ude_proto_stub_mock.assert_called_once()
        assert not client._should_stop_receiver_thread
        thread_mock.assert_called_once()
        thread_mock.return_value.start.assert_called_once()

    def test_reset_channel(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        client = UDEClient(address)

        client._reset_channel()
        insecure_channel_mock.return_value.close.assert_called_once()
        assert insecure_channel_mock.call_count == 2
        assert ude_proto_stub_mock.call_count == 2

    def test_call_with_retry(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = 'ping'
        expected_response = 'pong'
        rpc_func = MagicMock()
        rpc_func.return_value = expected_response
        ude_proto_stub_mock.return_value.step = rpc_func

        address = "localhost"
        client = UDEClient(address)

        response = client._call_with_retry(rpc_func_name=RpcFuncNames.STEP,
                                           msg=msg,
                                           timeout=0.5,
                                           max_retry_attempts=3)
        assert expected_response == response
        rpc_func.assert_called_once_with(msg, timeout=0.5)

    def test_internal_close(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        address = "localhost"
        client = UDEClient(address)

        client._close()
        assert client._channel is None
        assert client._conn is None
        assert client._receiver_thread is None
        assert client._should_stop_receiver_thread
        insecure_channel_mock.return_value.close.assert_called_once()
        thread_mock.return_value.join.assert_called_once()

    def test_call_with_retry_failed_with_rpc_error(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = 'ping'
        rpc_func = MagicMock()

        e = grpc.RpcError()
        e.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_func.side_effect = e
        ude_proto_stub_mock.return_value.step = rpc_func

        address = "localhost"
        client = UDEClient(address)

        with self.assertRaises(grpc.RpcError):
            _ = client._call_with_retry(rpc_func_name=RpcFuncNames.STEP,
                                        msg=msg,
                                        timeout=0.5,
                                        max_retry_attempts=3)
            assert rpc_func.call_count == 6
            assert insecure_channel_mock.return_value.close.call_count == 6
            assert insecure_channel_mock.call_count == 7

    def test_call_with_retry_failed_with_exception(self, ude_proto_stub_mock, insecure_channel_mock, thread_mock):
        msg = 'ping'
        rpc_func = MagicMock()
        rpc_func.side_effect = Exception()
        ude_proto_stub_mock.return_value.step = rpc_func

        address = "localhost"
        client = UDEClient(address)

        with self.assertRaises(Exception):
            _ = client._call_with_retry(rpc_func_name=RpcFuncNames.STEP,
                                        msg=msg,
                                        timeout=0.5,
                                        max_retry_attempts=3)
            rpc_func.assert_called_once_with(msg, timeout=0.5)

