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
from unittest.mock import patch, MagicMock

from ude.environment.adapters.remote_environment_adapter import RemoteEnvironmentAdapter

from gym.spaces.space import Space
import grpc


@patch("ude.environment.adapters.remote_environment_adapter.UDEClient")
class RemoteEnvironmentAdapterTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize(self, ude_client_mock):
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address)
        ude_client_mock.assert_called_once_with(address=address, port=None,
                                                options=None,
                                                compression=grpc.Compression.NoCompression,
                                                credentials=None,
                                                timeout=10.0,
                                                max_retry_attempts=5)

    def test_initialize_with_port(self, ude_client_mock):
        address = "localhost"
        port = 5005
        adapter = RemoteEnvironmentAdapter(address=address,
                                           port=port,
                                           timeout=5.0,
                                           max_retry_attempts=4)
        ude_client_mock.assert_called_once_with(address=address, port=port,
                                                options=None,
                                                compression=grpc.Compression.NoCompression,
                                                credentials=None,
                                                timeout=5.0,
                                                max_retry_attempts=4)

    def test_initialize_with_compression(self, ude_client_mock):
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address,
                                           compression=grpc.Compression.Gzip)

        ude_client_mock.assert_called_once_with(address=address, port=None,
                                                options=None,
                                                compression=grpc.Compression.Gzip,
                                                credentials=None,
                                                timeout=10.0,
                                                max_retry_attempts=5)

    def test_initialize_with_options(self, ude_client_mock):
        address = "localhost"
        custom_option = [('test', 42)]
        adapter = RemoteEnvironmentAdapter(address=address,
                                           options=custom_option)
        ude_client_mock.assert_called_once_with(address=address, port=None,
                                                options=custom_option,
                                                compression=grpc.Compression.NoCompression,
                                                credentials=None,
                                                timeout=10.0,
                                                max_retry_attempts=5)

    def test_initialize_with_credentials(self, ude_client_mock):
        address = "localhost"
        credentials = MagicMock()
        adapter = RemoteEnvironmentAdapter(address=address,
                                           credentials=credentials)
        ude_client_mock.assert_called_once_with(address=address, port=None,
                                                options=None,
                                                compression=grpc.Compression.NoCompression,
                                                credentials=credentials,
                                                timeout=10.0,
                                                max_retry_attempts=5)

    def test_step(self, ude_client_mock):
        action_dict = {"agent": 1}
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address)
        adapter.step(action_dict)
        ude_client_mock.return_value.step.assert_called_once_with(action_dict=action_dict)

    def test_reset(self, ude_client_mock):
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address)
        adapter.reset()
        ude_client_mock.return_value.reset.assert_called_once()

    def test_close(self, ude_client_mock):
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address)
        adapter.close()
        ude_client_mock.return_value.close.assert_called_once()

    def test_observation_space(self, ude_client_mock):
        address = "localhost"

        expected_observation_space = Space([4, 2])
        ude_client_mock.return_value.observation_space = expected_observation_space
        adapter = RemoteEnvironmentAdapter(address=address)
        ret_observation_space = adapter.observation_space
        assert ret_observation_space == expected_observation_space

    def test_action_space(self, ude_client_mock):
        address = "localhost"
        expected_action_space = Space([4, 2])
        ude_client_mock.return_value.action_space = expected_action_space
        adapter = RemoteEnvironmentAdapter(address=address)
        ret_action_space = adapter.action_space
        assert ret_action_space == expected_action_space

    def test_side_channel(self, ude_client_mock):
        address = "localhost"
        adapter = RemoteEnvironmentAdapter(address=address)
        ret_side_channel = adapter.side_channel
        assert ret_side_channel == ude_client_mock.return_value
