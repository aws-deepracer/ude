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

from ude.side_channels.adapters.single_side_channel import SingleSideChannel


class SingleSideChannelTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize(self):
        side_channel = SingleSideChannel()

    def test_send(self):
        side_channel = SingleSideChannel()
        observer_mock = MagicMock()
        side_channel.register(observer_mock)
        side_channel.send(key="key", value=10, store_local=True)
        observer_mock.on_received.assert_called_once_with(side_channel=side_channel,
                                                          key="key",
                                                          value=10)
