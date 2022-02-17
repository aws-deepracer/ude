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

from ude.communication.grpc_auth import GrpcAuth


class GrpcAuthTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize(self):
        key = "hello"
        auth = GrpcAuth(key=key)
        assert key == auth._key

    def test_call(self):
        key = "hello"
        auth = GrpcAuth(key=key)
        context_mock = MagicMock()
        callback_mock = MagicMock()
        auth(context=context_mock,
             callback=callback_mock)
        callback_mock.assert_called_with((('rpc-auth-header', key),), None)
