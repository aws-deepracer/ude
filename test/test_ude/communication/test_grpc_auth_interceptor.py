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

from ude.communication.grpc_auth_interceptor import AuthInterceptor


class AuthInterceptorTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize(self):
        key = "hello"
        interceptor = AuthInterceptor(key=key)
        assert interceptor._valid_metadata == ('rpc-auth-header', key)

    def test_call_valid(self):
        key = "hello"
        interceptor = AuthInterceptor(key=key)
        continuation_mock = MagicMock()
        handler_call_details_mock = MagicMock()
        handler_call_details_mock.invocation_metadata = [('rpc-auth-header', key)]
        handler = interceptor.intercept_service(continuation=continuation_mock,
                                                handler_call_details=handler_call_details_mock)
        continuation_mock.assert_called_once_with(handler_call_details_mock)
        assert handler == continuation_mock.return_value

    def test_call_invalid(self):
        key = "hello"
        interceptor = AuthInterceptor(key=key)
        continuation_mock = MagicMock()
        handler_call_details_mock = MagicMock()
        handler_call_details_mock.invocation_metadata = []
        handler = interceptor.intercept_service(continuation=continuation_mock,
                                                handler_call_details=handler_call_details_mock)
        assert handler == interceptor._deny


