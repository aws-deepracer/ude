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
"""A class for GRPC custom authentication interceptor with key."""
from typing import Any
import grpc


class AuthInterceptor(grpc.ServerInterceptor):
    """
    GRPC custom authentication interceptor with authentication key.
    """
    def __init__(self, key: str) -> None:
        """
        Initialize GRPC custom authentication interceptor.
        Args:
            key (str): authentication key.
        """
        self._valid_metadata = ('rpc-auth-header', key)

        def deny(_, context: Any) -> None:
            """
            deny callback.

            Args:
                _:
                context (Any): callback context.
            """
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Invalid key')

        self._deny = grpc.unary_unary_rpc_method_handler(deny)

    def intercept_service(self, continuation: Any, handler_call_details: Any) -> grpc.RpcMethodHandler:
        """
        Intercepts incoming RPCs before handing them over to a handler.

        Args:
            continuation (Any): A function that takes a HandlerCallDetails and
              proceeds to invoke the next interceptor in the chain, if any,
              or the RPC handler lookup logic, with the call details passed
              as an argument, and returns an RpcMethodHandler instance if
              the RPC is considered serviced, or None otherwise.
            handler_call_details (Any): A HandlerCallDetails describing the RPC.

        Returns:
          An RpcMethodHandler with which the RPC may be serviced if the
          interceptor chooses to service this RPC, or None otherwise.
        """
        meta = handler_call_details.invocation_metadata

        if meta and meta[0] == self._valid_metadata:
            return continuation(handler_call_details)
        else:
            return self._deny
