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
"""A class for GRPC custom authentication with key."""
from typing import Any
import grpc


class GrpcAuth(grpc.AuthMetadataPlugin):
    """
    GRPC custom authentication with authentication key.
    """
    def __init__(self, key: str) -> None:
        """
        Initialize GRPC custom authentication.

        Args:
            key (str): authentication key.
        """
        self._key = key

    def __call__(self, context: Any, callback: Any) -> None:
        """
        Callback.

        Args:
            context (Any): callback context.
            callback (Any): callback function pointer.
        """
        callback((('rpc-auth-header', self._key),), None)
