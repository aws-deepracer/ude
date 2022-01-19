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
"""UDE Exception classes"""


class UDEException(Exception):
    """
    UDEException class
    """
    def __init__(self, message: str, error_code: int) -> None:
        """
        Initialize UDEException

        Args:
            message (str): message
            error_code (int): error code
        """
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Returns the string representation of the class.

        Returns:
            str: the string representation of the class
        """
        return "{} ({})".format(self.message, self.error_code)


class UDEClientError(UDEException):
    """
    UDEClientError class
    """
    def __init__(self, message: str = '', error_code: int = 400) -> None:
        """
        Initialize UDEClientError

        Args:
            message (str): message
            error_code (int): error code
        """
        super().__init__(error_code=error_code,
                         message=message)


class UDEClientException(UDEException):
    """
    UDEClientException class
    """
    def __init__(self, message: str = '', error_code: int = 500) -> None:
        """
        Initialize UDEClientException

        Args:
            message (str): message
            error_code (int): error code
        """
        super().__init__(error_code=error_code,
                         message=message)


class UDEServerError(UDEException):
    """
    UDEServerError class
    """
    def __init__(self, message: str = '', error_code: int = 400) -> None:
        """
        Initialize UDEServerError

        Args:
            message (str): message
            error_code (int): error code
        """
        super().__init__(error_code=error_code,
                         message=message)


class UDEServerException(UDEException):
    """
    UDEServerException class
    """
    def __init__(self, message: str = '', error_code: int = 500) -> None:
        """
        Initialize UDEServerException

        Args:
            message (str): message
            error_code (int): error code
        """
        super().__init__(error_code=error_code,
                         message=message)


class UDECommunicationException(UDEException):
    """
    UDECommunicationException class
    """
    def __init__(self, port: int, message: str = '', error_code: int = 500) -> None:
        """
        Initialize UDECommunicationException

        Args:
            port (int): port used
            message (str): message
            error_code (int): error code
        """
        self.port = port
        super().__init__(error_code=error_code,
                         message=message)


class UDEEnvException(Exception):
    """
    UDEEnvException class
    """
    def __init__(self, exception: Exception = None) -> None:
        """
        Initialize UDEEnvException

        Args:
            exception (Exception): exception object
        """
        self.message = repr(exception) if exception else ""
        super().__init__()

    def __str__(self):
        """
         Returns the string representation of the class.

         Returns:
             str: the string representation of the class
         """
        return self.message
