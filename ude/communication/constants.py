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
"""Module to contain communication related constants"""
from ude.ude_objects.ude_side_channel_message_pb2 import (
    UDEBoolDataProto, UDEIntDataProto,
    UDEFloatDataProto, UDEFloatListDataProto,
    UDEStringDataProto, UDEBytesDataProto
)
from enum import Enum


"""
builtin type to UDE Data Proto message.
"""
BUILTIN_TYPE_TO_SIDE_CHANNEL_DATA_MSG = {bool: UDEBoolDataProto,
                                         int: UDEIntDataProto,
                                         float: UDEFloatDataProto,
                                         list: UDEFloatListDataProto,
                                         str: UDEStringDataProto,
                                         bytes: UDEBytesDataProto}


"""
builtin type to Side Channel message arg.
"""
BUILTIN_TYPE_TO_SIDE_CHANNEL_MSG_ARG = {bool: "boolVal",
                                        int: "intVal",
                                        float: "floatVal",
                                        list: "floatListVal",
                                        str: "stringVal",
                                        bytes: "bytesVal"}


class UDEMessageType(Enum):
    """
    UDE Message Type
    """
    DATA = "dataMsg"
    EMPTY = "emptyMsg"
    SIDE_CHANNEL = "sideChannelMsg"


"""UDE Default communication port"""
UDE_COMM_DEFAULT_PORT = 3003


class UDEStepInvokeType(Enum):
    """
    UDE Step Invoke Type
    """
    WAIT_FOREVER = 'wait_forever'
    PERIODIC = 'periodic'


# GRPC Max Message Length
GRPC_MAX_MESSAGE_LENGTH = -1  # unlimited
