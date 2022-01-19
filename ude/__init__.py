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
"""UDE modules"""
from .communication.constants import UDEStepInvokeType, UDE_COMM_DEFAULT_PORT
from .communication.ude_server import UDEServer
from .communication.ude_client import UDEClient

from .environment.adapters.remote_environment_adapter import RemoteEnvironmentAdapter
from .environment.interfaces import UDEEnvironmentInterface, UDEEnvironmentAdapterInterface
from .environment.ude_environment import UDEEnvironment
from .environment.constants import UDEResetMode

from .side_channels.adapters.dummy_side_channel import DummySideChannel
from .side_channels.adapters.single_side_channel import SingleSideChannel
from .side_channels.constants import SideChannelDataType, BUILTIN_TYPE_TO_SIDE_CHANNEL_DATATYPE
from .side_channels.ude_side_channel import AbstractSideChannel, SideChannelObserverInterface

from .serialization_context import UDESerializationContext
from .ude_typing import *
from .exception import *

from .gym_ude import UDEToGymWrapper

from gym.spaces.space import Space
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.tuple import Tuple
from gym.spaces.utils import *

from grpc import Compression
from grpc import ChannelCredentials
from grpc import ServerCredentials
