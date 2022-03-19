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
"""A class to control custom classes serialization"""
import threading
from typing import Union
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # https://ray-project.github.io/2017/10/15/fast-python-serialization-with-ray-and-arrow.html
    import pyarrow

from gym.spaces.space import Space
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.tuple import Tuple
from ude.exception import UDEEnvException


def _serialize_gym_space(val: Space) -> Union[dict, Space]:
    """
    Convert gym.Space to serializable dictionary.

    Args:
        val (Space): gym.Space to convert.

    Returns:
        Union[dict, Space]: dictionary converted from gym.Space.
                            (Returns argument if not supported gym.Space class)
    """
    if isinstance(val, Box):
        return {'class': type(val).__name__,
                'low': val.low,
                'high': val.high,
                'dtype': None if val.dtype is None else str(val.dtype)}
    elif isinstance(val, Discrete):
        return {'class': type(val).__name__,
                'n': val.n}
    elif isinstance(val, MultiBinary):
        return {'class': type(val).__name__,
                'n': val.n}
    elif isinstance(val, MultiDiscrete):
        return {'class': type(val).__name__,
                'nvec': val.nvec}
    elif isinstance(val, Dict):
        return {'class': type(val).__name__,
                'spaces': {key: _serialize_gym_space(item) for key, item in val.spaces.items()}}
    elif isinstance(val, Tuple):
        return {'class': type(val).__name__,
                'spaces': [_serialize_gym_space(item) for item in val.spaces]}
    elif isinstance(val, Space):
        return {'class': type(val).__name__,
                'shape': val.shape,
                'dtype': None if val.dtype is None else str(val.dtype)}
    else:
        return val


def _deserialize_gym_space(data: dict) -> Union[Space, dict]:
    """
    Convert to gym.Space class from dictionary.

    Args:
        data (dict): dictionary to convert back to gym.Space

    Returns:
        Union[Space, dict]: gym.Space converted from dictionary
                            (Returns argument if not supported gym.Space class)
    """
    if data['class'] == Box.__name__:
        return Box(low=data['low'], high=data['high'], dtype=data['dtype'])
    elif data['class'] == Discrete.__name__:
        return Discrete(n=data['n'])
    elif data['class'] == MultiBinary.__name__:
        return MultiBinary(n=data['n'])
    elif data['class'] == MultiDiscrete.__name__:
        return MultiDiscrete(nvec=data['nvec'])
    elif data['class'] == Dict.__name__:
        spaces = {key: _deserialize_gym_space(item) for key, item in data['spaces'].items()}
        return Dict(spaces=spaces)
    elif data['class'] == Tuple.__name__:
        spaces = [_deserialize_gym_space(item) for item in data['spaces']]
        return Tuple(spaces=spaces)
    elif data['class'] == Space.__name__:
        return Space(shape=data['shape'], dtype=data['dtype'])
    else:
        return data


def _serialize_ude_env_exception(val: UDEEnvException) -> dict:
    return {'message': val.message}


def _deserialize_ude_env_exception(data: dict) -> UDEEnvException:
    ex = UDEEnvException()
    ex.message = data['message']
    return ex


class UDESerializationContext(object):
    """
    UDESerializationContext class
    """
    _context = None
    _context_lock = threading.RLock()

    @staticmethod
    def get_context() -> pyarrow.SerializationContext:
        """
        Returns the serialization context instance.

        Returns:
            pyarrow.SerializationContext: the serialization context instance
        """
        with UDESerializationContext._context_lock:
            if UDESerializationContext._context is None:
                context = pyarrow.SerializationContext()
                context = UDESerializationContext._register(context=context)
                UDESerializationContext._context = context
            return UDESerializationContext._context

    @staticmethod
    def _register(context: pyarrow.SerializationContext) -> pyarrow.SerializationContext:
        """
        Register serializer and deserializer handler for gym.Space.

        Args:
            context (pyarrow.SerializationContext): serialization context.

        Returns:
            pyarrow.SerializationContext: serialization context.
        """
        # Register gym.Space serializer
        context.register_type(Space, Space.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(Box, Box.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(Dict, Dict.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(Discrete, Discrete.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(MultiBinary, MultiBinary.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(MultiDiscrete, MultiDiscrete.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)
        context.register_type(Tuple, Tuple.__name__,
                              custom_serializer=_serialize_gym_space,
                              custom_deserializer=_deserialize_gym_space)

        # Register Exception serializer
        context.register_type(UDEEnvException, UDEEnvException.__name__,
                              custom_serializer=_serialize_ude_env_exception,
                              custom_deserializer=_deserialize_ude_env_exception)

        return context
