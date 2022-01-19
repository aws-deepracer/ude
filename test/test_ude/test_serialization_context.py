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

from typing import cast

from ude.serialization_context import (
    UDESerializationContext,
    _serialize_gym_space,
    _deserialize_gym_space,
    _serialize_ude_env_exception,
    _deserialize_ude_env_exception
)

from ude.exception import UDEEnvException

from gym.spaces.space import Space
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.tuple import Tuple

import numpy as np


class UDESerializationContextTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_space_serialization(self):
        space = Space(shape=(4, 2), dtype=np.float32)
        serialized_space = _serialize_gym_space(space)
        deserialized_space = cast(Space, _deserialize_gym_space(serialized_space))
        assert space.shape == deserialized_space.shape
        assert space.dtype == deserialized_space.dtype

    def test_box_serialization(self):
        box = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        serialized_box = _serialize_gym_space(box)
        deserialized_box = cast(Box, _deserialize_gym_space(serialized_box))
        assert box == deserialized_box

        box = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        serialized_box = _serialize_gym_space(box)
        deserialized_box = cast(Box, _deserialize_gym_space(serialized_box))
        assert box == deserialized_box

    def test_discrete_serialization(self):
        discrete = Discrete(2)
        serialized_discrete = _serialize_gym_space(discrete)
        deserialized_discrete = cast(Discrete, _deserialize_gym_space(serialized_discrete))
        assert discrete == deserialized_discrete

    def test_multi_binary_serialization(self):
        multi_binary = MultiBinary(2)
        serialized_multi_binary = _serialize_gym_space(multi_binary)
        deserialized_multi_binary = cast(MultiBinary, _deserialize_gym_space(serialized_multi_binary))
        assert multi_binary == deserialized_multi_binary

    def test_multi_discrete_serialization(self):
        multi_discrete = MultiDiscrete([5, 2, 2])
        serialized_multi_discrete = _serialize_gym_space(multi_discrete)
        deserialized_multi_discrete = cast(MultiDiscrete, _deserialize_gym_space(serialized_multi_discrete))
        assert multi_discrete == deserialized_multi_discrete

    def test_dict_serialization(self):
        obs_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        serialized_dict = _serialize_gym_space(obs_space)
        deserialized_dict = cast(Dict, _deserialize_gym_space(serialized_dict))
        assert obs_space == deserialized_dict

    def test_tuple_serialization(self):
        obs_space = Tuple([Discrete(2), Discrete(3)])
        serialized_tuple = _serialize_gym_space(obs_space)
        deserialized_tuple = cast(Tuple, _deserialize_gym_space(serialized_tuple))
        assert obs_space == deserialized_tuple

    def test_ude_env_exception_serialization(self):
        ex = Exception("test")
        ude_exception = UDEEnvException(ex)
        serialized_exception = _serialize_ude_env_exception(ude_exception)
        deserialized_exception = cast(UDEEnvException, _deserialize_ude_env_exception(serialized_exception))
        assert ude_exception.message == deserialized_exception.message

    def test_get_context_singleton(self):
        context = UDESerializationContext.get_context()
        context2 = UDESerializationContext.get_context()
        assert context == context2

    def test_context_serialize(self):
        context = UDESerializationContext.get_context()
        space = Space(shape=(4, 2), dtype=np.float32)
        box = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        discrete = Discrete(2)
        multi_binary = MultiBinary(2)
        multi_discrete = MultiDiscrete([5, 2, 2])
        obs_space_dict = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        obs_space_tuple = Tuple([Discrete(2), Discrete(3)])

        serialized_obj = bytes(context.serialize(space).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(space).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(box).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(box).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(discrete).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(discrete).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(multi_binary).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(multi_binary).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(multi_discrete).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(multi_discrete).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(obs_space_dict).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(obs_space_dict).__name__ == type(deseriazlied_obj).__name__

        serialized_obj = bytes(context.serialize(obs_space_tuple).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(obs_space_tuple).__name__ == type(deseriazlied_obj).__name__

        env_exception = UDEEnvException(Exception("test"))
        serialized_obj = bytes(context.serialize(env_exception).to_buffer())
        deseriazlied_obj = context.deserialize(serialized_obj)
        assert type(env_exception).__name__ == type(deseriazlied_obj).__name__
