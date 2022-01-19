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

from ude.gym_ude import UDEToGymWrapper
from ude.environment.constants import UDEResetMode
from gym.spaces.space import Space


class UDEToGymWrapperTest(TestCase):
    def setUp(self) -> None:
        self._ude_env_mock = MagicMock()
        observation_space = Space([4, 2])
        observation_space_dict = {"agent_0": observation_space}
        self._ude_env_mock.observation_space = observation_space_dict

    def test_creation(self):
        UDEToGymWrapper(ude_env=self._ude_env_mock)
        assert self._ude_env_mock.reset_mode == UDEResetMode.MANUAL

    def test_step(self):
        action = 1

        mock_obs = "next_state"
        mock_reward = 42
        mock_done = False

        step_dict = ({"agent_0": mock_obs},
                     {"agent_0": mock_reward},
                     {"agent_0": mock_done},
                     {"agent_0": action},
                     {})
        self._ude_env_mock.step.return_value = step_dict

        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)

        obs, reward, done, info = ude_gym_wrapper.step(action)
        assert obs == mock_obs
        assert reward == mock_reward
        assert done == mock_done
        assert info == {}

    def test_step_done(self):
        action = 1

        mock_obs = "next_state"
        mock_reward = 42
        mock_done = True

        step_dict = ({"agent_0": mock_obs},
                     {"agent_0": mock_reward},
                     {"agent_0": mock_done},
                     {"agent_0": action},
                     {})
        self._ude_env_mock.step.return_value = step_dict

        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)

        obs, reward, done, info = ude_gym_wrapper.step(action)
        assert obs == mock_obs
        assert reward == mock_reward
        assert done == mock_done
        assert info == {}
        assert ude_gym_wrapper.game_over

    def test_reset(self):
        action = 1

        mock_obs = "next_state"
        mock_reset_obs = "new_episode_state"
        mock_reward = 42
        mock_done = True

        step_dict = ({"agent_0": mock_obs},
                     {"agent_0": mock_reward},
                     {"agent_0": mock_done},
                     {"agent_0": action},
                     {})
        reset_dict = {"agent_0": mock_reset_obs}
        self._ude_env_mock.step.return_value = step_dict
        self._ude_env_mock.reset.return_value = reset_dict

        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)

        obs, reward, done, info = ude_gym_wrapper.step(action)
        assert obs == mock_obs
        assert reward == mock_reward
        assert done == mock_done
        assert info == {}
        assert ude_gym_wrapper.game_over

        new_episode_obs = ude_gym_wrapper.reset()
        assert new_episode_obs == mock_reset_obs
        assert not ude_gym_wrapper.game_over

    def test_close(self):
        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)
        ude_gym_wrapper.close()
        self._ude_env_mock.close.assert_called_once()

    def test_render(self):
        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)
        # There should be no effect
        ude_gym_wrapper.render()

    def test_seed(self):
        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)
        # There should be no effect
        ude_gym_wrapper.seed()

    def test_action_space(self):
        action_space = Space([4, 2])
        action_space_dict = {"agent_0": action_space}
        self._ude_env_mock.action_space = action_space_dict
        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)
        assert ude_gym_wrapper.action_space == action_space

    def test_observation_space(self):
        observation_space = Space([4, 2])
        ude_gym_wrapper = UDEToGymWrapper(ude_env=self._ude_env_mock)
        assert ude_gym_wrapper.observation_space.shape == observation_space.shape
