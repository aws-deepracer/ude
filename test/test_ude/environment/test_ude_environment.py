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
from unittest.mock import patch

from ude.environment.ude_environment import UDEEnvironment
from ude.environment.constants import UDEResetMode
from ude.environment.adapters.remote_environment_adapter import RemoteEnvironmentAdapter

from gym.spaces.space import Space


@patch("ude.environment.interfaces.UDEEnvironmentAdapterInterface")
class UDEEnvironmentTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_reset_mode(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO)
        assert env.reset_mode == UDEResetMode.AUTO

    def test_set_reset_mode(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO)
        env.reset_mode = UDEResetMode.MANUAL
        assert env.reset_mode == UDEResetMode.MANUAL

    def test_env(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        assert env.env == ude_env_adapter_mock.return_value

    def test_side_channel(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        assert env.side_channel == ude_env_adapter_mock.return_value.side_channel

    def test_is_remote(self, ude_env_adapter_mock):
        with patch("ude.environment.adapters.remote_environment_adapter.UDEClient"):
            env = UDEEnvironment(ude_env_adapter=RemoteEnvironmentAdapter("localhost"))
        assert env.is_remote

    def test_is_remote_with_non_remote_env_adapter(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        assert not env.is_remote

    def test_is_local(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        assert env.is_local

    def test_is_local_with_remote_env_adapter(self, ude_env_adapter_mock):
        with patch("ude.environment.adapters.remote_environment_adapter.UDEClient"):
            env = UDEEnvironment(ude_env_adapter=RemoteEnvironmentAdapter("localhost"))
        assert not env.is_local

    def test_step(self, ude_env_adapter_mock):
        action_dict = {"agent": 1}

        next_state = {"agent": "next_state"}
        done = {"agent": False}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs == next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)

    def test_step_auto_reset(self, ude_env_adapter_mock):
        action_dict = {"agent": 1}

        next_state = {"agent": "next_state"}
        reset_next_state = {"agent": "reset_next_state"}
        done = {"agent": True}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        ude_env_adapter_mock.return_value.reset.return_value = reset_next_state
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs != next_state
        assert ret_obs == reset_next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info

        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_called_once()

    def test_step_no_auto_reset(self, ude_env_adapter_mock):
        action_dict = {"agent": 1}

        next_state = {"agent": "next_state"}
        done = {"agent": True}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs == next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_not_called()

    def test_step_no_auto_reset_with_remote_env_adapter(self, ude_env_adapter_mock):
        # Even UDEEnvironment is configured with reset_mode as AUTO
        # if adapter is RemoteEnvironmentAdapter, it shouldn't reset as
        # remote environment will own the responsibility of auto-reset.
        action_dict = {"agent": 1}

        next_state = {"agent": "next_state"}
        done = {"agent": True}
        reward = {"agent": 42}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        with patch("ude.environment.adapters.remote_environment_adapter.UDEClient") as ude_client_mock:
            ude_client_mock.return_value.step.return_value = step_dict

            env = UDEEnvironment(ude_env_adapter=RemoteEnvironmentAdapter("localhost"),
                                 reset_mode=UDEResetMode.AUTO)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs == next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_client_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_not_called()

    def test_step_auto_reset_with_cond_all_and_partial_dones(self, ude_env_adapter_mock):
        action_dict = {"agent1": 1,
                       "agent2": 2}

        next_state = {"agent1": "next_state1",
                      "agent2": "next_state2"}
        done = {"agent1": True,
                "agent2": False}
        reward = {"agent1": 42,
                  "agent2": 43}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO,
                             game_over_cond=all)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs == next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_not_called()

    def test_step_auto_reset_with_cond_all_and_all_dones(self, ude_env_adapter_mock):
        action_dict = {"agent1": 1,
                       "agent2": 2}

        next_state = {"agent1": "next_state1",
                      "agent2": "next_state2"}
        reset_next_state = {"agent1": "reset_next_state1",
                            "agent2": "reset_next_state2"}
        done = {"agent1": True,
                "agent2": True}
        reward = {"agent1": 42,
                  "agent2": 43}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        ude_env_adapter_mock.return_value.reset.return_value = reset_next_state
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO,
                             game_over_cond=all)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs != next_state
        assert ret_obs == reset_next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_called_once()

    def test_step_auto_reset_with_cond_any_and_partial_dones(self, ude_env_adapter_mock):
        action_dict = {"agent1": 1,
                       "agent2": 2}

        next_state = {"agent1": "next_state1",
                      "agent2": "next_state2"}
        reset_next_state = {"agent1": "reset_next_state1",
                            "agent2": "reset_next_state2"}
        done = {"agent1": True,
                "agent2": False}
        reward = {"agent1": 42,
                  "agent2": 43}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        ude_env_adapter_mock.return_value.reset.return_value = reset_next_state
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO,
                             game_over_cond=any)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs != next_state
        assert ret_obs == reset_next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_called_once()

    def test_step_auto_reset_with_cond_any_and_all_dones(self, ude_env_adapter_mock):
        action_dict = {"agent1": 1,
                       "agent2": 2}

        next_state = {"agent1": "next_state1",
                      "agent2": "next_state2"}
        reset_next_state = {"agent1": "reset_next_state1",
                            "agent2": "reset_next_state2"}
        done = {"agent1": True,
                "agent2": True}
        reward = {"agent1": 42,
                  "agent2": 43}
        info = {}
        step_dict = (next_state,
                     reward,
                     done,
                     action_dict,
                     info)
        ude_env_adapter_mock.return_value.step.return_value = step_dict
        ude_env_adapter_mock.return_value.reset.return_value = reset_next_state
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock(),
                             reset_mode=UDEResetMode.AUTO,
                             game_over_cond=any)
        ret_obs, ret_reward, ret_done, ret_last_action, ret_info = env.step(action_dict=action_dict)

        assert ret_obs != next_state
        assert ret_obs == reset_next_state
        assert ret_reward == reward
        assert ret_done == done
        assert ret_last_action == action_dict
        assert ret_info == info
        ude_env_adapter_mock.return_value.step.assert_called_once_with(action_dict=action_dict)
        ude_env_adapter_mock.return_value.reset.assert_called_once()

    def test_reset(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        env.reset()
        ude_env_adapter_mock.return_value.reset.assert_called_once()

    def test_close(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        env.close()
        ude_env_adapter_mock.return_value.close.assert_called_once()

    def test_observation_space(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        expected_observation_space = Space([4, 2])
        ude_env_adapter_mock.return_value.observation_space = expected_observation_space
        ret_observation_space = env.observation_space
        assert expected_observation_space == ret_observation_space

    def test_action_space(self, ude_env_adapter_mock):
        env = UDEEnvironment(ude_env_adapter=ude_env_adapter_mock())
        expected_action_space = Space([4, 2])
        ude_env_adapter_mock.return_value.action_space = expected_action_space
        ret_action_space = env.action_space
        assert expected_action_space == ret_action_space
