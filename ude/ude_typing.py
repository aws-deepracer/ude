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
"""Module to contain UDE specific typing."""
from typing import Dict, Any, Tuple, Union


# Represents a generic identifier for an agent (e.g., "agent1").
AgentID = Any

# A dict keyed by agent ids, e.g. {"agent-1": value}.
MultiAgentDict = Dict[AgentID, Any]

# UDE Step Result type:
# - observation(s), reward(s), done(s), last action(s), info
UDEStepResult = Tuple[MultiAgentDict, MultiAgentDict,
                      MultiAgentDict, MultiAgentDict,
                      Dict]

# UDE Reset Result type:
# - observation(s), info
UDEResetResult = Tuple[MultiAgentDict, Dict]

# Side Channel Data type
SideChannelData = Union[bool, int, float, list, str, bytes]
