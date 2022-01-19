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
"""A class for Dummy Side Channel."""
from ude.side_channels.ude_side_channel import AbstractSideChannel
from ude.ude_typing import SideChannelData


class DummySideChannel(AbstractSideChannel):
    """
    DummySideChannel to be used for Environment not supporting side channel.
    """
    def __init__(self):
        """
        Initialize DummySideChannel
        """
        super().__init__()

    def _send(self, key: str, value: SideChannelData, store_local: bool = False) -> None:
        """
        Send the side channel message. DummySideChannel actually does nothing! :)

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
            store_local (bool, optional): The flag whether to store locally or not.
        """
        return
