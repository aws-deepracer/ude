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
"""A class for Abstract Side Channel."""
import abc
import copy
from typing import Any
import threading

from ude.ude_typing import SideChannelData


# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta('ABC', (object,), {})


class SideChannelObserverInterface(ABC):
    """
    SideChannelObserverInterface class to get a callback
    when message is received.
    """
    @abc.abstractmethod
    def on_received(self, side_channel: 'AbstractSideChannel', key: str, value: SideChannelData) -> None:
        """
        Callback when side channel instance receives new message.

        Args:
            side_channel (AbstractSideChannel): side channel instance
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
        """
        pass


class AbstractSideChannel(ABC):
    """
    AbstractSideChannel class to be a base class for side channel implementations.
    """
    def __init__(self):
        """
        Initialize AbstractSideChannel.
        """
        self._data_map = {}
        self._observers = set()
        self._key_observer_map = {}
        self._lock = threading.RLock()

    def register(self, observer: SideChannelObserverInterface, key: str = None) -> None:
        """
        Register observer for callback notification when the message is received
        with given key. (None means receiving the notification for all the messages)

        Args:
            observer (SideChannelObserverInterface): observer instance to callback
            key (str): message key to get notification.
                       None means receiving the notification for all the messages.
        """
        with self._lock:
            if key is None:
                self._observers.add(observer)
                # Remove given observer from all sets in key_observer_map
                # as observer will get notification for all keys now.
                for key in self._key_observer_map:
                    self._key_observer_map[key].discard(observer)
            else:
                if observer in self._observers:
                    # Remove observer from the notification of all keys now.
                    self._observers.discard(observer)
                if key not in self._key_observer_map:
                    self._key_observer_map[key] = set()
                self._key_observer_map[key].add(observer)

    def unregister(self, observer: SideChannelObserverInterface, key: str = None) -> None:
        """
        Unregister observer from callback notification list for given key.

         Args:
            observer (SideChannelObserverInterface): observer instance to remove from callback list.
            key (str): message key to remove from notification list.
                       None means removing the notification for all the messages.
        """
        with self._lock:
            if key is None:
                self._observers.discard(observer)
                for key in self._key_observer_map:
                    self._key_observer_map[key].discard(observer)
            else:
                if key in self._key_observer_map:
                    self._key_observer_map[key].discard(observer)

    def notify(self, key, value) -> None:
        """
        Notify the observers with key and value received.

        Args:
            key (str): the key of the message
            value (SideChannelData): The data of the message.
        """
        with self._lock:
            observers = copy.copy(self._observers)
            key_observer_map = copy.copy(self._key_observer_map)
        # Notify data received
        for callback in observers:
            callback.on_received(side_channel=self, key=key, value=value)
        if key in key_observer_map:
            for callback in key_observer_map[key]:
                callback.on_received(side_channel=self, key=key, value=value)

    def get(self, key: str, default: Any = None) -> SideChannelData:
        """
        Return the data with given key and return given default if not found.

        Args:
            key (str): key for message value
            default (Any): default value to return when not found.

        Returns:
            SideChannelData: the data value with given key.
                             - Returns default if not found.

        """
        if key in self._data_map:
            return self._data_map[key]
        return default

    def store(self, key: str, value: SideChannelData) -> None:
        """
        Store the given key and value locally.

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
        """
        self._data_map[key] = value

    def send(self, key: str, value: SideChannelData, store_local: bool = False) -> None:
        """
        Send key and value pair to environment as a side channel message.

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
            store_local (bool, optional): The flag whether to store locally or not.
        """
        if store_local:
            self.store(key=key, value=value)
        self._send(key=key, value=value, store_local=store_local)

    @abc.abstractmethod
    def _send(self, key: str, value: SideChannelData, store_local: bool = False) -> None:
        """
        Actual implementation of sending key and value pair to environment as a side channel message.
        * Note: The caller of _send function, send function, actually already stored the message locally
                if store_local is True.
                However, we are still passing this as this value is used as sync flag to synchronize
                the side channel message to every UDE Clients connected to UDE Server when store_local is True.
                Maybe, we can figure out better way to do this, but keeping this for now.

        Args:
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
            store_local (bool, optional): The flag whether to store locally or not.
        """
        raise NotImplementedError()
