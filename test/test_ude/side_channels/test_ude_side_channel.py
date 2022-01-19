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
from unittest.mock import MagicMock


from ude.side_channels.ude_side_channel import AbstractSideChannel


def concreter(abclass):
    class ConcreteCls(abclass):
        pass
    ConcreteCls.__abstractmethods__ = frozenset()
    return type('DummyConcrete' + abclass.__name__, (ConcreteCls,), {})


class AbstractSideChannelTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_register(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer)
        side_channel.notify("key", "value")

        observer.on_received.assert_called_once_with(side_channel=side_channel,
                                                     key="key",
                                                     value="value")

    def test_register_with_key(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer, key="key")
        side_channel.notify("key", "value")
        side_channel.notify("key2", "value")
        # Should have called with registered key only!
        observer.on_received.assert_called_once_with(side_channel=side_channel,
                                                     key="key",
                                                     value="value")

    def test_register_two_observers_with_key(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        observer2 = MagicMock()
        side_channel.register(observer=observer, key="key")
        side_channel.register(observer=observer2, key="key")
        side_channel.notify("key", "value")
        side_channel.notify("key2", "value")
        # Should have called with registered key only!
        observer.on_received.assert_called_once_with(side_channel=side_channel,
                                                     key="key",
                                                     value="value")
        observer2.on_received.assert_called_once_with(side_channel=side_channel,
                                                      key="key",
                                                      value="value")

    def test_unregister(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer)
        side_channel.unregister(observer=observer)
        side_channel.notify("key", "value")
        observer.on_received.assert_not_called()

    def test_unregister_with_key(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer, key="key")
        side_channel.unregister(observer=observer, key="key")
        side_channel.notify("key", "value")
        observer.on_received.assert_not_called()

    def test_unregister_all_even_with_key(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer, key="key")
        side_channel.unregister(observer=observer)
        side_channel.notify("key", "value")
        observer.on_received.assert_not_called()

    def test_register_with_key_then_all(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer, key="key")
        side_channel.register(observer=observer)
        side_channel.notify("key", "value")

        # Should have called with registered key only!
        observer.on_received.assert_called_once_with(side_channel=side_channel,
                                                     key="key",
                                                     value="value")

        side_channel.notify("key2", "value")
        # Should only called once as observer should be registered for all now!
        assert observer.on_received.call_count == 2

    def test_register_all_then_with_key(self):
        side_channel = concreter(AbstractSideChannel)()
        observer = MagicMock()
        side_channel.register(observer=observer)
        side_channel.register(observer=observer, key="key")
        side_channel.notify("key2", "value")
        # Now key with key2 shouldn't notify the observer
        observer.on_received.assert_not_called()

        side_channel.notify("key", "value")

        # Should have called with registered key only!
        observer.on_received.assert_called_once_with(side_channel=side_channel,
                                                     key="key",
                                                     value="value")

    def test_store_and_get(self):
        side_channel = concreter(AbstractSideChannel)()
        side_channel.store("key", "value")
        ret_val = side_channel.get("key")
        assert "value" == ret_val

    def test_send(self):
        side_channel = concreter(AbstractSideChannel)()
        side_channel._send = MagicMock(name="_send")
        side_channel.send(key="key", value="value", store_local=False)
        ret_val = side_channel.get("key")
        assert ret_val is None
        side_channel._send.assert_called_once_with(key="key",
                                                   value="value",
                                                   store_local=False)

    def test_send_with_store_local(self):
        side_channel = concreter(AbstractSideChannel)()
        side_channel._send = MagicMock(name="_send")
        side_channel.send(key="key", value="value", store_local=True)
        ret_val = side_channel.get("key")
        assert ret_val == "value"
        side_channel._send.assert_called_once_with(key="key",
                                                   value="value",
                                                   store_local=True)






