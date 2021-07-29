import uuid
from mlagents_envs.side_channel.side_channel import (SideChannel,IncomingMessage)

class Skeleton_SideChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.msg = msg.read_float32_list()
        # We simply read a string from the message and print it.
        print(msg.read_float32_list())

    def send_string(self, data: str) -> None:
        pass

    def get_info(self):
        return self.msg