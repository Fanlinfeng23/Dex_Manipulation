import asyncio
import json
import unittest

from quest_teleop.service import Peer


class Socket:
    def __init__(self, blocked=False):
        self.closed = False
        self.blocked = blocked
        self.started = asyncio.Event()
        self.delivered = asyncio.Event()
        self.release = asyncio.Event()
        self.messages = []

    async def send_str(self, value):
        self.started.set()
        if self.blocked:
            await self.release.wait()
        self.messages.append(json.loads(value))
        self.delivered.set()

    async def close(self):
        self.closed = True


class BroadcastTests(unittest.IsolatedAsyncioTestCase):
    async def test_brief_send_stall_recovers_but_long_stall_closes(self):
        socket = Socket(True)
        peer = Peer(socket)
        try:
            peer.reply({'type': 'command_ack', 'id': 'pause', 'ok': True})
            await socket.started.wait()
            await asyncio.sleep(.2)
            self.assertFalse(socket.closed)
            socket.release.set()
            await asyncio.wait_for(socket.delivered.wait(), timeout=.5)
            self.assertEqual(socket.messages[0]['id'], 'pause')
            socket.release.clear(); socket.delivered.clear()
            peer.publish('{"type":"state"}')
            await asyncio.wait_for(peer.task, timeout=1.5)
            self.assertTrue(socket.closed)
        finally:
            await peer.close()

    async def test_slow_viewer_cannot_delay_another_inputs_ack_or_build_state_queue(self):
        blocked, normal = Socket(True), Socket()
        slow, fast = Peer(blocked), Peer(normal)
        try:
            slow.publish('{"seq": 0}')
            await blocked.started.wait()
            for seq in range(1, 1000):
                slow.publish(json.dumps({'seq': seq}))
            fast.reply({'type': 'frame_ack', 'seq': 10, 'accepted': True})
            await asyncio.wait_for(normal.delivered.wait(), timeout=.1)
            self.assertFalse(blocked.closed)
            self.assertEqual(normal.messages[0]['seq'], 10)
            self.assertEqual(json.loads(slow.latest)['seq'], 999)
            self.assertEqual(len(slow.replies), 0)
        finally:
            await slow.close()
            await fast.close()


if __name__ == '__main__':
    unittest.main()
