import socket
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientSession

from quest_teleop.config import SDK_ROOT
from quest_teleop.keyboard import run
from quest_teleop.server import create_app


@unittest.skipUnless((SDK_ROOT / 'SDK_PYTHON' /
                      ('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform vendor SDK required')
class LauncherTests(unittest.IsolatedAsyncioTestCase):
    async def check_owned_server(self, usb_fails):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        args = SimpleNamespace(port=port, adb='mock-adb', serial='test', config=None,
                               hardware=False, robot_ip=None, vel_ratio=None,
                               acc_ratio=None)
        console = MagicMock()
        console.__enter__.return_value = console
        console.poll.return_value = ['quit']
        app = create_app()

        async def usb(*_):
            async with ClientSession() as session:
                health = await (await session.get(f'http://127.0.0.1:{port}/api/health')).json()
                self.assertTrue(health['ok'])
                self.assertEqual(health['keyboard_control'], 1)
            if usb_fails:
                raise RuntimeError('test USB failure')

        console_name = 'WindowsConsole' if sys.platform == 'win32' else 'UnixConsole'
        with patch(f'quest_teleop.keyboard.{console_name}', return_value=console), \
                patch('quest_teleop.keyboard.create_app', return_value=app), \
                patch('quest_teleop.keyboard.run_usb', new=AsyncMock(side_effect=usb)) as setup_usb:
            if usb_fails:
                with self.assertRaisesRegex(RuntimeError, 'test USB failure'):
                    await run(args)
            else:
                await run(args)
            setup_usb.assert_awaited_once_with(port, args.adb, args.serial)
        self.assertFalse(app['preview'].control._process.is_alive())
        with socket.socket() as probe:
            if sys.platform == 'win32':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                # A closed server may leave accepted sockets in TIME_WAIT.
                # Rebinding with the server's reuse setting detects a live
                # listener without confusing TIME_WAIT with a leaked service.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))

    async def test_starts_service_before_usb_and_closes_owned_worker_on_quit(self):
        await self.check_owned_server(False)

    async def test_usb_failure_closes_owned_service_and_worker(self):
        await self.check_owned_server(True)


if __name__ == '__main__':
    unittest.main()
