import asyncio
import json
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from quest_teleop.config import SDK_ROOT
from quest_teleop.server import create_app
from quest_teleop.keyboard import KeyboardClient, run as run_keyboard
from test_core import packet


@unittest.skipUnless((SDK_ROOT/'SDK_PYTHON'/('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform vendor SDK required')
class ServerTests(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def next_type(self, ws, kind):
        for _ in range(30):
            data=await ws.receive_json(timeout=2)
            if data['type']==kind:return data
        self.fail(f'Missing message type: {kind}')

    async def test_health_pages_and_origin(self):
        for path in ('/','/quest','/static/xr-input.js','/vendor/three/build/three.module.js'):
            response=await self.client.get(path);self.assertEqual(response.status,200)
        response=await self.client.get('/api/health')
        self.assertTrue((await response.json())['virtual_only'])
        response=await self.client.get('/ws/ui',headers={'Origin':'https://untrusted.example'})
        self.assertEqual(response.status,403)

    async def test_trajectory_pairs_input_with_output_and_read_is_non_destructive(self):
        ui=await self.client.ws_connect('/ws/ui');await self.next_type(ui,'welcome')
        data=packet(captured_ms=time.perf_counter()*1000)
        await ui.send_json(data);await ui.send_json({'type':'command','action':'calibrate'})
        while not (await self.next_type(ui,'state'))['enabled']:pass
        service=self.app['preview']
        service.last_recorded=float('-inf')
        service.record_state(time.perf_counter(),service.control.state,service.control.frame)
        response=await self.client.get('/api/trajectory');self.assertEqual(response.status,200)
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        archive=await response.json()
        self.assertTrue(archive['virtual_only']);self.assertEqual(archive['retention_s'],120)
        sample=next(s for s in reversed(archive['samples']) if s['state']['enabled'])
        self.assertEqual(sample['input']['seq'],sample['state']['seq'])
        self.assertEqual(sample['input']['controllers']['left']['position'],data['controllers']['left']['position'])
        self.assertIn('input_to_world_rotation',sample['state'])
        self.assertIn('actual',sample['state']['arms']['left'])
        self.assertIn('target',sample['state']['arms']['left'])
        self.assertNotIn('chain',sample['state']['arms']['left'])
        again=await (await self.client.get('/api/trajectory')).json()
        self.assertIn(sample,again['samples'])
        self.assertTrue(service.control.state['enabled'])
        await ui.close()

    async def test_large_trajectory_export_yields_to_live_input_tasks(self):
        service = self.app['preview']
        service.history.clear()
        service.last_recorded = time.perf_counter() + 60
        now = time.perf_counter()
        rows = [{'recorded_s': now, 'export_index': i, 'payload': '手柄' * 100}
                for i in range(512)]
        service.history.extend(rows)
        original_dumps = json.dumps
        encoded = 0
        progress = []

        def observe_encoding(value, *args, **kwargs):
            nonlocal encoded
            if isinstance(value, dict) and 'export_index' in value:
                encoded += 1
            return original_dumps(value, *args, **kwargs)

        async def other_io_task():
            while True:
                if 0 < encoded < len(rows):
                    progress.append(encoded)
                await asyncio.sleep(0)

        task = asyncio.create_task(other_io_task())
        try:
            with patch('quest_teleop.server.json.dumps', side_effect=observe_encoding):
                response = await self.client.get('/api/trajectory')
                archive = await response.json()
            self.assertEqual(archive['samples'], rows)
            self.assertTrue(progress, 'Export blocked other tasks until all rows were encoded')
            self.assertLessEqual(min(progress), 16)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_trajectory_expires_and_has_a_memory_bound(self):
        service=self.app['preview'];service.history.clear();service.last_recorded=float('-inf')
        snapshot=service.control.state;now=time.perf_counter()
        for i in range(4000): service.record_state(now+i/30,snapshot)
        self.assertLessEqual(len(service.history),3600)
        self.assertLessEqual(service.history[-1]['recorded_s']-service.history[0]['recorded_s'],120)
        service.prune_history(now+4000/30+121)
        self.assertEqual(len(service.history),0)

    async def test_urdf_manifest_and_mesh_routes(self):
        response = await self.client.get('/api/robot-model')
        data = await response.json()
        self.assertEqual(data['id'],'marvin_m6s_lite_615_v1')
        mesh_urls = [v['mesh_url'] for a in data['arms'].values() for link in a['links'] for v in link['visuals']]
        self.assertEqual(len(mesh_urls),18)
        for url in mesh_urls:
            response = await self.client.head(url)
            self.assertEqual(response.status,200)
            self.assertGreater(response.content_length,84)
        response = await self.client.get('/robot-mesh/unknown')
        self.assertEqual(response.status,404)

    async def test_ownership_source_selection_and_disconnect(self):
        ui=await self.client.ws_connect('/ws/ui')
        self.assertTrue((await self.next_type(ui,'welcome'))['control'])
        observer=await self.client.ws_connect('/ws/ui')
        self.assertFalse((await self.next_type(observer,'welcome'))['control'])
        await observer.send_json({'type':'command','action':'source','value':'quest'})
        await self.next_type(observer,'error')
        service=self.app['preview'];self.assertEqual(service.control.state['source'],'simulation')
        quest=await self.client.ws_connect('/ws/quest');await self.next_type(quest,'welcome')
        await quest.send_json({'type':'command','action':'arm'});await self.next_type(quest,'error')
        await ui.send_json({'type':'command','action':'source','value':'quest'})
        for _ in range(30):
            if (await self.next_type(ui,'state'))['source']=='quest':break
        await quest.send_json(packet())
        await quest.send_json({'type':'command','action':'arm'})
        for _ in range(30):
            if (await self.next_type(ui,'state'))['enabled']:break
        self.assertTrue(service.control.state['enabled'])
        await quest.send_json({'type':'command','action':'source','value':'simulation'})
        await self.next_type(quest,'error');self.assertEqual(service.control.state['source'],'quest')
        await quest.close()
        for _ in range(30):
            if not (await self.next_type(ui,'state'))['enabled']:break
        self.assertFalse(service.control.state['enabled'])
        await observer.close();await ui.close()

    async def test_malformed_input_pauses_and_second_quest_is_rejected(self):
        ui=await self.client.ws_connect('/ws/ui');await self.next_type(ui,'welcome')
        await ui.send_json(packet());await ui.send_json({'type':'command','action':'arm'})
        for _ in range(30):
            if (await self.next_type(ui,'state'))['enabled']:break
        self.assertTrue(self.app['preview'].control.state['enabled'])
        await ui.send_str('{invalid JSON');await self.next_type(ui,'error')
        while (await self.next_type(ui,'state'))['enabled']: pass
        self.assertFalse(self.app['preview'].control.state['enabled'])
        first=await self.client.ws_connect('/ws/quest');await self.next_type(first,'welcome')
        second=await self.client.ws_connect('/ws/quest');await self.next_type(second,'error')
        await second.close();await first.close();await ui.close()

    async def test_quest_calibration_and_home_commands(self):
        ui=await self.client.ws_connect('/ws/ui');await self.next_type(ui,'welcome')
        await ui.send_json({'type':'command','action':'source','value':'quest'})
        while (await self.next_type(ui,'state'))['source']!='quest': pass
        quest=await self.client.ws_connect('/ws/quest');await self.next_type(quest,'welcome')
        data=packet(captured_ms=time.perf_counter()*1000);await quest.send_json(data)
        await quest.send_json({'type':'command','action':'calibrate'})
        while True:
            initial=await self.next_type(ui,'state')
            if initial['enabled']:break
        self.assertEqual(initial['calibration_revision'],1)
        for i in range(1,9):
            data['seq']=i;data['captured_ms']=time.perf_counter()*1000
            for c in data['controllers'].values():
                c['grip']=1;c['position'][2]=-.42
            await quest.send_json(data)
            while True:
                moved=await self.next_type(ui,'state')
                if moved['seq']==i:break
        for side in ('left','right'):
            self.assertFalse(np.allclose(moved['arms'][side]['joints_deg'],initial['arms'][side]['joints_deg']))
        await quest.send_json({'type':'command','action':'reset'})
        while True:
            home=await self.next_type(ui,'state')
            if home['reset_revision']>initial['reset_revision']:break
        self.assertFalse(home['enabled']);self.assertFalse(home['following'])
        self.assertTrue(home['resetting'])
        for side in ('left','right'):
            self.assertFalse(np.allclose(home['arms'][side]['joints_deg'],initial['arms'][side]['joints_deg']))
        # Continue fresh input during the time-based return; B suppresses trigger following.
        for _ in range(150):
            data['seq']+=1;data['captured_ms']=time.perf_counter()*1000
            await quest.send_json(data)
            while True:
                home=await self.next_type(ui,'state')
                if home['seq']==data['seq']:break
            if not home['resetting']:break
        self.assertFalse(home['resetting'])
        self.assertFalse(home['enabled'])
        for side in ('left','right'):
            np.testing.assert_allclose(home['arms'][side]['joints_deg'],initial['arms'][side]['joints_deg'])
        await quest.close();await ui.close()

    async def test_frame_ack_and_command_retry_do_not_repeat_calibration(self):
        ui = await self.client.ws_connect('/ws/ui'); await self.next_type(ui, 'welcome')
        frame = packet(captured_ms=time.perf_counter()*1000)
        await ui.send_json(frame)
        ack = await self.next_type(ui, 'frame_ack')
        self.assertTrue(ack['accepted']); self.assertEqual(ack['seq'], frame['seq'])
        command = {'type': 'command', 'id': 'one-calibration', 'action': 'calibrate',
                   'session': frame['session'], 'captured_ms': frame['captured_ms'], 'frame': frame}
        await ui.send_json(command)
        ack = await self.next_type(ui, 'command_ack'); self.assertTrue(ack['ok'], ack)
        first = self.app['preview'].control.state['calibration_revision']
        for _ in range(3):
            await ui.send_json(command)
            self.assertEqual(await self.next_type(ui, 'command_ack'), ack)
        self.assertEqual(self.app['preview'].control.state['calibration_revision'], first)
        await ui.send_json({'type': 'command', 'id': ['invalid'], 'action': 'reset'})
        await self.next_type(ui, 'error')
        # A malformed ID does not break the websocket handler or execute B.
        await ui.send_json({'type': 'command', 'id': 'pause-after-invalid', 'action': 'pause'})
        self.assertTrue((await self.next_type(ui, 'command_ack'))['ok'])
        self.assertFalse(self.app['preview'].control.state['enabled'])
        await ui.close()

    async def test_bound_calibration_cannot_clear_a_latched_tracking_loss(self):
        ui = await self.client.ws_connect('/ws/ui'); await self.next_type(ui, 'welcome')
        frame = packet(captured_ms=time.perf_counter()*1000)
        frame['safety'] = {'tracking_lost': True}
        await ui.send_json(frame); await self.next_type(ui, 'frame_ack')
        await ui.send_json({'type': 'command', 'id': 'loss-calibration', 'action': 'calibrate',
                           'session': frame['session'], 'captured_ms': frame['captured_ms'], 'frame': frame})
        ack = await self.next_type(ui, 'command_ack')
        self.assertFalse(ack['ok']); self.assertFalse(self.app['preview'].control.state['enabled'])
        await ui.close()

    async def test_input_gap_pauses_without_disconnect_and_is_recorded_once(self):
        ui = await self.client.ws_connect('/ws/ui')
        self.assertTrue((await self.next_type(ui, 'welcome'))['input_recovery'])
        frame = packet(captured_ms=time.perf_counter() * 1000)
        await ui.send_json(frame); await self.next_type(ui, 'frame_ack')
        await ui.send_json({'type': 'command', 'id': 'calibrate-before-gap', 'action': 'calibrate'})
        self.assertTrue((await self.next_type(ui, 'command_ack'))['ok'])
        await ui.send_json({'type': 'command', 'id': 'start-before-gap', 'action': 'follow', 'value': True})
        self.assertTrue((await self.next_type(ui, 'command_ack'))['ok'])
        service = self.app['preview']
        held = {side: arm['joints_deg'] for side, arm in service.control.state['arms'].items()}
        pause = {'type': 'command', 'id': 'gap-1', 'action': 'input_pause', 'value': 'ack_timeout'}
        for _ in range(3):
            await ui.send_json(pause)
            self.assertTrue((await self.next_type(ui, 'command_ack'))['input_paused'])
        self.assertFalse(service.control.state['enabled'])
        frame['seq'] += 1; frame['captured_ms'] = time.perf_counter() * 1000
        for controller in frame['controllers'].values():
            controller['grip'] = 1.; controller['position'][2] -= .1
        await ui.send_json(frame)
        self.assertTrue((await self.next_type(ui, 'frame_ack'))['accepted'])
        for _ in range(30):
            state = await self.next_type(ui, 'state')
            if state['seq'] == frame['seq']: break
        self.assertEqual(state['seq'], frame['seq'])
        self.assertTrue(state['fresh']); self.assertFalse(state['enabled']); self.assertFalse(state['following'])
        for side in held:
            np.testing.assert_allclose(state['arms'][side]['joints_deg'], held[side])
        service.last_recorded = float('-inf'); service.last_input_key = None
        now = time.perf_counter()
        service.record_state(now, service.control.state, service.control.frame)
        service.record_state(now + .04, service.control.state, service.control.frame)
        self.assertTrue(service.history[-2]['new_input']); self.assertFalse(service.history[-1]['new_input'])
        archive = await (await self.client.get('/api/trajectory')).json()
        events = [event for event in archive['events'] if event.get('id') == 'gap-1']
        self.assertEqual([event['type'] for event in events], ['input_pause', 'input_resumed'])
        self.assertEqual(events[0]['cause'], 'ack_timeout')
        self.assertEqual(events[1]['seq'], frame['seq'])
        self.assertLessEqual(events[0]['recorded_s'], events[1]['recorded_s'])
        self.assertFalse(ui.closed)
        service.prune_history(time.perf_counter() + 121)
        self.assertFalse(service.events)
        await ui.close()

    async def test_high_trigger_calibration_succeeds_and_records_the_input_used(self):
        ui = await self.client.ws_connect('/ws/ui'); await self.next_type(ui, 'welcome')
        await ui.send_json({'type': 'command', 'id': 'source-quest', 'action': 'source', 'value': 'quest'})
        await self.next_type(ui, 'command_ack')
        quest = await self.client.ws_connect('/ws/quest'); await self.next_type(quest, 'welcome')
        frame = packet(captured_ms=time.perf_counter() * 1000)
        frame['controllers']['right']['grip'] = 1.
        await quest.send_json(frame); await self.next_type(quest, 'frame_ack')
        command = {'type': 'command', 'id': 'a-high-trigger', 'action': 'calibrate'}
        for _ in range(2):
            await ui.send_json(command)
            reply = await self.next_type(ui, 'command_ack')
            self.assertTrue(reply['ok'])
            self.assertEqual(reply['calibration_input']['raw_triggers'], {'left': 0., 'right': 1.})
        service = self.app['preview']
        self.assertTrue(service.control.state['enabled']); self.assertFalse(service.control.state['following'])
        self.assertEqual(service.control.state['trigger_source'], 'web')
        self.assertFalse(service.control.state['ui_follow'])
        events = [e for e in service.events if e.get('id') == command['id']]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'calibration')
        self.assertEqual(events[0]['input']['seq'], frame['seq'])
        await quest.close(); await ui.close()

    async def keyboard_command(self, ws, action, command_id=None):
        await ws.send_json({'type': 'command', 'id': command_id or str(time.perf_counter()), 'action': action})
        return await self.next_type(ws, 'command_ack')

    async def test_keyboard_with_existing_desktop_and_quest(self):
        ui = await self.client.ws_connect('/ws/ui'); await self.next_type(ui, 'welcome')
        keyboard = await self.client.ws_connect('/ws/keyboard')
        self.assertTrue((await self.next_type(keyboard, 'welcome'))['control'])
        self.assertTrue((await self.keyboard_command(keyboard, 'keyboard_attach'))['ok'])
        self.assertIsNotNone(self.app['preview'].owner)
        self.assertEqual(self.app['preview'].control.state['source'], 'quest')
        second = await self.client.ws_connect('/ws/keyboard')
        await self.next_type(second, 'error'); await second.close()
        quest = await self.client.ws_connect('/ws/quest'); await self.next_type(quest, 'welcome')
        await quest.send_json(packet(captured_ms=time.perf_counter() * 1000))
        await self.next_type(quest, 'frame_ack')
        self.assertTrue((await self.keyboard_command(keyboard, 'calibrate'))['ok'])
        started = await self.keyboard_command(keyboard, 'keyboard_toggle', 'one-space')
        self.assertTrue(started['ok']); self.assertTrue(started['keyboard_follow'])
        self.assertEqual(await self.keyboard_command(keyboard, 'keyboard_toggle', 'one-space'), started)
        stopped = await self.keyboard_command(keyboard, 'keyboard_toggle')
        self.assertTrue(stopped['ok']); self.assertFalse(stopped['keyboard_follow'])
        self.assertTrue(self.app['preview'].control.state['enabled'])
        self.assertFalse((await self.keyboard_command(ui, 'keyboard_attach'))['ok'])
        self.assertFalse((await self.keyboard_command(keyboard, 'source'))['ok'])
        await quest.send_json(packet(seq=1, captured_ms=time.perf_counter() * 1000))
        await self.next_type(quest, 'frame_ack')
        reset = await self.keyboard_command(keyboard, 'reset')
        self.assertTrue(reset['ok'], reset)
        self.assertFalse(self.app['preview'].control.state['enabled'])
        await keyboard.close()
        for _ in range(50):
            if self.app['preview'].keyboard is None:
                break
            await asyncio.sleep(.02)
        self.assertIsNone(self.app['preview'].keyboard)
        self.assertFalse(self.app['preview'].control.state['keyboard_mode'])
        await quest.close(); await ui.close()

    async def test_console_client_a_space_stop_b_and_quit_with_live_pose_stream(self):
        quest = await self.client.ws_connect('/ws/quest'); await self.next_type(quest, 'welcome')
        service = self.app['preview']

        async def feed():
            seq = 0
            while True:
                await quest.send_json(packet(seq=seq, captured_ms=time.perf_counter() * 1000))
                await self.next_type(quest, 'frame_ack')
                seq += 1
                await asyncio.sleep(.015)

        actions = []

        class Console:
            def poll(self):
                state = service.control.state
                checks = [(state['fresh'], 'calibrate'), (state['enabled'], 'keyboard_toggle'),
                          (state['following'], 'keyboard_toggle'),
                          (state['enabled'] and not state['following'], 'reset'),
                          (state['reset_revision'] > 0, 'quit')]
                ready, action = checks[len(actions)]
                if ready:
                    actions.append(action)
                    return [action]
                return []

        stream = asyncio.create_task(feed())
        keyboard = await self.client.ws_connect('/ws/keyboard'); await self.next_type(keyboard, 'welcome')
        reports = []
        try:
            await asyncio.wait_for(KeyboardClient(keyboard, report=reports.append).run(Console()), 6)
            self.assertEqual(actions, ['calibrate', 'keyboard_toggle', 'keyboard_toggle', 'reset', 'quit'])
            self.assertFalse(service.control.state['keyboard_mode'])
            self.assertFalse(service.control.state['enabled'])
            self.assertTrue(any('遥操已开始' in text for text in reports))
            self.assertTrue(any('遥操已停止' in text for text in reports))
        finally:
            stream.cancel()
            try:
                await stream
            except asyncio.CancelledError:
                pass
            await keyboard.close(); await quest.close()

    async def test_launcher_reuses_service_runs_usb_and_exits_without_stopping_service(self):
        console = MagicMock()
        console.poll.return_value = ['quit']
        console.__enter__.return_value = console
        args = SimpleNamespace(port=self.client.server.port, adb='adb-test', serial='quest-test',
                               config=None, hardware=False, robot_ip=None,
                               vel_ratio=None, acc_ratio=None)
        console_name = 'WindowsConsole' if sys.platform == 'win32' else 'UnixConsole'
        with patch(f'quest_teleop.keyboard.{console_name}', return_value=console), \
                patch('quest_teleop.keyboard.run_usb', new_callable=AsyncMock) as usb:
            await run_keyboard(args)
            usb.assert_awaited_once_with(args.port, args.adb, args.serial)
        health = await (await self.client.get('/api/health')).json()
        self.assertTrue(health['ok'])
        self.assertFalse(self.app['preview'].control.state['keyboard_mode'])
        self.assertFalse(self.app['preview'].control.state['enabled'])


if __name__=='__main__':unittest.main()
