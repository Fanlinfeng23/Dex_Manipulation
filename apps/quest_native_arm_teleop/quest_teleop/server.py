from __future__ import annotations

import asyncio
import json
import time
from aiohttp import web

from .config import APP_ROOT, load_config
from .service import PreviewServer


def create_app(config=None, *, service=None):
    service = service or PreviewServer(config or load_config())
    app = web.Application(client_max_size=16384)
    app['preview'] = service
    app.on_startup.append(service.start)
    app.on_shutdown.append(service.stop)

    async def fresh_code(request, response):
        # Force refreshes to load the current controller mapping and matching UI.
        if request.path in ('/', '/quest') or request.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'no-store'

    app.on_response_prepare.append(fresh_code)

    async def index(request):
        return web.FileResponse(APP_ROOT / 'web/index.html')

    async def quest(request):
        if service.native_backend:
            raise web.HTTPNotFound(text='原生采集使用 XRoboToolkit 官方应用')
        return web.FileResponse(APP_ROOT / 'web/quest.html')

    async def health(request):
        hardware = bool(service.config.get('hardware', {}).get('enabled', False))
        state = service.control.state or {}
        result = {'ok': service.control.healthy and not state.get('hardware_fault'), 'virtual_only': not hardware,
                                  'hardware': hardware,
                                  'hardware_connected': state.get('hardware_connected', False),
                                  'hardware_enabled': state.get('hardware_enabled', False),
                                  'hardware_fault': state.get('hardware_fault'),
                                  'motion_scope': state.get('motion_scope'),
                                  'control_sides': state.get('control_sides'),
                                  'hardware_identity': state.get('hardware_identity'),
                                  'source': state.get('source'), 'input_fresh': state.get('fresh', False),
                                  'robot_ip': service.config.get('hardware', {}).get('robot_ip') if hardware else None,
                                  'keyboard_control': 1,
                                  'follow_control': True,
                                  'input_recovery': True,
                                  'calibration_release_gate': True,
                                  'input_stale_ms': 150, 'recovery_timeout_ms': 1000,
                                  'timer_period_ms': service.control.timer_resolution.period_ms,
                                  'model': service.config['model'], 'visual_model': service.config.get('visual_model'),
                                  'pose_send_hz': 60, 'control_hz': service.config['control_hz'],
                                  'broadcast_hz': service.config['broadcast_hz'], 'fault': service.control.fault}
        if service.native_backend:
            result.update({'input_backend': 'xrobotoolkit', **service.native_status()})
        return web.json_response(result)

    async def robot_model(request):
        return web.json_response(service.control.visual_data or {'kind': 'schematic', 'model': service.config['model']})

    async def trajectory(request):
        service.prune_history(time.perf_counter())
        hardware = bool(service.config.get('hardware', {}).get('enabled', False))
        # Freeze the window before yielding. Encoding the entire 120-second
        # archive in json_response blocks pose ACKs on this same event loop.
        samples, events = list(service.history), list(service.events)
        metadata = {'format_version': 1, 'virtual_only': not hardware,
                    'hardware': hardware, 'retention_s': 120,
                    'max_sample_hz': min(30, service.config['broadcast_hz']),
                    'config': service.config}
        response = web.StreamResponse(headers={
            'Content-Type': 'application/json', 'Cache-Control': 'no-store',
            'Content-Disposition': 'attachment; filename="quest-trajectory.json"'})
        await response.prepare(request)
        await response.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False)[:-1].encode('utf-8'))
        for name, rows in (('samples', samples), ('events', events)):
            await response.write(f',"{name}":['.encode('utf-8'))
            for start in range(0, len(rows), 16):
                chunk = ','.join(json.dumps(row, ensure_ascii=False, allow_nan=False)
                                 for row in rows[start:start + 16])
                await response.write(((',' if start else '') + chunk).encode('utf-8'))
                # write() need not yield when the socket buffer has room.
                await asyncio.sleep(0)
            await response.write(b']')
        await response.write(b'}')
        await response.write_eof()
        return response

    async def robot_mesh(request):
        path = service.control.meshes.get(request.match_info['mesh_id'])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Content-Type': 'model/stl'})

    app.router.add_get('/', index)
    app.router.add_get('/quest', quest)
    app.router.add_get('/api/health', health)
    app.router.add_get('/api/robot-model', robot_model)
    app.router.add_get('/api/trajectory', trajectory)
    app.router.add_get('/robot-mesh/{mesh_id}', robot_mesh)
    app.router.add_get('/ws/{role}', service.websocket)
    app.router.add_static('/static/', APP_ROOT / 'web', show_index=False)
    app.router.add_static('/vendor/three/', APP_ROOT / 'node_modules/three', show_index=False)
    return app
