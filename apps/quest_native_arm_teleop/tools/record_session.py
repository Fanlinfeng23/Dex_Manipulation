#!/usr/bin/env python3
"""Read-only recorder of the server's sampled trajectory (not full-rate native SDK)."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

async def record(args):
    deadline = time.monotonic() + args.seconds
    start_unix_ms = time.time() * 1000 if args.from_now else None
    seen = set()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
        with path.open('x', encoding='utf-8') as stream:
            while time.monotonic() < deadline:
                try:
                    async with session.get(args.url + '/api/trajectory') as response:
                        response.raise_for_status()
                        data = await response.json()
                    for kind, rows in (('sample', data['samples']), ('event', data['events'])):
                        for row in rows:
                            if start_unix_ms is not None and row.get('recorded_unix_ms', 0) < start_unix_ms:
                                continue
                            text = json.dumps({'kind': kind, **row}, ensure_ascii=False, separators=(',', ':'))
                            # Bounded input archive; discard duplicate polling windows.
                            if text not in seen:
                                stream.write(text + '\n'); seen.add(text); count += 1
                    # Keep only the current server retention window for deduplication.
                    seen = {json.dumps({'kind': kind, **row}, ensure_ascii=False, separators=(',', ':'))
                            for kind, rows in (('sample', data['samples']), ('event', data['events'])) for row in rows}
                    stream.flush()
                except (aiohttp.ClientError, TimeoutError) as exc:
                    stream.write(json.dumps({'kind': 'recorder_gap', 'wall_time': time.time(), 'error': str(exc)}) + '\n')
                    stream.flush()
                await asyncio.sleep(min(5, max(0, deadline - time.monotonic())))
    print(f'{path}: {count} sampled records (not full-rate native SDK frames)')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8766')
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--output', required=True)
    p.add_argument('--from-now', action='store_true',
                   help='only save samples recorded after this command starts')
    args = p.parse_args()
    if args.seconds <= 0: p.error('--seconds must be positive')
    asyncio.run(record(args))
