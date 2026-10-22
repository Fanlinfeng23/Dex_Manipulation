#!/usr/bin/env python3
"""Rebuild the local startup adapter; never installs or changes active approval."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
APK = ROOT.parent / 'downloads/XRoboToolkit-Quest-1.0.1.apk'
UPSTREAM_SHA = 'c8a077451e7f6b66ca688668d7b32f5a6a6c19a8c0fa4657f2e1344493c174df'


def main():
    if hashlib.sha256(APK.read_bytes()).hexdigest() != UPSTREAM_SHA:
        raise RuntimeError('Official input APK SHA256 mismatch')
    java = ROOT / 'jdk-17.0.20.1+1-jre/bin/java'
    decoded = ROOT / 'rebuild-decoded'
    unsigned = ROOT / 'rebuild-unsigned.apk'
    signed = ROOT / 'rebuild-signed'
    def run(*args):
        subprocess.run(list(map(str, args)), check=True, cwd=ROOT)
    run(java, '-jar', ROOT / 'apktool.jar', 'd', '-f', APK, '-o', decoded)
    manifest = decoded / 'AndroidManifest.xml'
    text = manifest.read_text().replace('package="com.xrobotoolkit.client.quest"',
        'package="com.xrobotoolkit.client.quest.tianji"').replace(
        'android:name="com.unity3d.player.UnityPlayerActivity"',
        'android:name="com.tianji.xrobotoolkit.AutoSetupActivity"').replace(
        'android:label="@string/app_name"', 'android:label="XRoboToolkit Tianji Auto"')
    manifest.write_text(text)
    yaml = decoded / 'apktool.yml'
    yaml.write_text(yaml.read_text().replace('versionName: 1.0.1', 'versionName: 1.0.1-tianji.1'))
    target = decoded / 'smali/com/tianji/xrobotoolkit/AutoSetupActivity.smali'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'AutoSetupActivity.smali', target)
    run(java, '-jar', ROOT / 'apktool.jar', 'b', decoded, '-o', unsigned)
    run(java, '-jar', ROOT / 'signer.jar', '--apks', unsigned, '--out', signed, '--overwrite')
    output = signed / 'rebuild-aligned-debugSigned.apk'
    with zipfile.ZipFile(APK) as original, zipfile.ZipFile(output) as built:
        files = [n for n in original.namelist() if n.startswith(('assets/', 'lib/')) and not n.endswith('/')]
        if not all(original.read(n) == built.read(n) for n in files):
            raise RuntimeError('Unity/native capture payload changed unexpectedly')
    print(json.dumps({'apk': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'unchanged_capture_files': len(files)}, indent=2))
    print('Build only. Install and update the SHA256 approval after capture-only verification.')


if __name__ == '__main__':
    main()
