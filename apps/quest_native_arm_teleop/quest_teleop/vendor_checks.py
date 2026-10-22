"""Read-only checks against the vendor C ABI and controller configuration."""
import configparser
import ctypes
import tempfile
from pathlib import Path

import numpy as np

from .config import SDK_ROOT


def bind_robot_abi(robot, dcss_type):
    """Declare only the official C functions used by this adapter."""
    if ctypes.sizeof(dcss_type) != 1428:
        raise RuntimeError('100343 DCSS 结构大小不匹配；拒绝连接')
    pointer = ctypes.POINTER(ctypes.c_double)
    functions = {'OnLinkTo': ([ctypes.c_ubyte] * 4, ctypes.c_bool),
                 'OnGetBuf': ([ctypes.POINTER(dcss_type)], ctypes.c_bool),
                 'OnGetSDKVersion': ([], ctypes.c_long),
                 'OnRelease': ([], ctypes.c_bool), 'OnClearSet': ([], ctypes.c_bool),
                 'OnSetSend': ([], ctypes.c_bool),
                 'OnRecvFile': ([ctypes.c_char_p, ctypes.c_char_p], ctypes.c_bool)}
    for arm in ('A', 'B'):
        for name, args in (('OnSetTargetState', [ctypes.c_int]), ('OnSetImpType', [ctypes.c_int]),
                           ('OnSetJointLmt', [ctypes.c_int, ctypes.c_int]),
                           ('OnSetJointCmdPos', [pointer]), ('OnSetJointKD', [pointer, pointer])):
            functions[f'{name}_{arm}'] = (args, ctypes.c_bool)
        functions[f'OnGetServoErr_{arm}'] = ([ctypes.POINTER(ctypes.c_long)], None)
        functions[f'OnClearErr_{arm}'] = ([], None)
    for name, (args, result) in functions.items():
        function = getattr(robot.robot, name)
        function.argtypes, function.restype = args, result


def read_servo_errors(robot):
    """Read the seven raw drive error words for each arm without clearing them."""
    result = {}
    for arm, side in (('A', 'left'), ('B', 'right')):
        values = (ctypes.c_long * 7)()
        getattr(robot.robot, f'OnGetServoErr_{arm}')(values)
        result[side] = [int(value) for value in values]
    return result


def read_int_parameter(robot, name):
    # MarvinSDK.h declares long*, which is 64-bit on Linux x86_64.
    # The vendor Python get_param wrapper allocates c_int instead; do not
    # use that undersized buffer on this platform.
    encoded = name.encode('ascii')
    if len(encoded) > 29:
        raise ValueError('厂家参数名最多 29 字节')
    call = robot.robot.OnGetIntPara
    call.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long)]
    call.restype = ctypes.c_long
    value = ctypes.c_long()
    result = call(ctypes.create_string_buffer(encoded, 30), ctypes.byref(value))
    if result != 0:
        raise RuntimeError(f'读取 {name} 失败：{result}')
    return value.value


def version_info(robot):
    sdk = int(robot.SDK_version())
    controller = read_int_parameter(robot, 'VERSION')
    if sdk // 1000 != controller // 1000 or controller // 1000 != 100343:
        raise RuntimeError(f'控制器 {controller} 与本机 SDK {sdk} 不匹配；未发送运动指令')
    return {'sdk_version': sdk, 'controller_version': controller}


def compare_model(path, config):
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path, encoding='utf-8-sig')
    lines = [line.strip() for line in (SDK_ROOT / 'CommonConfig' / f"{config['model']}.MvKDCfg")
             .read_text().splitlines() if line.strip()]
    if len(lines) != 26:
        raise ValueError('不支持的厂家模型文件结构')
    for arm in range(2):
        model = [[float(x) for x in line.rstrip(',').split(',')] for line in lines[arm * 13:(arm + 1) * 13]]
        prefix = f'R.A{arm}'
        if ini.getint(prefix + '.BASIC', 'Dof') != 7 or ini.getint(prefix + '.BASIC', 'Type') != int(model[0][0]):
            raise ValueError(f'{prefix}: 控制器机型或自由度不匹配')
        for joint in range(8):
            section = prefix + (f'.L{joint}.DH' if joint < 7 else '.FLANGE')
            dh = [ini.getfloat(section, key) for key in ('Alpha', 'A', 'D', 'Theta')]
            if not np.allclose(dh, model[joint + 1][:4], atol=1e-5, rtol=0):
                raise ValueError(f'{prefix} J{joint + 1}: 控制器 DH 与模型不匹配')
            if joint < 7:
                bounds = [ini.getfloat(prefix + f'.L{joint}.BASIC', key) for key in ('LimitNeg', 'LimitPos')]
                if not np.allclose(sorted(bounds), sorted(model[joint + 1][4:6]), atol=1e-5, rtol=0):
                    raise ValueError(f'{prefix} J{joint + 1}: 控制器关节限位与模型不匹配')
        for index, face in enumerate(('PP', 'NP', 'NN', 'PN')):
            bounds = [ini.getfloat(prefix + '.CTRL', f'BD67{face}{j}') for j in range(3)]
            if not np.allclose(bounds, model[9 + index], atol=1e-4, rtol=0):
                raise ValueError(f'{prefix}: J6/J7 耦合限位与模型不匹配')
    names = [ini.get(section, 'Name') for section in ini.sections() if ini.has_option(section, 'Name')]
    return {'controller_name': names[0] if names else None, 'model_verified': config['model']}


def controller_model_info(robot, config):
    with tempfile.TemporaryDirectory(prefix='quest-controller-read-') as folder:
        path = Path(folder) / 'robot.ini'
        if not robot.receive_file(str(path), '/home/FUSION/Config/cfg/robot.ini'):
            raise RuntimeError('无法只读获取控制器模型配置；未使能')
        return compare_model(path, config)
