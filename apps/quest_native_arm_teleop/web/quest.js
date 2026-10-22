import * as THREE from '/vendor/three/build/three.module.js';
import {readControllers, readHead, missingController, buttonCommand} from './xr-input.js';
import {InputTransport, POSE_SEND_HZ} from './transport.js';

const $ = id => document.getElementById(id);
let socket, transport, xrSession = null, xrMode = null, referenceSpace, sequence = 0, generation = 0;
let sessionId = crypto.randomUUID(), previousButtons = {calibrate:true,reset:true};
let state = null, connected = false, rejected = false, vrSupported = false, arSupported = false, renderer;
let inputError = '', errorUntil = 0;
let sentFrames = 0, lastDiagnostics = 0;
let inputDiagnostics = {left:'等待采集',right:'等待采集'};
// The WebXR spec withholds input-source poses while visible-blurred or hidden.
const xrSessionFocused = session => session?.visibilityState === 'visible';
$('quest-url').textContent = `${location.origin}/quest`;
function command(action, frame) { transport?.command(action, undefined, frame); }
function invalidFrame() {
  transport?.sample({version:1,type:'frame',session:sessionId,reference:generation,seq:sequence++,
    captured_ms:performance.now(),focused:false,
    controllers:{left:missingController(),right:missingController()}});
}
function refreshButtons() {
  $('enter-ar').disabled = !arSupported || !connected || !!xrSession || !renderer;
  $('enter-vr').disabled = !vrSupported || !connected || !!xrSession || !renderer;
  $('exit-vr').disabled = !xrSession;
}
function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/quest`);
  socket.onopen = () => {
    connected = true; sessionId = crypto.randomUUID(); sequence = 0;
    transport = new InputTransport(socket, {session:sessionId,
      onFrameSent:() => sentFrames++,
      onPause:message => {inputError = message; errorUntil = performance.now() + 6000;
        previousButtons = {calibrate:true,reset:true}; updatePanel();},
      onNotice:message => {inputError = message; errorUntil = performance.now() + 4000; updatePanel();},
      onFault:message => {inputError = message; errorUntil = performance.now() + 6000;
        connected = false; previousButtons = {calibrate:true,reset:true}; refreshButtons(); updatePanel();}});
    $('socket-status').textContent = '采集 WebSocket：已连接';
    previousButtons = {calibrate:true,reset:true}; // Require a fresh press after reconnect.
    refreshButtons();
  };
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === 'command_ack' && message.ok && transport?.commands.some(c =>
      c.data.id === message.id && ['arm','calibrate'].includes(c.data.action))) {
      inputError = ''; errorUntil = 0; updatePanel();
    }
    if (transport?.receive(message)) return;
    if (message.type === 'error') {
      $('xr-detail').textContent = message.message;
      inputError = message.message; errorUntil = performance.now() + 4000; updatePanel();
      if (message.message.includes('已有 Quest')) rejected = true;
    } else if (message.type === 'state') {
      state = message;
      $('server-status').textContent = message.source === 'quest'
        ? `电脑已选择 Quest · ${message.reason}` : '服务已连接；请在电脑预演页选择 Quest USB。';
      updatePanel();
    }
  };
  socket.onclose = () => {
    transport?.close(); transport = null;
    connected = false; state = null; refreshButtons(); updatePanel();
    $('socket-status').textContent = rejected ? '采集 WebSocket：已有其他 Quest 页面占用' : '采集 WebSocket：已断开，正在重连';
    $('server-status').textContent = rejected ? '已有采集页面连接；关闭旧页面后刷新。' : '电脑服务已断开，正在重连…';
    if (!rejected) setTimeout(connect,1500);
  };
}

const scene = new THREE.Scene(); scene.background = new THREE.Color('#eaf0f6');
const camera = new THREE.PerspectiveCamera(55, 2, .01, 30); camera.position.set(0,1.6,0);
scene.add(new THREE.HemisphereLight(0xffffff,0x8090a0,3));
const grid = new THREE.GridHelper(10,40,0x9aafc1,0xcdd8e3);
scene.add(grid);
const markers = {};
for (const [side,color] of [['left',0xed7a26],['right',0x3185d5]]) {
  const marker = new THREE.Mesh(new THREE.SphereGeometry(.035,16,12),new THREE.MeshStandardMaterial({color}));
  marker.visible = false; scene.add(marker); markers[side] = marker;
}
// A simple in-headset readout; detailed kinematic preview remains on the desktop.
const canvas = document.createElement('canvas'); canvas.width = 1024; canvas.height = 384;
const context = canvas.getContext('2d'); const texture = new THREE.CanvasTexture(canvas);
texture.colorSpace = THREE.SRGBColorSpace;
const panel = new THREE.Mesh(new THREE.PlaneGeometry(1.6,.6),new THREE.MeshBasicMaterial({map:texture}));
panel.position.set(0,1.55,-1.8); scene.add(panel);
function updatePanel() {
  context.fillStyle = '#ffffff'; context.fillRect(0,0,1024,384);
  context.fillStyle = '#26354a'; context.font = 'bold 46px sans-serif';
  const sideLabel = state?.control_sides?.length===1 ? ` · 仅${state.control_sides[0]==='right'?'右':'左'}臂` : '';
  context.fillText(`Quest 双臂实验室 · ${state?.hardware?'实机控制':'虚拟预演'}${state?.motion_scope==='wrist'?' · 仅手腕':sideLabel}`,48,78);
  context.font = '32px sans-serif';
  context.fillStyle = state?.following ? '#218865' : '#53657a';
  const status = !connected ? 'USB / 服务未连接' : performance.now() < errorUntil ? inputError : state?.source !== 'quest' ? '请在电脑选择 Quest USB' : state?.reason ?? '等待状态';
  context.fillText(status,48,152,930);
  const calibrationHint = state?.keyboard_mode
    ? '颈前固定头显，相机朝向双手 · 联合校准和状态见电脑终端'
    : state?.heading_source === 'fixed'
      ? '请让头显相机朝向双手 · 校准和状态见电脑终端'
      : '平视正前方按 A · 校准朝向与相对位姿';
  context.fillStyle = '#53657a';
  context.fillText(xrSession
    ? `左手：${inputDiagnostics.left} · 右手：${inputDiagnostics.right}`
    : calibrationHint,48,228,930);
  context.fillText(state?.keyboard_mode
    ? '联合模式：终端自动开始 / 空格暂停 · 丢踪后稳定追踪自动重接'
    : state?.motion_scope==='wrist'?'仅转动手柄 · J6/J7 ±3° · J1–J5 固定':state?.control_sides?.length===1?'仅右手柄/右扳机 · 各轴启动角 ±3° · B 暂停':state?.hardware?'先松开扳机 · 扳机跟随 · B 暂停双臂':'先松开扳机 · 扳机跟随 · B 恢复初始姿态',48,304,930); texture.needsUpdate = true;
  // In passthrough, show the in-headset status when setup or tracking needs
  // attention and keep the robot workspace clear while both arms follow.
  panel.visible = xrMode !== 'immersive-ar' || !state?.following;
}
updatePanel();
try {
  renderer = new THREE.WebGLRenderer({antialias:true,alpha:true}); renderer.xr.enabled = true;
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
  $('xr-canvas').append(renderer.domElement);
  new ResizeObserver(() => {
    if (xrSession) return;
    const {width,height} = $('xr-canvas').getBoundingClientRect();
    renderer.setSize(width,height); camera.aspect = width/height; camera.updateProjectionMatrix();
  }).observe($('xr-canvas'));
  renderer.setAnimationLoop((time,frame) => {
    if (frame && xrSession && referenceSpace) {
      const {controllers,buttons,diagnostics} = readControllers(frame,xrSession,referenceSpace);
      const head = readHead(frame,xrSession,referenceSpace);
      const sample = {version:1,type:'frame',session:sessionId,reference:generation,seq:sequence++,
        captured_ms:performance.now(),focused:xrSessionFocused(xrSession),controllers,head,buttons};
      transport?.sample(sample);
      if (time - lastDiagnostics > 250) {
        inputDiagnostics = diagnostics;
        const describe = side => `${side === 'left' ? '左' : '右'}手${diagnostics[side]} / 扳机 ${Math.round(controllers[side].grip * 100)}%`;
        $('frame-status').textContent = `会话 ${xrSession.visibilityState} · 发送上限 ${POSE_SEND_HZ} Hz · 已发送 ${sentFrames} 帧 · ${describe('left')} · ${describe('right')}`;
        lastDiagnostics = time;
        updatePanel();
      }
      for (const side of ['left','right']) {
        markers[side].visible = controllers[side].tracked;
        markers[side].position.set(...controllers[side].position);
      }
      // A/B carry their original button-press sample independently of pose ACKs.
      const action = buttonCommand(buttons, previousButtons);
      if (action) command(action === 'reset' && state?.hardware ? 'pause' : action, sample);
      previousButtons = buttons;
    }
    renderer.render(scene,camera);
  });
} catch (error) { $('xr-detail').textContent = `WebGL 初始化失败：${error.message}`; }

async function enterXR(mode) {
  $('enter-ar').disabled = $('enter-vr').disabled = true;
  try {
    // Keep requestSession directly under this real user click. Browser pages
    // cannot reliably enter immersive XR through an automatic script.
    const session = await navigator.xr.requestSession(mode,{optionalFeatures:['local-floor']});
    xrSession = session; xrMode = mode; generation++;
    scene.background = mode === 'immersive-ar' ? null : new THREE.Color('#eaf0f6');
    grid.visible = mode !== 'immersive-ar';
    panel.visible = mode !== 'immersive-ar' || !state?.following;
    previousButtons = {calibrate:true,reset:true};
    session.addEventListener('end', () => {
      delete document.documentElement.dataset.questXrActive;
      invalidFrame(); command('pause'); xrSession = null; xrMode = null; referenceSpace = null;
      scene.background = new THREE.Color('#eaf0f6');
      grid.visible = true;
      panel.visible = true;
      $('xr-status').textContent = '采集已结束'; refreshButtons();
      $('frame-status').textContent = '手柄采集：已结束';
      for (const marker of Object.values(markers)) marker.visible = false;
    });
    session.addEventListener('visibilitychange', () => {
      previousButtons = {calibrate:true,reset:true};
      $('frame-status').textContent = `会话状态：${session.visibilityState}；${session.visibilityState === 'visible' ? '等待新鲜手柄帧' : '输入已暂停'}`;
      if (!xrSessionFocused(session)) { invalidFrame(); command('pause'); }
    });
    session.addEventListener('inputsourceschange', () => {
      $('frame-status').textContent = '手柄输入源已变化；正在等待真实 6DoF 位姿';
    });
    let spaceType = 'local-floor';
    try { referenceSpace = await session.requestReferenceSpace(spaceType); }
    catch { spaceType = 'local'; referenceSpace = await session.requestReferenceSpace(spaceType); }
    referenceSpace.addEventListener('reset', () => {
      generation++; previousButtons = {calibrate:true,reset:true}; invalidFrame(); command('pause');
    });
    renderer.xr.setReferenceSpaceType(spaceType);
    await renderer.xr.setSession(session);
    renderer.xr.setReferenceSpace(referenceSpace);
    document.documentElement.dataset.questXrActive = mode;
    // This controls display/XR callbacks, not the headset's camera or IMU rate.
    if (typeof session.updateTargetFrameRate === 'function' &&
        Array.from(session.supportedFrameRates || []).includes(90)) {
      try { await session.updateTargetFrameRate(90); } catch { /* Keep the supported session default. */ }
    }
    $('xr-status').textContent = `${mode === 'immersive-ar' ? '透视' : 'VR'} 采集中${session.frameRate ? ` · ${session.frameRate} Hz 画面` : ''} · ${POSE_SEND_HZ} Hz 位姿发送`;
    $('xr-detail').textContent = state?.keyboard_mode
      ? '采集已启动；请把头显固定在颈前、相机朝向双手，让两只手柄持续可见。联合校准、开始和暂停状态见电脑终端。'
      : state?.heading_source === 'fixed'
        ? '采集已启动；请佩戴头显并朝向双手，让两只手柄持续可见。校准和状态见电脑操作台。'
        : '平视操作正前方、松开左右扳机键，按 A 校准朝向与相对位姿；方向保持固定，转头不会改变操控方向。';
  } catch (error) {
    delete document.documentElement.dataset.questXrActive;
    if (xrSession) await xrSession.end().catch(() => {});
    $('xr-status').textContent = '无法进入 XR'; $('xr-detail').textContent = error.message;
  }
  refreshButtons();
}
$('enter-ar').onclick = () => enterXR('immersive-ar');
$('enter-vr').onclick = () => enterXR('immersive-vr');
$('exit-vr').onclick = () => xrSession?.end();
$('quest-pause').onclick = () => command('pause');
window.addEventListener('pagehide', () => { invalidFrame(); command('pause'); transport?.close(); socket?.close(); });
async function checkSupport() {
  if (isSecureContext && typeof navigator.xr?.isSessionSupported === 'function') {
    [arSupported,vrSupported] = await Promise.all(['immersive-ar','immersive-vr'].map(
      mode => navigator.xr.isSessionSupported(mode).catch(() => false)));
  }
  $('xr-status').textContent = arSupported ? 'WebXR 透视已就绪' : vrSupported ? 'WebXR VR 已就绪；当前浏览器不支持透视' : '当前浏览器无法采集 Quest 手柄';
  if (!arSupported && !vrSupported) $('xr-detail').textContent = !isSecureContext ? '当前网址不是安全上下文。请通过 USB 转发，在 Quest 自带浏览器打开 http://127.0.0.1:端口/quest。' : !navigator.xr ? '当前浏览器没有 WebXR 接口。请使用更新后的 Quest 自带浏览器。' : '浏览器不支持沉浸式 XR 会话。请在 Quest 自带浏览器打开本页；电脑浏览器仅用于接入检查。';
  refreshButtons();
}
checkSupport().then(() => {
  if (arSupported || vrSupported) connect();
  else {
    $('server-status').textContent = '此处为接入说明；电脑预演页可使用模拟手柄。';
    $('socket-status').textContent = '采集 WebSocket：未启动（当前浏览器不支持 XR）';
  }
});
