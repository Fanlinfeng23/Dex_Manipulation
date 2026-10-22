// Native WebXR coordinates and xr-standard gamepad mapping; no Unity conversion.
export function missingController() {
  return {position:[0,0,0], quaternion:[0,0,0,1], tracked:false, grip:0};
}

function sessionUsable(session) {
  // WebXR does not expose input-source poses while the session is blurred.
  return session.visibilityState === 'visible';
}

export function readHead(frame, session, referenceSpace) {
  const missing = {quaternion:[0,0,0,1], tracked:false};
  if (!sessionUsable(session)) return missing;
  const pose = frame.getViewerPose(referenceSpace);
  if (!pose || pose.emulatedPosition) return missing;
  const q = pose.transform.orientation;
  return {quaternion:[q.x,q.y,q.z,q.w], tracked:true};
}

export function readControllers(frame, session, referenceSpace) {
  const controllers = {left:missingController(), right:missingController()};
  const buttons = {calibrate:false, reset:false};
  const diagnostics = {left:'未连接手柄', right:'未连接手柄'};
  if (!sessionUsable(session)) {
    diagnostics.left = diagnostics.right = `会话${session.visibilityState}`;
    return {controllers, buttons, diagnostics};
  }
  for (const source of session.inputSources) {
    const side = source.handedness;
    // Hand-tracking input is deliberately excluded from this controller milestone.
    if (!(side in controllers) || source.hand ||
        source.gamepad?.mapping !== 'xr-standard') continue;
    diagnostics[side] = '缺少握持空间';
    // B can return the virtual arms home even when optical pose is unavailable.
    // A still requires two valid poses, enforced by the server calibration gate.
    if (side === 'right') {
      buttons.calibrate = !!source.gamepad.buttons[4]?.pressed;
      buttons.reset = !!source.gamepad.buttons[5]?.pressed;
    }
    if (!source.gripSpace) continue;
    const pose = frame.getPose(source.gripSpace, referenceSpace);
    if (!pose) {
      diagnostics[side] = '无光学位姿';
      continue;
    }
    if (pose.emulatedPosition) {
      diagnostics[side] = '仅估算位置';
      continue;
    }
    const {position:p, orientation:q} = pose.transform;
    diagnostics[side] = '真实 6DoF';
    controllers[side] = {position:[p.x,p.y,p.z], quaternion:[q.x,q.y,q.z,q.w],
      // Protocol v1 calls the follow signal "grip"; its source is the index trigger (button 0).
      tracked:true, grip:source.gamepad.buttons[0]?.value ?? 0};
  }
  return {controllers, buttons, diagnostics};
}

export function buttonCommand(buttons, previous) {
  // B takes priority; keeping either button down must not repeat a command.
  if (buttons.reset) return previous.reset ? null : 'reset';
  if (buttons.calibrate && !previous.calibrate) return 'calibrate';
  return null;
}
