// A bounded input channel. XR samples at the display rate; this channel sends
// only the newest original sample, at most 60 Hz, with one pose in flight.
export const POSE_SEND_HZ = 60;
export const POSE_INTERVAL_MS = 1000 / POSE_SEND_HZ;
export const INPUT_STALE_MS = 150;
export const RECOVERY_TIMEOUT_MS = 1000;

export class InputTransport {
  constructor(socket, {session, now = () => performance.now(),
    setTimer = (callback, ms) => globalThis.setTimeout(callback, ms),
    clearTimer = id => globalThis.clearTimeout(id), makeId = () => crypto.randomUUID(),
    onFault = () => {}, onPause = () => {}, onNotice = () => {}, onFrameSent = () => {}} = {}) {
    Object.assign(this, {socket, session, now, setTimer, clearTimer, makeId, onFault, onPause, onNotice, onFrameSent});
    this.latest = null; this.inflight = null; this.lastSentAt = -Infinity;
    this.canRecover = false; this.recovery = null;
    this.commands = []; this.released = new Set(); this.trackingLost = false; this.focusLost = false;
    this.previous = null; this.closed = false;
    this.schedule();
  }

  schedule() {
    if (this.closed) return;
    // Schedule relative to the actual last send. A fixed timer interval can
    // wake just early, skip a slot, and accidentally halve the effective rate.
    const remaining = POSE_INTERVAL_MS-(this.now()-this.lastSentAt);
    const delay = this.inflight || this.recovery ? 5 : remaining > 0 ? remaining : this.latest ? 1 : POSE_INTERVAL_MS;
    // Browser timers truncate fractional milliseconds. Round up so 60 Hz's
    // 16.67 ms deadline does not wake early and incur another timer interval.
    this.timer = this.setTimer(() => {this.pump();this.schedule();}, Math.ceil(delay));
  }

  sample(frame) {
    if (this.closed) return;
    // Inspect every XR sample, including samples overwritten before transmission.
    for (const side of ['left', 'right']) {
      if (this.previous?.controllers[side].grip > .35 && frame.controllers[side].grip <= .35) {
        this.released.add(side);
      }
    }
    if (!frame.focused || ['left', 'right'].some(side => !frame.controllers[side].tracked)) {
      this.trackingLost = true;
    }
    if (!frame.focused) this.focusLost = true;
    this.previous = frame;
    this.latest = frame;
  }

  command(action, value, frame) {
    if (this.closed) return false;
    // A pending pause is a barrier: no A/B/start can run before it is confirmed.
    if (this.recovery) {
      this.onNotice('正在确认暂停；连接恢复后请重新操作');
      return false;
    }
    const captured = frame?.captured_ms ?? this.now();
    if (frame && (!this.latest || frame.captured_ms >= this.latest.captured_ms)) this.sample(frame);
    if (action === 'pause' || (action === 'follow' && value === false)) {
      // Pause takes priority over any queued/retried A, B, source or scale action.
      this.commands = [];
      this.latest = null;
    }
    if (this.commands.length >= 8) {
      this.fail('操作指令排队过多，已暂停；请重新校准');
      return false;
    }
    const data = {type:'command', id:this.makeId(), session:this.session, captured_ms:captured, action};
    if (value !== undefined) data.value = value;
    if (frame !== undefined) data.frame = {...frame, safety:this.pendingSafety()};
    this.commands.push({data, lastSentAt:-Infinity, createdAt:this.now()});
    this.pump();
    return true;
  }

  receive(data) {
    if (this.closed) return false;
    if (data.type === 'welcome') {
      this.canRecover = data.input_recovery === true;
      return false; // The page also handles ownership/capabilities.
    }
    if (data.type === 'frame_ack') {
      if (this.inflight?.frame.session === data.session && this.inflight.frame.seq === data.seq) {
        if (this.now() - this.inflight.sentAt > INPUT_STALE_MS) {
          this.recover('ack_timeout');
          return true;
        }
        this.inflight = null;
        if (!data.accepted && data.reason &&
            !['duplicate','out_of_order','superseded','inactive_source'].includes(data.reason)) this.onNotice(data.reason);
      }
      return true;
    }
    if (data.type === 'command_ack') {
      if (this.recovery?.data.id === data.id) {
        if (!data.ok || this.now() - this.recovery.startedAt > RECOVERY_TIMEOUT_MS) {
          this.fail('输入暂停未及时确认，已断开；恢复后请重新校准');
        } else {
          this.recovery = null;
          // Require a sample captured after the pause ACK. Never replay data
          // collected while waiting, even if the old pose ACK arrives late.
          this.latest = this.previous = null;
          this.onNotice('连接已恢复；等待新采样，继续遥操请重新按 A 校准');
        }
        return true;
      }
      const index = this.commands.findIndex(command => command.data.id === data.id);
      if (index >= 0) {
        this.commands.splice(index, 1);
        if (!data.ok) this.onNotice(data.error || '操作未执行，请重新操作');
      }
      return true;
    }
    return false;
  }

  send(data) {
    if (this.socket.readyState !== 1 || this.socket.bufferedAmount > 4096) return false;
    try { this.socket.send(JSON.stringify(data)); return true; }
    catch { this.fail('输入连接发送失败，已暂停；请重新校准'); return false; }
  }

  pump() {
    if (this.closed) return;
    const now = this.now();
    if (this.recovery) {
      this.pumpRecovery();
      return;
    }
    if (this.inflight && now - this.inflight.sentAt > INPUT_STALE_MS) {
      this.recover('ack_timeout');
      return;
    }
    // Only one command at a time. Retry the same ID so the server can deduplicate.
    while (this.commands.length) {
      const command = this.commands[0];
      const stopping = command.data.action === 'pause' || (command.data.action === 'follow' && command.data.value === false);
      if (stopping && now - command.createdAt > 150) {
        this.fail('暂停指令未获确认，已断开输入连接');
        return;
      }
      if (now - command.data.captured_ms > 250) {
        this.commands.shift();
        this.onNotice('操作指令已过期，未重放；请重新操作');
        continue;
      }
      if (now - command.lastSentAt >= 60 && this.send(command.data)) command.lastSentAt = now;
      break;
    }
    if (this.closed || this.inflight || !this.latest || now - this.lastSentAt < POSE_INTERVAL_MS) return;
    if (now - this.latest.captured_ms > INPUT_STALE_MS) {
      this.recover('sample_expired');
      return;
    }
    const frame = {...this.latest, safety:this.pendingSafety()};
    if (this.send(frame)) {
      this.latest = null; this.released.clear(); this.trackingLost = false; this.focusLost = false;
      // Command serialization and socket.send above can take time. Start the
      // next rate-limit interval after the actual send, not before that work.
      const sentAt = this.now();
      this.inflight = {frame, sentAt}; this.lastSentAt = sentAt;
      this.onFrameSent(frame, sentAt);
    }
  }

  pendingSafety() {
    return {released:[...this.released], tracking_lost:this.trackingLost, focus_lost:this.focusLost};
  }

  recover(cause) {
    if (this.closed || this.recovery) return;
    // Keep the old fail-closed behavior with older servers, and never soften
    // an explicit operator stop whose confirmation is still pending.
    if (!this.canRecover || this.commands.some(({data}) =>
      data.action === 'pause' || (data.action === 'follow' && data.value === false))) {
      this.fail('输入确认或采样超过 150 ms，已断开；恢复后请重新校准');
      return;
    }
    const now = this.now();
    this.commands = [];
    this.latest = this.inflight = this.previous = null;
    this.recovery = {startedAt:now, lastSentAt:-Infinity, data:{type:'command',
      id:this.makeId(), session:this.session, captured_ms:now, action:'input_pause', value:cause}};
    this.pumpRecovery();
    if (!this.closed) this.onPause('输入短暂超时；正在确认暂停并恢复采集连接');
  }

  pumpRecovery() {
    const recovery = this.recovery;
    if (!recovery || this.closed) return;
    const now = this.now();
    if (now - recovery.startedAt > RECOVERY_TIMEOUT_MS) {
      this.fail('输入暂停超过 1 秒未获确认，已断开；请检查 USB / 电脑负载');
      return;
    }
    // Same ID on every retry; server deduplication prevents an old pause from
    // cancelling a later calibration. No poses are sent until this ACK arrives.
    if (now - recovery.lastSentAt >= 60 && this.send(recovery.data)) recovery.lastSentAt = now;
  }

  fail(reason) {
    if (this.closed) return;
    this.close();
    this.onFault(reason);
    // Closing also triggers the server watchdog/disconnect pause when a safety
    // message cannot get through. WebSocket.close preserves already sent order.
    this.socket.close(4000, 'input transport paused');
  }

  close() {
    this.closed = true;
    this.clearTimer(this.timer);
    this.latest = this.inflight = this.previous = null;
    this.recovery = null;
    this.commands = []; this.released.clear(); this.trackingLost = false; this.focusLost = false;
  }
}
