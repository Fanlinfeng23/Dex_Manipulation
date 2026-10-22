import assert from 'node:assert/strict';
import {InputTransport, POSE_SEND_HZ, POSE_INTERVAL_MS, RECOVERY_TIMEOUT_MS} from '../web/transport.js';
assert.equal(POSE_SEND_HZ,60);

function fixture(session = 'test-session', recovery = false) {
  const clock = {value:0};
  const socket = {readyState:1, bufferedAmount:0, messages:[], closed:false,
    send(text) {this.messages.push({at:clock.value, data:JSON.parse(text)});},
    close() {this.closed = true; this.readyState = 3;}};
  const faults = [], notices = [], pauses = [];
  let ids = 0, cleared = false;
  const transport = new InputTransport(socket, {session, now:() => clock.value,
    setTimer:() => 1, clearTimer:() => {cleared = true;}, makeId:() => `${session}-${ids++}`,
    onFault:message => faults.push(message), onNotice:message => notices.push(message),
    onPause:message => pauses.push(message)});
  transport.receive({type:'welcome', input_recovery:recovery});
  const frame = (seq, {grip=0, tracked=true, focused=true}={}) => ({version:1, type:'frame',
    session, reference:0, seq, captured_ms:clock.value, focused,
    controllers:Object.fromEntries(['left','right'].map(side => [side,
      {position:[seq,0,0], quaternion:[0,0,0,1], tracked, grip}]))});
  const ack = data => transport.receive({type:'frame_ack', session:data.session, seq:data.seq, accepted:true});
  const frames = () => socket.messages.filter(message => message.data.type === 'frame');
  return {clock,socket,transport,frame,ack,frames,faults,notices,pauses,cleared:() => cleared};
}

// Sample at 90 Hz and pump independently: at most 60 Hz, with original sample
// timestamps/sequence numbers. No resampling an old pose with a fresh timestamp.
{
  const f=fixture(); let seq=0, nextSample=0, newest;
  for(let step=0;step<=10000;step++) {
    const ms=step/10;
    f.clock.value=ms;
    if(ms>=nextSample) {newest=f.frame(seq++);f.transport.sample(newest);nextSample+=1000/90;}
    {
      f.transport.pump();
      const sent=f.frames().at(-1);
      if(sent?.at===ms) {assert.equal(sent.data.seq,newest.seq);assert.equal(sent.data.captured_ms,newest.captured_ms);f.ack(sent.data);}
    }
  }
  const frames=f.frames();
  assert.equal(frames.length,60); // 0.1 ms simulation steps round the interval up to 16.7 ms.
  for(let i=1;i<frames.length;i++) assert.ok(frames[i].at-frames[i-1].at>=POSE_INTERVAL_MS);
  f.clock.value=1001;f.transport.sample(f.frame(seq++));f.transport.pump();assert.equal(f.frames().length,60);
}

// Work performed by socket.send must not consume the next rate-limit interval.
{
  const f=fixture();const send=f.socket.send.bind(f.socket);
  f.socket.send=text=>{f.clock.value+=7;send(text);};
  f.transport.sample(f.frame(0));f.transport.pump();f.ack(f.frames()[0].data);
  f.clock.value=20;f.transport.sample(f.frame(1));f.transport.pump();
  assert.equal(f.frames().length,1);
  f.clock.value=27;f.transport.sample(f.frame(2));f.transport.pump();
  assert.equal(f.frames().length,2);
  assert.ok(f.frames()[1].at-f.frames()[0].at>=POSE_INTERVAL_MS);
}

// Only one pose in flight. Mismatched or older-session ACKs cannot unlock it.
{
  const f=fixture();f.transport.sample(f.frame(0));f.transport.pump();
  f.clock.value=20;f.transport.sample(f.frame(1));f.transport.pump();
  f.clock.value=40;f.transport.sample(f.frame(2));f.transport.pump();
  assert.equal(f.frames().length,1);
  f.transport.receive({type:'frame_ack',session:'old',seq:0,accepted:true});f.transport.pump();assert.equal(f.frames().length,1);
  f.ack(f.frames()[0].data);f.transport.pump();assert.equal(f.frames().length,2);
  assert.equal(f.frames()[1].data.seq,2);assert.equal(f.frames()[1].data.captured_ms,40);
}

// Overwritten samples still carry release and tracking-loss safety transitions.
{
  const f=fixture();f.transport.sample(f.frame(0,{grip:1}));f.transport.pump();
  f.clock.value=11;f.transport.sample(f.frame(1,{grip:0,tracked:false,focused:false}));
  f.clock.value=22;f.transport.sample(f.frame(2,{grip:1}));
  f.ack(f.frames()[0].data);f.transport.pump();
  assert.equal(f.frames()[1].data.seq,2);
  assert.deepEqual(f.frames()[1].data.safety,{released:['left','right'],tracking_lost:true,focus_lost:true});
  f.ack(f.frames()[1].data);f.clock.value=42;f.transport.sample(f.frame(3,{grip:1}));f.transport.pump();
  assert.deepEqual(f.frames()[2].data.safety,{released:[],tracking_lost:false,focus_lost:false});
}

// Backpressure retains only the newest sample and never builds a browser queue.
{
  const f=fixture();f.socket.bufferedAmount=4097;
  f.transport.sample(f.frame(0));f.transport.pump();
  f.clock.value=20;f.transport.sample(f.frame(1));f.transport.pump();assert.equal(f.socket.messages.length,0);
  f.clock.value=40;f.transport.sample(f.frame(2));f.socket.bufferedAmount=0;f.transport.pump();
  assert.equal(f.frames()[0].data.seq,2);assert.equal(f.frames()[0].data.captured_ms,40);
}

// A pose ACK timeout closes and clears everything. A late ACK cannot revive it.
for(const lateAck of [false,true]) {
  const f=fixture();f.transport.sample(f.frame(0));f.transport.pump();
  const sent=f.frames()[0].data;f.clock.value=100;f.transport.sample(f.frame(1));
  f.clock.value=151;
  if(lateAck)f.ack(sent);else f.transport.pump();
  assert.equal(f.socket.closed,true);assert.equal(f.faults.length,1);assert.equal(f.transport.latest,null);
  assert.equal(f.transport.inflight,null);assert.ok(f.cleared());
  f.ack(sent);f.transport.sample(f.frame(2));f.transport.pump();assert.equal(f.frames().length,1);
}

// A command uses its original capture time, ID and frame on every retry, and
// bypasses a waiting pose ACK. The acknowledged ID is never retried again.
{
  const f=fixture();const sample=f.frame(0);f.transport.sample(sample);f.transport.pump();
  f.clock.value=10;f.transport.command('calibrate',undefined,sample);
  f.clock.value=70;f.transport.pump();
  const commands=f.socket.messages.filter(message=>message.data.type==='command');
  assert.equal(commands.length,2);assert.deepEqual(commands[0].data,commands[1].data);
  assert.equal(commands[0].data.captured_ms,0);assert.deepEqual(commands[0].data.frame,
    {...sample,safety:{released:[],tracking_lost:false,focus_lost:false}});
  f.transport.receive({type:'command_ack',id:commands[0].data.id,ok:true});
  f.ack(sample);f.clock.value=140;f.transport.pump();
  assert.equal(f.socket.messages.filter(message=>message.data.type==='command').length,2);
}

// A/B include losses from overwritten samples, even when their current sample
// has recovered. A new healthy sample after the safety frame is sent can retry A.
for (const action of ['calibrate','reset']) {
  const f=fixture();f.transport.sample(f.frame(0,{grip:1}));f.transport.pump();
  f.clock.value=5;f.transport.sample(f.frame(1,{grip:0,focused:false}));
  f.clock.value=10;const recovered=f.frame(2);f.transport.sample(recovered);
  f.transport.command(action,undefined,recovered);
  const command=f.socket.messages.find(message=>message.data.type==='command').data;
  assert.equal(command.frame.focused,true);
  assert.deepEqual(command.frame.safety,{released:['left','right'],tracking_lost:true,focus_lost:true});
  f.transport.receive({type:'command_ack',id:command.id,ok:false,error:'sample contains loss'});
  f.ack(f.frames()[0].data);f.clock.value=20;f.transport.pump();
  assert.equal(f.frames()[1].data.safety.focus_lost,true);f.ack(f.frames()[1].data);
  f.clock.value=40;f.transport.command('calibrate',undefined,f.frame(3));
  const nextCommand=f.socket.messages.filter(message=>message.data.type==='command').at(-1).data;
  assert.deepEqual(nextCommand.frame.safety,{released:[],tracking_lost:false,focus_lost:false});
}

// Optical loss is distinct from focus loss so a pose-independent B return can
// continue through optical occlusion while a focus transition still stops it.
{
  const f=fixture();f.transport.sample(f.frame(0,{tracked:false}));
  f.transport.sample(f.frame(1));f.transport.command('reset',undefined,f.frame(2));
  const command=f.socket.messages.find(message=>message.data.type==='command').data;
  assert.deepEqual(command.frame.safety,{released:[],tracking_lost:true,focus_lost:false});
}

// Normal latest-frame coalescing / source switches are not actionable errors.
for(const reason of ['duplicate','out_of_order','superseded','inactive_source']) {
  const f=fixture();f.transport.sample(f.frame(0));f.transport.pump();
  f.transport.receive({type:'frame_ack',session:'test-session',seq:0,accepted:false,reason});
  assert.equal(f.transport.inflight,null);assert.deepEqual(f.notices,[]);
}

// Expired actions never get a fresh timestamp or replay after backpressure ends.
{
  const f=fixture();const sample=f.frame(0);f.socket.bufferedAmount=5000;
  f.clock.value=100;f.transport.command('reset',undefined,sample);
  f.clock.value=251;f.socket.bufferedAmount=0;f.transport.pump();
  assert.equal(f.socket.messages.length,0);assert.equal(f.transport.commands.length,0);assert.equal(f.notices.length,1);
}

// Pause preempts pending A/B and cannot disappear behind socket backpressure.
{
  const f=fixture();f.socket.bufferedAmount=5000;
  f.transport.command('calibrate',undefined,f.frame(0));f.transport.command('reset',undefined,f.frame(1));
  f.transport.sample(f.frame(2));f.transport.command('pause');
  assert.equal(f.transport.commands.length,1);assert.equal(f.transport.latest,null);
  assert.equal(f.transport.commands[0].data.action,'pause');
  f.clock.value=151;f.transport.pump();assert.equal(f.socket.closed,true);assert.equal(f.faults.length,1);
}

// Both queue overflow and old pose samples fail closed. Reconnection starts with
// a fresh transport and cannot replay either the old pose or old A/B commands.
{
  const f=fixture();f.socket.bufferedAmount=5000;
  for(let i=0;i<9;i++)f.transport.command('scale',i/10);
  assert.equal(f.socket.closed,true);assert.equal(f.transport.commands.length,0);
  const old=fixture('old');old.transport.sample(old.frame(0));old.socket.bufferedAmount=5000;old.transport.command('reset',undefined,old.frame(1));old.transport.close();
  const fresh=fixture('fresh');fresh.transport.pump();assert.equal(fresh.socket.messages.length,0);
  assert.equal(old.transport.latest,null);assert.equal(old.transport.commands.length,0);
  fresh.transport.sample(fresh.frame(0));fresh.clock.value=151;fresh.transport.pump();assert.equal(fresh.socket.closed,true);
}

// The desktop clutch stop preempts queued starts and closes a blocked connection.
{
  const f=fixture();f.socket.bufferedAmount=5000;
  f.transport.command('follow',true);f.transport.command('calibrate',undefined,f.frame(0));
  f.transport.command('follow',false);
  assert.equal(f.transport.commands.length,1);
  assert.equal(f.transport.commands[0].data.action,'follow');
  assert.equal(f.transport.commands[0].data.value,false);
  f.clock.value=151;f.transport.pump();
  assert.equal(f.socket.closed,true);assert.equal(f.faults.length,1);
}

// A 150 ms stall pauses over the same socket. Old ACKs, queued A/B and poses
// captured before the pause confirmation must never unlock/replay motion.
for (const trigger of ['timer', 'late_ack', 'stale_sample']) {
  const f=fixture('recover',true);
  f.transport.sample(f.frame(0));
  if(trigger!=='stale_sample')f.transport.pump();
  f.clock.value=151;
  if(trigger==='late_ack')f.ack(f.frames()[0].data);else f.transport.pump();
  assert.equal(f.socket.closed,false);assert.equal(f.pauses.length,1);
  const recovery=f.transport.recovery;
  assert.equal(recovery.data.action,'input_pause');
  assert.equal(recovery.data.value,trigger==='stale_sample'?'sample_expired':'ack_timeout');
  assert.equal(f.transport.inflight,null);assert.equal(f.transport.latest,null);
  assert.equal(f.transport.command('reset',undefined,f.frame(1)),false);
  const count=f.frames().length;
  f.clock.value=220;f.transport.sample(f.frame(2));f.transport.pump();
  f.ack({session:'recover',seq:0});
  f.transport.receive({type:'command_ack',id:'old-pause',ok:true});
  assert.equal(f.transport.recovery,recovery);assert.equal(f.frames().length,count);
  const retries=f.socket.messages.filter(m=>m.data.action==='input_pause');
  assert.equal(retries.length,2);assert.deepEqual(retries[0].data,retries[1].data);
  f.transport.receive({type:'command_ack',id:recovery.data.id,ok:true});
  assert.equal(f.transport.recovery,null);assert.equal(f.transport.latest,null);
  f.transport.pump();assert.equal(f.frames().length,count);
  f.clock.value=240;f.transport.sample(f.frame(3));f.transport.pump();
  assert.equal(f.frames().length,count+1);assert.equal(f.frames().at(-1).data.seq,3);
  f.ack({session:'recover',seq:0});assert.equal(f.transport.inflight.frame.seq,3);
  f.ack(f.frames().at(-1).data);assert.equal(f.transport.inflight,null);
  assert.deepEqual(f.faults,[]);assert.equal(f.socket.closed,false);
}

// No pause confirmation, a rejected pause or a confirmation after the hard
// deadline still disconnects. Continuous sampling cannot extend this deadline.
for (const result of ['missing', 'rejected', 'late']) {
  const f=fixture('recover',true);f.transport.sample(f.frame(0));f.transport.pump();
  f.clock.value=151;f.transport.pump();const id=f.transport.recovery.data.id;
  f.clock.value=result==='rejected'?180:152+RECOVERY_TIMEOUT_MS;
  f.transport.sample(f.frame(1));
  if(result==='missing')f.transport.pump();
  else f.transport.receive({type:'command_ack',id,ok:result==='late'});
  assert.equal(f.socket.closed,true);assert.equal(f.faults.length,1);
  f.transport.receive({type:'command_ack',id,ok:true});f.transport.pump();
  assert.equal(f.frames().length,1);assert.equal(f.transport.latest,null);
}

// User stops retain their tighter timeout even on a recovery-capable server.
for (const action of ['pause', 'follow']) {
  const f=fixture('recover',true);f.transport.sample(f.frame(0));f.transport.pump();
  f.transport.command(action,false);
  f.clock.value=151;f.transport.pump();
  assert.equal(f.socket.closed,true);assert.equal(f.pauses.length,0);
}

// Recovery clears queued A/B; safety edges omitted while waiting remain latched.
{
  const f=fixture('recover',true);f.transport.sample(f.frame(0,{grip:1}));f.transport.pump();
  f.clock.value=100;f.transport.command('reset',undefined,f.frame(1));
  f.clock.value=151;f.transport.pump();assert.equal(f.transport.commands.length,0);
  f.clock.value=160;f.transport.sample(f.frame(2,{focused:false}));
  f.clock.value=170;f.transport.sample(f.frame(3));
  f.transport.receive({type:'command_ack',id:f.transport.recovery.data.id,ok:true});
  f.clock.value=190;f.transport.sample(f.frame(4));f.transport.pump();
  assert.equal(f.frames().at(-1).data.safety.focus_lost,true);
  assert.equal(f.socket.messages.filter(m=>m.data.action==='reset').length,1);
}

console.log('PASS: 60 Hz cap, ACK gates, same-socket pause/recovery, fresh-only resume, hard timeout, command priority/TTL, safety latches and legacy fallback.');
