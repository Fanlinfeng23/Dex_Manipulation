// Run with Node >= 18 after starting the Python service on TEST_URL (default :8765).
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {readControllers, readHead, buttonCommand} from '../web/xr-input.js';

const sources = ['left','right'].map(side=>({handedness:side,gripSpace:{side},
  gamepad:{mapping:'xr-standard',buttons:Array.from({length:6},()=>({value:0,pressed:false}))}}));
sources[1].gamepad.buttons[0].value=.8;sources[1].gamepad.buttons[1].value=.2;sources[1].gamepad.buttons[4].pressed=true;
const fakeSession={inputSources:sources,visibilityState:'visible'};
const pose={emulatedPosition:false,transform:{position:{x:1,y:2,z:3},orientation:{x:0,y:0,z:0,w:1}}};
const headSpace={kind:'same-local-floor-as-controllers'};
const headPose={...pose,transform:{...pose.transform,orientation:{x:0,y:-Math.sin(Math.PI/8),z:0,w:Math.cos(Math.PI/8)}}};
assert.deepEqual(readHead({getViewerPose:space=>{assert.equal(space,headSpace);return headPose;}},fakeSession,headSpace),
  {quaternion:[0,-Math.sin(Math.PI/8),0,Math.cos(Math.PI/8)],tracked:true});
for(const unavailable of [null,{...headPose,emulatedPosition:true}]){
  assert.equal(readHead({getViewerPose:()=>unavailable},fakeSession,headSpace).tracked,false);
}
assert.equal(readHead({getViewerPose:()=>{throw Error('Hidden viewer must not be sampled');}},
  {...fakeSession,visibilityState:'visible-blurred'},headSpace).tracked,false);
let sample=readControllers({getPose:()=>pose},fakeSession,{});
assert.deepEqual(sample.controllers.left.position,[1,2,3]);assert.equal(sample.controllers.right.grip,.8);assert.equal(sample.buttons.calibrate,true);
sources[1].gamepad.buttons[0].value=0;sources[1].gamepad.buttons[1].value=1;
assert.equal(readControllers({getPose:()=>pose},fakeSession,{}).controllers.right.grip,0); // Side squeeze cannot enable following.
sources[1].gamepad.buttons[5].pressed=true;
for (const badPose of [null,{...pose,emulatedPosition:true}]) {
  sample=readControllers({getPose:()=>badPose},fakeSession,{});
  assert.equal(sample.controllers.left.tracked,false);assert.equal(sample.buttons.reset,true);
}
sample=readControllers({getPose:()=>pose},{...fakeSession,visibilityState:'visible-blurred'},{});
assert.equal(sample.controllers.right.tracked,false);
assert.equal(sample.buttons.reset,false);
sample=readControllers({getPose:()=>pose},{...fakeSession,inputSources:[{...sources[0],hand:{}}]},{});
assert.equal(sample.controllers.left.tracked,false);
const released={calibrate:false,reset:false},a={calibrate:true,reset:false},b={calibrate:false,reset:true},both={calibrate:true,reset:true};
assert.equal(buttonCommand(a,released),'calibrate');assert.equal(buttonCommand(b,released),'reset');
assert.equal(buttonCommand(a,a),null);assert.equal(buttonCommand(b,b),null);
assert.equal(buttonCommand(both,released),'reset');assert.equal(buttonCommand(both,b),null);
assert.equal(buttonCommand(a,both),null); // A held across B release must not recalibrate.

const outputDir=process.env.TEST_OUTPUT_DIR || 'runtime';
await mkdir(outputDir,{recursive:true});
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL || 'chrome',
  headless:process.env.HEADLESS !== '0'});
const context=await browser.newContext({viewport:{width:1440,height:1050}});
const page=await context.newPage();const errors=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error'||(m.type()==='warning'&&m.text().includes('THREE.')))errors.push(m.text());});
const url=process.env.TEST_URL || 'http://127.0.0.1:8765';
const expectState=async(key,value)=>page.waitForFunction(([k,v])=>document.body.dataset[k]===String(v),[key,value]);
const angles=side=>page.locator(`[id^="${side}-j"]`).evaluateAll(nodes=>nodes.map(n=>Number(n.value)));
const slider=async(id,value)=>page.locator(id).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
try {
  await page.goto(url);await page.locator('#arm:enabled').waitFor();
  assert.equal(page.url(),new URL('/',url).href);assert.equal(await page.title(),'Quest 双臂实验室');
  await expectState('modelVisual','urdf');
  assert.ok((await page.locator('#model').textContent()).includes('M6S Lite'));
  assert.ok((await page.locator('#model-status').textContent()).includes('18'));
  await page.locator('#source').selectOption('simulation');
  const startupResetRevision=await page.evaluate(()=>Number(document.body.dataset.resetRevision));
  await page.locator('#reset').click();
  await expectState('resetRevision',startupResetRevision+1);await expectState('resetting',false);
  const initialLeft=await angles('left'), initialRight=await angles('right');
  await page.screenshot({path:join(outputDir,'desktop-home.png'),fullPage:true});
  assert.equal(await page.locator('#scene canvas').count(),1);
  assert.equal(await page.locator('#scene-error').isVisible(),false);
  assert.match(await page.locator('#pause').textContent(),/开始遥操/);
  assert.equal(await page.locator('#pause').isDisabled(),true);
  await page.locator('#arm').click();await expectState('enabled',true);
  const toggleRevision=await page.locator('body').getAttribute('data-calibration-revision');
  await page.locator('#pause').click();await expectState('following',true);
  assert.equal(await page.locator('#pause').getAttribute('aria-pressed'),'true');
  assert.match(await page.locator('#pause').textContent(),/暂停/);
  await slider('#position-2',-.03);await page.waitForTimeout(700);
  assert.notDeepEqual(await angles('left'),initialLeft);
  await page.screenshot({path:join(outputDir,'desktop-toggle-running.png'),fullPage:true});
  await page.locator('#pause').click();await expectState('following',false);await expectState('enabled',true);
  assert.match(await page.locator('#pause').textContent(),/开始遥操/);
  assert.equal(await page.locator('#pause').getAttribute('aria-pressed'),'false');
  const toggleHeld=await angles('left');
  await slider('#position-0',.12);await page.waitForTimeout(200);
  assert.deepEqual(await angles('left'),toggleHeld);
  await page.locator('#pause').click();await expectState('following',true);await page.waitForTimeout(200);
  assert.deepEqual(await angles('left'),toggleHeld);
  assert.equal(await page.locator('body').getAttribute('data-calibration-revision'),toggleRevision);
  await page.locator('#pause').click();await expectState('following',false);
  await page.screenshot({path:join(outputDir,'desktop-toggle-paused.png'),fullPage:true});
  await page.locator('#reset').click();await expectState('resetting',true);await expectState('resetting',false);
  // The original independent simulation checkboxes still work after button use.
  await page.locator('#arm').click();await expectState('enabled',true);
  await page.locator('#grip').check();await expectState('following',true);
  await slider('#position-2',-.04);await page.waitForTimeout(1200);
  assert.notDeepEqual(await angles('left'),initialLeft);assert.deepEqual(await angles('right'),initialRight);
  await page.locator('#grip').uncheck();await expectState('following',false);
  const heldLeft=await angles('left');await slider('#position-0',.10);await page.waitForTimeout(200);
  assert.deepEqual(await angles('left'),heldLeft);
  await page.locator('#grip').check();await expectState('following',true);await page.waitForTimeout(250);
  assert.deepEqual(await angles('left'),heldLeft);
  await page.locator('#grip').uncheck();await page.locator('#right-side').click();await page.locator('#grip').check();
  await page.waitForTimeout(150);await slider('#position-1',.03);await page.waitForTimeout(1000);
  assert.notDeepEqual(await angles('right'),initialRight);assert.deepEqual(await angles('left'),heldLeft);
  await page.keyboard.press('Escape');await expectState('enabled',false);
  await page.locator('#reset').click();await expectState('resetting',true);
  await page.waitForTimeout(200);assert.notDeepEqual(await angles('left'),initialLeft);
  assert.equal(await page.locator('#arm').isDisabled(),true);
  await page.screenshot({path:join(outputDir,'desktop-returning.png'),fullPage:true});
  await expectState('resetting',false);
  assert.deepEqual(await angles('left'),initialLeft);assert.deepEqual(await angles('right'),initialRight);
  await page.locator('#demo').click();await expectState('following',true);await page.waitForTimeout(4000);
  assert.equal(await page.locator('body').getAttribute('data-enabled'),'true',await page.locator('#reason').textContent());
  assert.notDeepEqual(await angles('left'),initialLeft);assert.notDeepEqual(await angles('right'),initialRight);
  await page.screenshot({path:join(outputDir,'desktop-following.png'),fullPage:true});
  await page.locator('#pause').click();await expectState('following',false);await expectState('enabled',true);
  await page.locator('#scale').evaluate(el=>{el.value='.5';el.dispatchEvent(new Event('change',{bubbles:true}));});
  await page.waitForFunction(()=>document.getElementById('scale-value').value==='0.50');
  await page.locator('#source').selectOption('quest');await expectState('source','quest');
  assert.equal(await page.locator('#quest-panel').isVisible(),true);
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:join(outputDir,'mobile.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  // Exercise the actual Quest websocket endpoint with synthetic WebXR-format frames.
  await page.evaluate(async()=>{
    const {buttonCommand}=await import('/static/xr-input.js');
    const {InputTransport}=await import('/static/transport.js');
    const ws=new WebSocket(`ws://${location.host}/ws/quest`);window.qaQuest=ws;
    await new Promise(resolve=>ws.onopen=resolve);
    window.qaFault=null;window.qaSent=[];window.qaPauseCount=0;window.qaHoldAckUntil=0;
    const input=new InputTransport(ws,{session:'browser-qa',onFault:reason=>window.qaFault=reason,
      onPause:()=>window.qaPauseCount++,
      onFrameSent:(_frame,sentAt)=>window.qaSent.push(sentAt)});window.qaTransport=input;
    ws.onmessage=event=>{
      const data=JSON.parse(event.data), delay=window.qaHoldAckUntil-performance.now();
      if(delay>0&&['frame_ack','command_ack'].includes(data.type))setTimeout(()=>input.receive(data),delay);
      else input.receive(data);
    };
    let seq=0;window.qaTracked=true;window.qaGrip=0;window.qaRightGrip=0;window.qaOffset=0;window.qaReference=0;
    window.qaButtons={calibrate:false,reset:false};let previousButtons=window.qaButtons;
    window.qaTimer=setInterval(()=>{
      const frame={version:1,type:'frame',session:'browser-qa',reference:window.qaReference,
      seq:seq++,captured_ms:performance.now(),focused:true,head:{quaternion:[0,0,0,1],tracked:true},controllers:{
        left:{position:[-.25,1.2,-.4+window.qaOffset],quaternion:[0,0,0,1],tracked:window.qaTracked,grip:window.qaGrip},
        right:{position:[.25,1.2,-.4],quaternion:[0,0,0,1],tracked:true,grip:window.qaRightGrip}}};
      input.sample(frame);
      const action=buttonCommand(window.qaButtons,previousButtons);
      if(action)input.command(action,undefined,frame);
      previousButtons=window.qaButtons;
    },1000/90);
  });
  await page.locator('#arm:enabled').waitFor();
  // Quest follow works directly from the desktop, then shares state with Space.
  await page.locator('#arm').click();await expectState('enabled',true);
  await page.locator('#pause').click();await expectState('following',true);
  await page.locator('#pause').click();await expectState('following',false);await expectState('enabled',true);
  await page.evaluate(async()=>{
    const ws=new WebSocket(`ws://${location.host}/ws/keyboard`);window.qaKeyboard=ws;
    const pending=new Map();
    window.qaKeyCommand=action=>new Promise((resolve,reject)=>{
      const id=crypto.randomUUID();pending.set(id,{resolve,reject});ws.send(JSON.stringify({type:'command',id,action}));
    });
    ws.onmessage=event=>{const data=JSON.parse(event.data);if(data.type==='command_ack'){
      const waiter=pending.get(data.id);if(waiter){pending.delete(data.id);data.ok?waiter.resolve(data):waiter.reject(Error(data.error));}
    }};
    await new Promise(resolve=>ws.onopen=resolve);
    await window.qaKeyCommand('keyboard_attach');
    window.qaKeyHeartbeat=setInterval(()=>window.qaKeyCommand('keyboard_heartbeat').catch(()=>{}),200);
  });
  await expectState('enabled',false);
  await page.locator('#arm').click();await expectState('enabled',true);
  await page.evaluate(()=>window.qaKeyCommand('keyboard_toggle'));await expectState('following',true);
  assert.match(await page.locator('#pause').textContent(),/暂停/);
  await page.locator('#pause').click();await expectState('keyboardFollow',false);await expectState('following',false);
  await page.locator('#pause').click();await expectState('keyboardFollow',true);await expectState('following',true);
  await page.evaluate(()=>window.qaKeyCommand('keyboard_toggle'));await expectState('following',false);
  assert.match(await page.locator('#pause').textContent(),/开始遥操/);
  await page.setViewportSize({width:390,height:844});
  await page.locator('#pause').click();await expectState('following',true);
  await page.locator('#pause').click();await expectState('following',false);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:join(outputDir,'mobile-toggle-paused.png'),fullPage:true});
  await page.evaluate(()=>{clearInterval(window.qaKeyHeartbeat);window.qaKeyboard.close();});
  await expectState('enabled',false);
  await page.setViewportSize({width:1440,height:1050});
  // A high browser-reported trigger must not prevent calibration or start an
  // arm by itself. The page shows raw values separately from software control.
  await page.evaluate(()=>{window.qaGrip=window.qaRightGrip=1;});
  await page.waitForFunction(()=>document.getElementById('trigger-status').textContent.includes('左 100% · 右 100%'));
  const highTriggerPose=await angles('left');
  await page.locator('#arm').click();await expectState('enabled',true);
  await page.waitForTimeout(250);await expectState('following',false);
  assert.deepEqual(await angles('left'),highTriggerPose);
  assert.match(await page.locator('#reason').textContent(),/点击.*开始遥操/);
  assert.match(await page.locator('#trigger-status').textContent(),/跟随控制：网页/);
  await page.screenshot({path:join(outputDir,'desktop-calibrated-high-trigger.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:join(outputDir,'mobile-calibrated-high-trigger.png'),fullPage:true});
  await page.locator('#pause').click();await expectState('following',true);
  assert.deepEqual(await angles('left'),highTriggerPose);
  assert.match(await page.locator('#trigger-status').textContent(),/跟随控制：网页/);
  await page.locator('#arm').click();
  await page.waitForFunction(()=>!document.getElementById('toast').hidden&&document.getElementById('toast').textContent.includes('先暂停跟随'));
  await page.locator('#pause').click();await expectState('following',false);
  await page.locator('#arm').click();
  await page.waitForFunction(()=>document.getElementById('toast').hidden);
  await page.evaluate(()=>{window.qaGrip=window.qaRightGrip=0;window.qaTransport.command('pause');});
  await expectState('enabled',false);
  await page.waitForFunction(()=>document.getElementById('trigger-status').textContent.includes('左 0% · 右 0%'));
  await page.setViewportSize({width:1440,height:1050});
  const revision=key=>page.evaluate(k=>Number(document.body.dataset[k]),key);
  const calibrationRevision=await revision('calibrationRevision'), resetRevision=await revision('resetRevision');
  const questBefore=await angles('left');
  await page.evaluate(()=>window.qaButtons={calibrate:true,reset:false});await expectState('enabled',true);
  await expectState('calibrationRevision',calibrationRevision+1);await page.waitForTimeout(200);
  assert.equal(await revision('calibrationRevision'),calibrationRevision+1); // Holding A fires once.
  assert.deepEqual(await angles('left'),questBefore);
  assert.equal(await page.locator('#arm').isEnabled(),true); // Recalibration remains available.
  await page.evaluate(()=>window.qaOffset=-.02);await page.waitForTimeout(150);
  assert.deepEqual(await angles('left'),questBefore); // A saves an anchor without moving the arm.
  await page.evaluate(()=>window.qaGrip=1);await expectState('following',true);
  await page.waitForTimeout(800); // First grip follows the displacement since A.
  assert.notDeepEqual(await angles('left'),questBefore);
  await page.evaluate(()=>{window.qaTracked=false;window.qaButtons={calibrate:true,reset:true};});
  await expectState('resetRevision',resetRevision+1);await expectState('enabled',false);await page.waitForTimeout(200);
  await expectState('resetting',true);assert.notDeepEqual(await angles('left'),initialLeft);
  await expectState('resetting',false);
  assert.deepEqual(await angles('left'),initialLeft);assert.deepEqual(await angles('right'),initialRight);
  assert.equal(await revision('resetRevision'),resetRevision+1); // B works without a pose and fires once.
  await page.evaluate(()=>{window.qaTracked=true;window.qaButtons={calibrate:true,reset:false};window.qaOffset=-.04;});
  await page.waitForTimeout(200);await expectState('enabled',false);
  assert.equal(await revision('calibrationRevision'),calibrationRevision+1); // Holding A across B cannot resume.
  assert.deepEqual(await angles('left'),initialLeft);
  await page.evaluate(()=>{window.qaGrip=0;window.qaButtons={calibrate:false,reset:false};});
  await page.waitForTimeout(100);
  await page.evaluate(()=>window.qaButtons={calibrate:true,reset:false});await expectState('enabled',true);
  await expectState('calibrationRevision',calibrationRevision+2);
  await page.evaluate(()=>window.qaGrip=1);await expectState('following',true);
  await page.evaluate(()=>window.qaOffset=-.06);await page.waitForTimeout(800);
  assert.notDeepEqual(await angles('left'),initialLeft);
  await page.evaluate(()=>window.qaTracked=false);await expectState('enabled',false);
  const lostPose=await angles('left');await page.evaluate(()=>{window.qaTracked=true;window.qaOffset=-.04;});await page.waitForTimeout(300);
  assert.deepEqual(await angles('left'),lostPose);await expectState('enabled',false);
  await page.evaluate(()=>window.qaGrip=0);await page.waitForTimeout(100);await page.locator('#arm').click();await expectState('enabled',true);
  await page.evaluate(()=>window.qaReference++);await expectState('enabled',false);
  await page.locator('#arm').click();await expectState('enabled',true);
  const transportResult=await page.evaluate(()=>({fault:window.qaFault,
    intervals:window.qaSent.slice(1).map((t,i)=>t-window.qaSent[i])}));
  assert.equal(transportResult.fault,null);
  assert.ok(transportResult.intervals.length>100);
  assert.ok(Math.min(...transportResult.intervals)>=1000/60-.1,
    `Pose sender exceeds 60 Hz: min interval ${Math.min(...transportResult.intervals)} ms`);
  // Simulate a transient 220 ms acknowledgment stall, including the pause ACK.
  // Capture must resume on the same socket; following must remain disabled.
  await page.locator('#pause').click();await expectState('following',true);
  const sentBeforeGap=await page.evaluate(()=>{
    window.qaBeforeGap=window.qaPauseCount;window.qaHoldAckUntil=performance.now()+220;
    return window.qaSent.length;
  });
  await page.waitForFunction(()=>window.qaPauseCount>window.qaBeforeGap);
  await expectState('enabled',false);
  await page.waitForFunction(count=>window.qaSent.length>count+5&&window.qaTransport.recovery===null,sentBeforeGap);
  const heldAfterGap=await angles('left');
  await page.evaluate(()=>window.qaOffset=-.08);await page.waitForTimeout(250);
  assert.deepEqual(await angles('left'),heldAfterGap);
  assert.equal(await page.evaluate(()=>window.qaQuest.readyState),1);
  assert.equal(await page.evaluate(()=>window.qaFault),null);
  assert.match(await page.locator('#reason').textContent(),/输入确认延迟/);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:join(outputDir,'mobile-input-recovered.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.setViewportSize({width:1440,height:1050});
  await page.screenshot({path:join(outputDir,'desktop-input-recovered.png'),fullPage:true});
  const recoveryArchive=await (await page.request.get(`${url}/api/trajectory`)).json();
  const gap=recoveryArchive.events.findLast(e=>e.type==='input_pause'&&e.source==='quest');
  assert.ok(gap);assert.equal(gap.cause,'ack_timeout');
  assert.ok(recoveryArchive.events.some(e=>e.type==='input_resumed'&&e.id===gap.id));
  assert.ok(recoveryArchive.samples.every(s=>typeof s.new_input==='boolean'&&typeof s.input_fresh==='boolean'));
  await page.locator('#arm').click();await expectState('enabled',true);
  await page.locator('#pause').click();await expectState('following',true);
  await page.evaluate(()=>{clearInterval(window.qaTimer);window.qaTransport.close();window.qaQuest.close();});await expectState('enabled',false);
  assert.equal(await page.locator('a[href="/api/trajectory"]').isVisible(),true);
  const archive=await (await page.request.get(`${url}/api/trajectory`)).json();
  const calibrated=archive.samples.find(s=>s.state.source==='quest'&&s.state.calibrated_heading_deg!==null);
  assert.ok(calibrated);assert.deepEqual(calibrated.input.head.quaternion,[0,0,0,1]);
  await page.goto(`${url}/quest`);await page.waitForFunction(()=>!document.getElementById('xr-status').textContent.includes('正在检查'));
  await page.waitForFunction(()=>document.getElementById('http-status').textContent.includes('已连接'));
  await page.waitForFunction(()=>document.getElementById('script-status').textContent.includes('已加载'));
  assert.equal(await page.locator('#enter-vr').isDisabled(),true);
  await page.screenshot({path:join(outputDir,'quest-page.png'),fullPage:true});
  // Restore convenient defaults for the next user session.
  await page.goto(url);await page.locator('#source:enabled').waitFor();await page.locator('#source').selectOption('simulation');
  await page.locator('#reset').click();
  await expectState('resetting',false);
  await page.locator('#scale').evaluate(el=>{el.value='.75';el.dispatchEvent(new Event('change',{bubbles:true}));});
  assert.deepEqual(errors,[]);
  // A missing STL must produce a visible error and disable the preview controls.
  await page.close();
  const missingMesh = await context.newPage();
  await missingMesh.route('**/robot-mesh/left-0-0',route=>route.fulfill({status:404,body:'missing mesh'}));
  await missingMesh.goto(url);
  await missingMesh.waitForFunction(()=>document.body.dataset.modelVisual==='error');
  assert.ok((await missingMesh.locator('#model-status').textContent()).includes('加载失败'));
  assert.equal(await missingMesh.locator('#demo').isDisabled(),true);
  await missingMesh.close();
  const brokenQuest=await context.newPage();
  await brokenQuest.route('**/static/quest.js',route=>route.fulfill({status:404,body:'missing module'}));
  await brokenQuest.goto(`${url}/quest`);
  await brokenQuest.waitForFunction(()=>document.getElementById('script-status').textContent.includes('加载失败'));
  await brokenQuest.waitForFunction(()=>document.getElementById('http-status').textContent.includes('已连接'));
  assert.equal(await brokenQuest.locator('#enter-vr').isDisabled(),true);await brokenQuest.close();
  console.log('PASS: high trigger A calibration without motion, explicit start, raw/source diagnostics, desktop/mobile, input recovery and existing A/B flows.');
} catch(error) {
  const details=await page.evaluate(()=>({state:{...document.body.dataset},
    reason:document.getElementById('reason')?.textContent,toast:document.getElementById('toast')?.textContent})).catch(()=>null);
  console.error('Browser failure context:',JSON.stringify(details));
  await writeFile(join(outputDir,'failure.json'),JSON.stringify({error:String(error),details},null,2));
  await page.screenshot({path:join(outputDir,'failure.png'),fullPage:true}).catch(()=>{});
  throw error;
} finally {await browser.close();}
