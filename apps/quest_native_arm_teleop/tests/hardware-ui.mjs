// This injects feedback-shaped UI state on a PREVIEW service. It never starts hardware.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
const url=process.env.TEST_URL||'http://127.0.0.1:8766';
const health=await (await fetch(url+'/api/health')).json();
assert.equal(health.hardware,false,'Run this test only against an isolated preview service');
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||'chrome',headless:process.env.HEADLESS!=='0'});
const page=await browser.newPage({viewport:{width:1440,height:1050}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
let fault=false, wrist=false, expectedJoint;
await page.routeWebSocket('**/ws/ui',ws=>{
  const server=ws.connectToServer();
  server.onMessage(message=>{
    const data=JSON.parse(message);
    if(data.type==='state'){
      data.hardware=true;data.virtual_only=false;data.hardware_connected=true;data.source='quest';data.fresh=true;
      data.hardware_feedback={age_ms:[3,4],serials:[100,101],states:[0,0]};
      data.hardware_fault=fault?'TEST ONLY: feedback timeout':null;
      if(wrist){data.motion_scope='wrist';data.wrist_limits={excursion_deg:3,max_speed_deg_s:2,locked_joints:[1,2,3,4,5]};}
      for(const side of ['left','right']){
        const a=data.arms[side];a.measured_joints_deg=a.joints_deg.slice();
        a.joints_deg=a.joints_deg.map(v=>v+5);a.measured_chain=a.chain;a.measured_pose=a.actual;a.tracking_error_deg=5;
      }
      expectedJoint=data.arms.left.measured_joints_deg[0].toFixed(1);
    }
    ws.send(JSON.stringify(data));
  });
  // Suppress outgoing input/commands; display-only test data is never applied.
  ws.onMessage(()=>{});
});
try{
  await page.goto(url);
  await page.waitForFunction(()=>document.body.dataset.modelVisual==='urdf',null,{polling:100});
  assert.equal(await page.locator('#mode-label').innerText(),'天机实机遥操');
  assert.equal(await page.locator('#reset').isDisabled(),true);
  assert.equal(await page.locator('#demo').isDisabled(),true);
  assert.equal(await page.locator('#source option[value="simulation"]').evaluate(el=>el.disabled),true);
  assert.equal(await page.locator('#left-j0').evaluate(el=>el.value),expectedJoint);
  assert.match(await page.locator('#feedback-status').innerText(),/5.0°/);
  wrist=true;
  await page.waitForFunction(()=>document.getElementById('mode-label').textContent==='实机 · 仅手腕');
  assert.match(await page.locator('#scope-note').innerText(),/J1–J5 固定.*±3°.*2°\/s.*忽略平移/);
  assert.equal(await page.locator('#scale').isDisabled(),true);
  assert.equal(await page.title(),'Quest 双臂实验室');
  assert.equal(await page.locator('#scene canvas').count(),1);
  assert.equal(await page.locator('#scene-error').isVisible(),false);
  if(process.env.TEST_OUTPUT_DIR){
    await mkdir(process.env.TEST_OUTPUT_DIR,{recursive:true});
    await page.screenshot({path:process.env.TEST_OUTPUT_DIR+'/wrist-mode-desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:process.env.TEST_OUTPUT_DIR+'/wrist-mode-mobile.png',fullPage:true});
  }
  fault=true;
  await page.waitForFunction(()=>document.getElementById('feedback-status').textContent.includes('TEST ONLY'),null,{polling:100});
  assert.equal(await page.locator('#arm').isDisabled(),true);
  assert.deepEqual(errors,[]);
  console.log('PASS: measured joints, feedback fault and locked simulation/home controls (mock feedback, no robot).');
}finally{await browser.close();}
