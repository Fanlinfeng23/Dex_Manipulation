import {ArmScene} from './scene.js';
import {Quaternion,Euler} from 'three';
import {InputTransport, POSE_SEND_HZ, POSE_INTERVAL_MS} from './transport.js';

const $=id=>document.getElementById(id);
let state=null;
function updateModeLabel(data){
  const hardware=!!data?.hardware;
  const wrist=data?.motion_scope==='wrist';
  $('mode-label').textContent=hardware?(wrist?'实机 · 仅手腕':'天机实机遥操'):(wrist?'预演 · 仅手腕':'虚拟预演');
  $('scope-note').textContent=hardware
    ? `天机实机 · ${data.hardware_fault?'故障已锁存':data.hardware_enabling?'正在使能':data.hardware_enabled?'伺服就绪':'待机'} · ${wrist?'J1–J5 固定；J6/J7 ±'+data.wrist_limits.excursion_deg+'°，'+data.wrist_limits.max_speed_deg_s+'°/s；只转动手柄，忽略平移':'模型显示实测关节 · 未启用碰撞检测'}`
    : `${document.body.dataset.modelVisual==='urdf'?'URDF 外形':'简化连杆'} · 安装面间距 98 mm · 未启用碰撞检测 · 不连接实机`;
}
let scene;
try{scene=new ArmScene($('scene'),(message,kind)=>{
  $('model-status').textContent=message;document.body.dataset.modelVisual=kind;
  $('model-status').classList.toggle('error',kind==='error');
  updateModeLabel(state);
});}catch(error){$('scene-error').hidden=false;$('scene-error').textContent='无法初始化 WebGL，请使用支持硬件加速的浏览器。'+error.message;}
let socket,transport,owner=false,side='left',seq=0,session=crypto.randomUUID();
let demoStart=null,demoPhase=0,toastTimer,lastResetRevision=null;
let followControl=false,followPending=null,lastUiFollow=null,lastPromptKey=null;
const hands={left:{position:[0,0,0],rotation:[0,0,0],grip:false},right:{position:[0,0,0],rotation:[0,0,0],grip:false}};
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4200);}
function command(action,value){
  const frame=['calibrate','reset'].includes(action)&&state?.source==='simulation'?buildFrame(performance.now()):undefined;
  if(transport?.command(action,value,frame))return transport.commands.at(-1)?.data.id;
}
function stopDemo(){demoStart=null;demoPhase=0;$('demo').textContent='▶ 轨迹演示';}
function renderFollowButton(data){
  const button=$('pause');
  button.textContent='Ⅱ 暂停并保持';
  button.setAttribute('aria-pressed',String(!!data?.following));
  button.disabled=!owner||socket?.readyState!==WebSocket.OPEN||!followControl||!!followPending||
    (!data?.enabled&&!data?.resetting)||!!data?.hardware_fault||!!data?.diagnostics?.worker_fault;
  button.title='取消双臂映射并保持当前实测位置；恢复前需重新校准';
}
function speak(message){
  if(!$('voice')?.checked||!('speechSynthesis' in window)||document.hidden)return;
  speechSynthesis.cancel();const utterance=new SpeechSynthesisUtterance(message);
  utterance.lang='zh-CN';utterance.rate=1.05;speechSynthesis.speak(utterance);
}
function operatorPrompt(data){
  const arms=data.arms||{};let key,text;
  if(data.diagnostics?.worker_fault||data.hardware_fault){key='fault';text=`故障：${data.hardware_fault||data.reason}`;}
  else if(data.input_backend==='xrobotoolkit'){
    if(!data.native_eligible||!data.fresh){key='input';text=`等待官方原生采集：${data.native_reason||data.reason}；机械臂保持`;}
    else if(!arms.left?.numeric_pose_gate||!arms.right?.numeric_pose_gate){key='tracking';text='双手柄数值位姿未通过门禁；机械臂保持';}
    else if(!data.enabled){key='calibrate';text='数值位姿已通过门禁；终端自动校准或按 A 校准';}
    else if(data.keyboard_follow&&data.following){key='native-following';text=data.native_joint?`原生联合遥操：双臂跟随，双手${data.hand_follow_gate?'门禁已开启':'等待同步'}；空格暂停`:'原生输入驱动双臂；终端空格可暂停';}
    else{key='native-ready';text='已校准；按终端空格开始双臂跟随';}
  }
  else if(data.keyboard_mode){
    if(data.tracking_loss_active||data.tracking_recovery_pending){key='tracking-recovery';text=data.reason;}
    else if(!data.fresh){key='input';text='等待 Quest 透视或 VR 采集：请在头显页面进入采集';}
    else if(!arms.left?.tracked||!arms.right?.tracked){key='tracking';text='请让头显相机持续看见左右手柄，等待真实 6DoF 追踪';}
    else if(!data.enabled){key='calibrate';text=data.calibration_revision>0?data.reason:'双手柄已追踪；联合终端会自动校准';}
    else if(data.keyboard_follow&&data.following){key='joint-following';text='双臂与灵巧手联合跟随中；终端空格可暂停';}
    else{key='joint-ready';text='已校准；按终端空格开始联合跟随';}
  }
  else if(!data.fresh){key='input';text='等待 Quest 透视或 VR 采集：请在头显页面进入采集';}
  else if(!arms.left?.tracked||!arms.right?.tracked){key='tracking';text='请拿起左右手柄，使两个手柄都显示“已追踪”';}
  else if(!data.enabled){
    const pressed=['left','right'].some(s=>(arms[s]?.raw_trigger??0)>.35);
    key=pressed?'release':'calibrate';text=pressed?'请完全松开左右食指扳机':'手柄已就绪：点击“校准双臂映射”，此操作不会移动机械臂';
  }else if(['left','right'].some(s=>arms[s]?.trigger_release_required)){
    key='release-after-calibration';text='校准完成：先完全松开扳机，再按住对应扳机接管机械臂';
  }else if(data.following){
    const active=['left','right'].filter(s=>arms[s]?.active).map(s=>s==='left'?'左臂':'右臂');
    key=`following-${active.join('-')}`;text=`${active.join('、')}跟随中；松开对应扳机即保持`;
  }else{key='ready';text='已校准：按住左扳机接管左臂，按住右扳机接管右臂';}
  $('next-step-text').textContent=text;$('next-step').dataset.kind=key==='fault'?'fault':data.following?'active':'ready';
  if(key!==lastPromptKey){lastPromptKey=key;speak(text);}
}
function connect(){
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/ui`);
  socket.addEventListener('close',()=>{$('trigger-status').textContent='扳机输入：连接已断开';});
  socket.onmessage=event=>{const data=JSON.parse(event.data);
    if(data.type==='command_ack'&&data.ok&&transport?.commands.some(c=>c.data.id===data.id&&['arm','calibrate'].includes(c.data.action))){
      clearTimeout(toastTimer);$('toast').hidden=true;
    }
    if(data.type==='command_ack'&&data.id===followPending)followPending=null;
    if(transport?.receive(data)){renderFollowButton(state);return;}
    if(data.type==='welcome'){owner=data.control;followControl=!!data.follow_control;renderFollowButton(state);if(!owner)toast('此窗口只读；请在启动终端使用键盘控制');}
    else if(data.type==='error'){toast(data.message);if(demoStart!==null){stopDemo();hands.left.grip=hands.right.grip=false;syncControls();}}
    else if(data.type==='state'){state=data;render(data);}};
  socket.onclose=()=>{transport?.close();transport=null;owner=false;state=null;followControl=false;followPending=null;lastUiFollow=null;renderFollowButton(null);stopDemo();hands.left.grip=hands.right.grip=false;syncControls();$('connection').innerHTML='<i class="dot"></i>服务已断开';$('state-label').textContent='连接中断';$('reason').textContent='请检查 Python 服务，正在尝试重连…';$('state-dot').className='status-dot';setTimeout(connect,1500);};
  socket.onopen=()=>{seq=0;session=crypto.randomUUID();lastResetRevision=null;
    transport=new InputTransport(socket,{session,onPause:message=>{
      followPending=null;stopDemo();hands.left.grip=hands.right.grip=false;syncControls();renderFollowButton(state);toast(message);
    },onNotice:message=>{followPending=null;renderFollowButton(state);toast(message);},onFault:message=>{
      followPending=null;stopDemo();hands.left.grip=hands.right.grip=false;syncControls();toast(message);}});};
}
for(let i=0;i<7;i++){
  const row=document.createElement('div');row.className='joint-row';row.innerHTML=`<span>J${i+1}</span><output id="left-j${i}">—</output><span>J${i+1}</span><output id="right-j${i}">—</output>`;$('joint-table').append(row);
}
for(const [kind,names,min,max,step] of [['position',['X','Y','Z'],-.4,.4,.002],['rotation',['Rx','Ry','Rz'],-60,60,1]]){
  names.forEach((name,i)=>{const row=document.createElement('label');row.className='slider-row';row.innerHTML=`<span>${name}</span><input aria-label="${kind==='position'?'位置':'旋转'} ${name}" id="${kind}-${i}" type="range" min="${min}" max="${max}" step="${step}" value="0"><output id="${kind}-out-${i}">0</output>`;$(kind+'-sliders').append(row);$(kind+'-'+i).oninput=event=>{stopDemo();hands[side][kind][i]=Number(event.target.value);syncControls();};});
}
function syncControls(){for(const kind of ['position','rotation'])hands[side][kind].forEach((value,i)=>{$(kind+'-'+i).value=value;$(kind+'-out-'+i).value=value.toFixed(kind==='position'?3:0);});$('grip').checked=hands[side].grip;$('left-side').className=side==='left'?'selected left':'';$('right-side').className=side==='right'?'selected right':'';document.querySelector('.sim-controls').classList.toggle('right',side==='right');}
function render(data){
  updateModeLabel(data);
  const triggerValue=s=>data.arms[s].raw_trigger==null?'—':`${Math.round(data.arms[s].raw_trigger*100)}%`;
  $('trigger-status').textContent=data.input_backend==='xrobotoolkit'
    ? `原生${data.native_joint?'联合':'双臂'}控制：终端 A 校准并继续 · B 暂停 · 空格暂停/恢复 · 光学有效性${data.native_tracking_validity==='reported'?'已上报':'未上报'}`
    : data.keyboard_mode
    ? '联合控制：终端空格暂停 / 继续 · 两侧 Quest 手柄与 Wuji Glove 同步门控'
    : `扳机输入：左 ${triggerValue('left')} · 右 ${triggerValue('right')}；跟随控制：${{keyboard:'键盘',web:'网页',controller:'手柄'}[data.trigger_source]??'请重启服务'}`;
  operatorPrompt(data);
  $('console-service').textContent=data.diagnostics?.worker_fault?'故障':'已连接';
  $('console-robot').textContent=!data.hardware?'虚拟预演':data.hardware_fault?'故障锁存':data.hardware_enabled?'state=3 阻抗就绪':data.hardware_enabling?'正在使能':'已连接·待机';
  $('console-quest').textContent=data.input_backend==='xrobotoolkit'
    ? !data.fresh?'未接收原生帧':data.native_eligible?`${data.input_hz.toFixed(0)} Hz · 数值门禁通过 · 光学有效性${data.native_tracking_validity==='reported'?'已上报':'未上报'}`:'原生门禁未通过'
    : !data.fresh?'未接收':Object.values(data.arms).every(a=>a.tracked)?`${data.input_hz.toFixed(0)} Hz·双手柄已追踪`:'已连接·追踪不全';
  $('console-calibration').textContent=data.enabled?'已校准':'未校准';
  for(const s of ['left','right']){
    const arm=data.arms[s],raw=arm.raw_trigger;
    $(s+'-hand-state').textContent=data.input_backend==='xrobotoolkit'
      ? !arm.numeric_pose_gate?'位姿门禁未通过':arm.active?'机械臂跟随中':'数值位姿可用'
      : !arm.tracked?'未追踪':arm.active?'机械臂跟随中':arm.trigger_release_required?'需先松扳机':'已追踪·保持';
    $(s+'-trigger-bar').style.width=`${Math.round((raw??0)*100)}%`;
    const poseError=arm.ik?.position_error_m==null?'':` · 末端误差 ${(arm.ik.position_error_m*1000).toFixed(1)} mm / ${(arm.ik.rotation_error_rad*180/Math.PI).toFixed(1)}°`;
    $(s+'-trigger-detail').textContent=`食指扳机 ${raw==null?'—':Math.round(raw*100)+'%'}${arm.active?' · 已接管':''}${poseError}`;
  }
  scene?.update(data);document.body.dataset.enabled=String(data.enabled);document.body.dataset.following=String(data.following);document.body.dataset.source=data.source;
  document.body.dataset.resetting=String(data.resetting);
  document.body.dataset.uiFollow=String(data.ui_follow);
  document.body.dataset.keyboardFollow=String(data.keyboard_follow);
  if(data.ui_follow!==lastUiFollow){
    if(data.source==='simulation'&&typeof data.ui_follow==='boolean'){
      hands.left.grip=hands.right.grip=data.ui_follow;syncControls();
    }else if(data.source==='simulation'&&!data.enabled&&lastUiFollow!==null){
      hands.left.grip=hands.right.grip=false;syncControls();
    }
    lastUiFollow=data.ui_follow;
  }
  document.body.dataset.calibrationRevision=String(data.calibration_revision);document.body.dataset.resetRevision=String(data.reset_revision);
  if(data.reset_revision!==lastResetRevision){scene?.clearTrails();lastResetRevision=data.reset_revision;}
  $('model').textContent=data.model_label;$('source').value=data.source;
  const limited=Object.values(data.arms).some(arm=>arm.ik?.limited);
  $('reason').textContent=data.reason;$('state-label').textContent=data.resetting?'恢复初始中':data.following?(limited?'受限跟随':data.input_backend==='xrobotoolkit'?'双臂跟随中':data.keyboard_mode?'联合跟随中':'扳机跟随中'):data.enabled?'已校准':'等待校准';
  $('state-dot').className='status-dot'+(data.following?' active':data.enabled?' ready':'');
  if(document.activeElement!==$('scale'))$('scale').value=data.scale;$('scale-value').value=data.scale.toFixed(2);
  $('simulation-panel').hidden=data.source!=='simulation';$('quest-panel').hidden=data.source!=='quest';
  $('arm').disabled=!owner||!data.fresh||data.resetting||!scene?.ready;renderFollowButton(data);$('reset').disabled=!owner||data.resetting;$('source').disabled=!owner||data.keyboard_mode;$('scale').disabled=!owner||data.keyboard_mode;$('demo').disabled=!owner||data.resetting||!scene?.ready;
  const simulationOption=$('source').querySelector('option[value="simulation"]');if(simulationOption)simulationOption.disabled=!!data.hardware;
  if(data.hardware){$('reset').disabled=true;$('demo').disabled=true;}
  if(data.motion_scope==='wrist'){$('demo').disabled=true;$('scale').disabled=true;}
  $('feedback-status').hidden=!data.hardware;
  if(data.hardware){
    const fb=data.hardware_feedback;
    $('feedback-status').textContent=data.hardware_fault?`实机故障：${data.hardware_fault}`:
      `实测反馈：${fb?fb.age_ms.map(v=>`${Math.round(v??0)} ms`).join(' / '):'等待'} · 最大关节偏差 ${Math.max(...Object.values(data.arms).map(a=>a.tracking_error_deg??0)).toFixed(1)}° · 发送缓冲重试 ${data.hardware_send_clear_retries??0}`;
  }
  if(data.hardware_fault){$('arm').disabled=true;$('pause').disabled=true;}
  if(data.diagnostics?.worker_fault)for(const id of ['arm','reset','source','scale','demo'])$(id).disabled=true;
  for(const s of ['left','right']){(data.arms[s].measured_joints_deg??data.arms[s].joints_deg).forEach((q,i)=>$(s+'-j'+i).value=q.toFixed(1));$(s+'-tracking').textContent=`${s==='left'?'左手':'右手'}：${data.input_backend==='xrobotoolkit'?(data.arms[s].numeric_pose_gate?'数值门禁通过':'数值门禁未通过')+(data.native_tracking_validity==='unreported'?' · 光学有效性未知':''):data.arms[s].tracked?(data.arms[s].active?'跟随':'追踪正常'):'无有效追踪'}`;}
  $('connection').innerHTML=`<i class="dot ${data.fresh?'green':''}"></i>${data.fresh?'输入已连接':'等待输入'}`;$('frequency').textContent=`${data.input_hz.toFixed(0)} Hz`;$('age').textContent=`帧龄 ${data.frame_age_ms===null?'—':Math.min(99999,Math.round(data.frame_age_ms))+' ms'}`;$('sequence').textContent=`帧 ${data.seq??'—'}`;
  $('frequency').title=`电脑接受的位姿帧率；发送上限 ${POSE_SEND_HZ} Hz，画面更新 25 Hz`;
  $('age').title=`估计额外延迟 ${data.queue_delay_ms??0} ms；收包后等待 ${data.receive_to_consume_ms??0} ms；控制步耗时 ${data.diagnostics?.control_tick_ms??0} ms`;
  if(demoPhase===2&&!data.enabled){stopDemo();hands.left.grip=hands.right.grip=false;syncControls();}
}
$('left-side').onclick=()=>{side='left';syncControls();};$('right-side').onclick=()=>{side='right';syncControls();};
$('grip').onchange=e=>{stopDemo();hands[side].grip=e.target.checked;if(state?.ui_follow!==null&&followControl)command('follow_manual');sample();};
$('arm').onclick=()=>command('calibrate','controller');
$('pause').onclick=()=>{
  if($('pause').disabled)return;
  stopDemo();hands.left.grip=hands.right.grip=false;syncControls();
  followPending=command('pause')??null;
  renderFollowButton(state);
};
$('voice').onchange=()=>{if(!$('voice').checked&&'speechSynthesis' in window)speechSynthesis.cancel();};
$('reset').onclick=()=>{stopDemo();for(const s of ['left','right'])hands[s]={position:[0,0,0],rotation:[0,0,0],grip:false};syncControls();scene?.clearTrails();command('reset');};
$('source').onchange=e=>{stopDemo();hands.left.grip=hands.right.grip=false;syncControls();command('source',e.target.value);};
$('scale').oninput=e=>$('scale-value').value=Number(e.target.value).toFixed(2);$('scale').onchange=e=>command('scale',Number(e.target.value));
$('view-reset').onclick=()=>scene?.resetView();$('sim-home').onclick=()=>{stopDemo();hands[side]={position:[0,0,0],rotation:[0,0,0],grip:false};syncControls();command('pause');};
$('demo').onclick=()=>{if(demoStart!==null){stopDemo();hands.left.grip=hands.right.grip=false;syncControls();command('pause');return;}command('pause');for(const s of ['left','right'])hands[s]={position:[0,0,0],rotation:[0,0,0],grip:false};demoStart=performance.now();demoPhase=0;$('demo').textContent='■ 停止演示';syncControls();};
function sample(){
  if(!owner||state?.source!=='simulation'||document.hidden)return;
  const now=performance.now();
  if(demoStart!==null){const elapsed=(now-demoStart)/1000;if(elapsed>.25&&demoPhase===0){command('calibrate');demoPhase=1;}
    if(demoPhase===1&&state?.enabled){demoPhase=2;demoStart=now;}
    if(demoPhase===2){const t=(now-demoStart)/1000;for(const [s,sign] of [['left',1],['right',-1]]){hands[s].grip=true;hands[s].position=[sign*.055*Math.sin(t*.55),.055*Math.sin(t*.7),-.045*(1-Math.cos(t*.6))];hands[s].rotation=[0,sign*4*Math.sin(t*.4),0];}syncControls();}}
  // Demo calibration may capture its own frame above. Timestamp this later
  // sample at construction so a higher seq never carries an older timestamp.
  transport?.sample(buildFrame(performance.now()));
}
function buildFrame(now){
  const controllers={};for(const [s,x] of [['left',-.25],['right',.25]]){const h=hands[s];const q=new Quaternion().setFromEuler(new Euler(...h.rotation.map(v=>v*Math.PI/180),'XYZ'));controllers[s]={position:[x+h.position[0],1.2+h.position[1],-.4+h.position[2]],quaternion:q.toArray(),tracked:true,grip:h.grip?1:0};}
  return {version:1,type:'frame',session,reference:0,seq:seq++,captured_ms:now,focused:!document.hidden,controllers};
}
document.addEventListener('visibilitychange',()=>{if(document.hidden&&owner&&state?.source==='simulation'){stopDemo();hands.left.grip=hands.right.grip=false;command('pause');syncControls();}});
window.addEventListener('keydown',e=>{if(e.code==='Escape'){stopDemo();command('pause');hands.left.grip=hands.right.grip=false;syncControls();}});
window.addEventListener('pagehide',()=>{if(owner)command('pause');transport?.close();socket?.close();});
syncControls();connect();setInterval(sample,POSE_INTERVAL_MS);
