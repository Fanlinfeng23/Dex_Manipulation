import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';

export class ArmScene {
  constructor(container, onStatus = () => {}) {
    this.onStatus = onStatus;
    this.ready = false;
    this.lastState = null;
    this.model = null;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#edf0f4');
    this.scene.fog = new THREE.Fog('#edf0f4', 4.5, 12);
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 30);
    this.renderer = new THREE.WebGLRenderer({antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = .65;
    this.controls.maxDistance = 7;
    this.controls.maxPolarAngle = Math.PI * .495;
    this.resetView();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8794a8, 2.6));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(2, 5, 3);light.castShadow = true;
    light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-2;light.shadow.camera.right=2;
    light.shadow.camera.top=2;light.shadow.camera.bottom=-2;light.shadow.bias=-.0005;
    this.scene.add(light);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(30,30), new THREE.MeshStandardMaterial({color:0xecf0f4,roughness:.94}));
    floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;floor.position.y=-.003;this.scene.add(floor);
    const grid=new THREE.GridHelper(12,60,0xc6cdd6,0xd9dfe7);grid.material.transparent=true;grid.material.opacity=.55;this.scene.add(grid);
    this.robot = new THREE.Group();this.robot.rotation.x=-Math.PI/2;this.scene.add(this.robot);
    const metal=new THREE.MeshStandardMaterial({color:0xc4ccd3,metalness:.5,roughness:.4});
    this.box([.23,.23,.035],[0,0,.025],metal);
    // Unit-depth support meshes are resized from the live left/right Base
    // positions. Their outside faces therefore coincide with the mounting
    // flange planes instead of retaining the old 500 mm example width.
    this.mountColumn=this.box([.11,1,1.1],[0,0,.57],metal);
    this.mountBeam=this.box([.14,1,.13],[0,0,1.15],metal);
    this.arms={};
    for(const [side,color] of [['left',0xef741a],['right',0x2584d9]]) {
      const accent=new THREE.MeshStandardMaterial({color,metalness:.25,roughness:.38});
      const shell=new THREE.MeshStandardMaterial({color:0xe0e5ea,metalness:.55,roughness:.3});
      const joints=[],links=[];
      for(let i=0;i<9;i++){
        const joint=new THREE.Mesh(new THREE.SphereGeometry(i===8?.034:.043,24,16),i===0?metal:accent);
        joint.castShadow=true;this.robot.add(joint);joints.push(joint);
        if(i<8){const link=new THREE.Mesh(new THREE.CylinderGeometry(.031,.035,1,24),shell);link.castShadow=true;this.robot.add(link);links.push(link);}
      }
      const target=new THREE.AxesHelper(.13);this.robot.add(target);
      const actual=new THREE.AxesHelper(.065);this.robot.add(actual);
      const traceGeometry=new THREE.BufferGeometry();
      traceGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(400*3),3).setUsage(THREE.DynamicDrawUsage));
      traceGeometry.setDrawRange(0,0);
      const trace=new THREE.Line(traceGeometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:.5}));trace.frustumCulled=false;this.robot.add(trace);
      const gap=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color,dashSize:.018,gapSize:.012,transparent:true,opacity:.7}));this.robot.add(gap);
      this.arms[side]={joints,links,target,actual,trace,gap,history:[],visuals:null};
    }
    const origin=new THREE.AxesHelper(.18);origin.position.set(.6,-.65,.006);this.robot.add(origin);
    const resize=()=>{const r=container.getBoundingClientRect();if(!r.width||!r.height)return;this.renderer.setSize(r.width,r.height);this.camera.aspect=r.width/r.height;this.camera.updateProjectionMatrix();};
    new ResizeObserver(resize).observe(container);resize();
    this.renderer.setAnimationLoop(()=>{this.controls.update();this.renderer.render(this.scene,this.camera);});
    this.loadModel();
  }
  async loadModel() {
    this.onStatus('正在加载机器人模型…', 'loading');
    const geometries = [];
    const materials = [];
    try {
      const response = await fetch('/api/robot-model', {cache:'no-store'});
      if (!response.ok) throw new Error(`模型描述请求失败 (${response.status})`);
      const model = await response.json();
      this.model = model;
      if (model.kind === 'schematic') {
        this.ready = true; this.onStatus('简化连杆示例', 'schematic'); return;
      }
      if (model.kind !== 'urdf') throw new Error('未知模型格式');
      const loader = new STLLoader();
      const pending = {};
      const jobs = [];
      for (const [side, color] of [['left',0xe78236],['right',0x328bd0]]) {
        const shell = new THREE.MeshStandardMaterial({color:0xcfd7df,metalness:.3,roughness:.48});
        const accent = new THREE.MeshStandardMaterial({color,metalness:.2,roughness:.48});
        materials.push(shell,accent);
        pending[side] = model.arms[side].links.map(link => {
          const group = new THREE.Group();
          group.userData.chainIndex = link.chain_index;
          group.name = link.name;
          for (const visual of link.visuals) {
            jobs.push((async () => {
              const response = await fetch(visual.mesh_url);
              if (!response.ok) throw new Error(`${link.name} 网格读取失败 (${response.status})`);
              const geometry = loader.parse(await response.arrayBuffer());
              geometries.push(geometry);
              const material = [1,3,6].includes(link.chain_index) ? accent : shell;
              const mesh = new THREE.Mesh(geometry, material);
              mesh.position.set(...visual.origin.position);
              mesh.quaternion.set(...visual.origin.quaternion);
              mesh.scale.set(...visual.scale);
              mesh.castShadow = true;
              group.add(mesh);
            })());
          }
          return group;
        });
      }
      // Wait for all jobs even on failure so every allocated resource is disposed.
      const results = await Promise.allSettled(jobs);
      const failed = results.find(result => result.status === 'rejected');
      if (failed) throw failed.reason;
      for (const side of ['left','right']) {
        const arm = this.arms[side]; arm.visuals = pending[side];
        for (const mesh of [...arm.joints,...arm.links]) mesh.visible = false;
        for (const group of arm.visuals) this.robot.add(group);
      }
      this.ready = true;
      this.onStatus('URDF V1.0 · 18 个网格已加载', 'urdf');
      if (this.lastState) this.update(this.lastState);
    } catch (error) {
      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      this.ready = false;
      this.onStatus(`URDF 加载失败：${error.message}；请检查文件后刷新`, 'error');
    }
  }
  box(size,position,material){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),material);mesh.position.set(...position);mesh.castShadow=true;mesh.receiveShadow=true;this.robot.add(mesh);return mesh;}
  resetView(){this.camera.position.set(2.45,1.95,2.35);this.controls.target.set(.22,.85,0);this.controls.update();}
  clearTrails(){for(const arm of Object.values(this.arms)){arm.history=[];arm.trace.geometry.setDrawRange(0,0);}}
  update(state){
    this.lastState = state;
    if (this.model && (state.model !== this.model.model ||
        (this.model.kind === 'urdf' && state.visual_model !== this.model.id))) {
      this.ready = false;
      this.onStatus('服务模型已改变，请刷新页面重新加载', 'error');
      return;
    }
    const leftBase=state.arms.left.chain[0].position;
    const rightBase=state.arms.right.chain[0].position;
    const mountSpan=Math.max(Math.abs(leftBase[1]-rightBase[1]),.001);
    const mountMid=[(leftBase[0]+rightBase[0])*.5,(leftBase[1]+rightBase[1])*.5,
                    (leftBase[2]+rightBase[2])*.5];
    this.mountBeam.position.set(...mountMid);this.mountBeam.scale.y=mountSpan;
    this.mountColumn.position.x=mountMid[0];this.mountColumn.position.y=mountMid[1];
    this.mountColumn.scale.y=mountSpan;
    const axis=new THREE.Vector3(0,1,0);
    for(const side of ['left','right']){
      const raw=state.arms[side],arm=this.arms[side];
      const value=state.hardware&&raw.measured_chain&&raw.measured_pose
        ? {...raw,chain:raw.measured_chain,actual:raw.measured_pose}:raw;
      if (arm.visuals) {
        for (const group of arm.visuals) {
          const pose=value.chain[group.userData.chainIndex];
          group.position.set(...pose.position);group.quaternion.set(...pose.quaternion);
        }
      } else {
        const points=value.chain.map(p=>new THREE.Vector3(...p.position));
        points.forEach((p,i)=>arm.joints[i].position.copy(p));
        arm.links.forEach((link,i)=>{const dir=points[i+1].clone().sub(points[i]);const length=dir.length();link.visible=length>.002;if(link.visible){link.position.copy(points[i]).addScaledVector(dir,.5);link.scale.y=length;link.quaternion.setFromUnitVectors(axis,dir.normalize());}});
      }
      for(const key of ['target','actual']){arm[key].position.set(...value[key].position);arm[key].quaternion.set(...value[key].quaternion);}
      arm.gap.geometry.setFromPoints([new THREE.Vector3(...value.actual.position),new THREE.Vector3(...value.target.position)]);arm.gap.computeLineDistances();
      if(value.active){const p=new THREE.Vector3(...value.actual.position);if(!arm.history.length||p.distanceTo(arm.history.at(-1))>.001){
        arm.history.push(p);if(arm.history.length>400)arm.history.shift();
        const positions=arm.trace.geometry.getAttribute('position');
        arm.history.forEach((point,i)=>positions.setXYZ(i,point.x,point.y,point.z));
        positions.needsUpdate=true;arm.trace.geometry.setDrawRange(0,arm.history.length);
      }}
    }
  }
}
