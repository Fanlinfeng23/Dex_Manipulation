import copy
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop.config import load_config, SDK_ROOT
from quest_teleop.engine import TeleopEngine
from quest_teleop.kinematics import Kinematics, IKError
from quest_teleop.continuous_ik import IKResult
from quest_teleop.math3d import XR_TO_WORLD, heading_basis, pose_matrix, relative_target, rotation_angle, smooth_target
from quest_teleop.protocol import parse_frame, ProtocolError


def packet(seq=0, captured_ms=10000):
    return dict(version=1, type='frame', session='test-session', reference=0, seq=seq,
                captured_ms=captured_ms, focused=True, head={'quaternion':[0,0,0,1], 'tracked':True}, controllers={
                    side:dict(position=[0,1.2,-.4], quaternion=[0,0,0,1], tracked=True, grip=0)
                    for side in ('left','right')})


class LinearKine:
    """Deterministic Cartesian fixture isolates state-machine behavior from vendor IK."""
    def __init__(self):
        self.arms = {s:{'base':np.eye(4)} for s in ('left','right')}
        self.fail_side = None

    def fk(self, side, q):
        return pose_matrix(np.asarray(q[:3]) / 1000, Rotation.from_euler('xyz',q[3:6],degrees=True).as_quat())

    def solve(self, side, target, reference):
        if side == self.fail_side:
            raise IKError('injected unreachable target')
        q = reference.copy()
        q[:3] = target[:3,3] * 1000
        q[3:6] = Rotation.from_matrix(target[:3,:3]).as_euler('xyz',degrees=True)
        return q

    def chain(self, side, q):
        return [self.fk(side,q)]

    def track(self, side, target, reference, max_delta_deg):
        return IKResult(self.solve(side, target, reference), True, 0., 0., 1)


class ProtocolTests(unittest.TestCase):
    def test_optional_head_orientation_is_validated(self):
        data=packet();data.pop('head')
        self.assertIsNone(parse_frame(data,10).head)
        data['head']={'quaternion':[0,0,0,1.01],'tracked':True}
        self.assertEqual(parse_frame(data,10).head.quaternion,[0,0,0,1])
        for head in [[], {'quaternion':[0,0,0,1],'tracked':1},
                     {'quaternion':[0,0,0,0],'tracked':True},
                     {'quaternion':[0,0,float('nan'),1],'tracked':True}]:
            with self.subTest(head=head), self.assertRaises(ProtocolError):
                data['head']=head;parse_frame(data,10)

    def test_valid_and_normalization(self):
        data=packet(); data['controllers']['left']['quaternion']=[0,0,0,1.01]
        self.assertEqual(parse_frame(data,10).controllers['left'].quaternion,[0,0,0,1])

    def test_malformed_frames(self):
        cases=[('version',True),('version',2),('seq',True),('seq',-1),('focused',1),
               ('captured_ms',float('nan')),('captured_ms',float('inf')),('session','')]
        for key,value in cases:
            with self.subTest(key=key,value=value), self.assertRaises(ProtocolError):
                data=packet();data[key]=value;parse_frame(data,10)
        for key,value in [('position',[0,0,float('nan')]),('position',[31,0,0]),
                          ('quaternion',[0,0,0,0]),('quaternion',[0,0,1]),
                          ('grip',True),('grip',1.1),('tracked',1)]:
            with self.subTest(key=key,value=value), self.assertRaises(ProtocolError):
                data=packet();data['controllers']['left'][key]=value;parse_frame(data,10)


class MappingTests(unittest.TestCase):
    def test_heading_preserves_forward_right_up_despite_room_yaw_pitch_and_roll(self):
        for yaw in (-180,-90,-45,0,45,90,179):
            for pitch,roll in [(0,0),(25,20),(-35,-15)]:
                with self.subTest(yaw=yaw,pitch=pitch,roll=roll):
                    facing=Rotation.from_euler('y',yaw,degrees=True)
                    head=facing*Rotation.from_euler('x',pitch,degrees=True)*Rotation.from_euler('z',roll,degrees=True)
                    basis,_=heading_basis(head.as_quat())
                    for direction,expected in [([0,0,-1],[1,0,0]),([1,0,0],[0,-1,0]),([0,1,0],[0,0,1])]:
                        np.testing.assert_allclose(basis@facing.apply(direction),expected,atol=1e-12)
                    np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(basis),1.)
        facing=Rotation.from_euler('y',-45,degrees=True)
        basis,_=heading_basis(facing.as_quat(),20)
        np.testing.assert_allclose(basis@facing.apply([0,0,-1]),
                                   Rotation.from_euler('z',20,degrees=True).apply([1,0,0]),atol=1e-12)

    def test_forward_left_up(self):
        anchor=np.eye(4)
        for xr,robot in [([0,0,-1],[1,0,0]),([-1,0,0],[0,1,0]),([0,1,0],[0,0,1])]:
            target=relative_target(pose_matrix(xr,[0,0,0,1]),anchor,anchor,XR_TO_WORLD,.35)
            np.testing.assert_allclose(target[:3,3],np.array(robot)*.35)
        self.assertAlmostEqual(np.linalg.det(XR_TO_WORLD),1)

    def test_up_axis_rotation_becomes_robot_yaw_without_position_orbit(self):
        anchor=pose_matrix([.4,.2,1.1],[0,0,0,1])
        controller=pose_matrix([0,0,0],Rotation.from_euler('y',30,degrees=True).as_quat())
        target=relative_target(controller,np.eye(4),anchor,XR_TO_WORLD,.35)
        np.testing.assert_allclose(target[:3,3],anchor[:3,3])
        np.testing.assert_allclose(target[:3,:3],Rotation.from_euler('z',30,degrees=True).as_matrix(),atol=1e-14)

    def test_mount_offset_accounts_for_lever_arm(self):
        offset=pose_matrix([.1,0,0],[0,0,0,1])
        controller=pose_matrix([0,0,0],Rotation.from_euler('y',90,degrees=True).as_quat())
        target=relative_target(controller@offset,offset,np.eye(4),XR_TO_WORLD,1)
        np.testing.assert_allclose(target[:3,3],[.1,.1,0],atol=1e-14)

    def test_filter_caps_translation_and_rotation(self):
        target=pose_matrix([1,0,0],Rotation.from_euler('z',90,degrees=True).as_quat())
        result=smooth_target(np.eye(4),target,.01,.001,.3,1)
        self.assertLessEqual(np.linalg.norm(result[:3,3]),.00300001)
        self.assertLessEqual(rotation_angle(result[:3,:3],np.eye(3)),.01000001)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.now=10.;self.data=packet();self.kine=LinearKine()
        self.engine=TeleopEngine(load_config(),self.kine,clock=lambda:self.now)
        self.feed();self.engine.arm()

    def feed(self, dt=.01):
        self.now+=dt;self.data['seq']+=1;self.data['captured_ms']=self.now*1000
        self.engine.receive(parse_frame(copy.deepcopy(self.data),self.now),'simulation')
        self.engine.step()

    def grip(self, side='left'):
        self.data['controllers'][side]['grip']=1;self.feed()

    def test_clutch_reengagement_never_jumps(self):
        original=self.engine.q['left'].copy();self.grip()
        np.testing.assert_allclose(self.engine.q['left'],original)
        self.data['controllers']['left']['position'][2]-=.04
        for _ in range(80):self.feed()
        self.assertGreater(self.engine.actual['left'][0,3],original[0]/1000+.01)
        self.data['controllers']['left']['grip']=0;self.feed()
        held=self.engine.q['left'].copy()
        self.data['controllers']['left']['position']=[1,1,1]
        self.feed();self.grip()
        np.testing.assert_allclose(self.engine.q['left'],held)
        self.assertTrue(self.engine.enabled)

    def test_duplicate_does_not_refresh_watchdog(self):
        frame=self.engine.frame;self.now+=.3
        self.assertFalse(self.engine.receive(parse_frame(self.data,self.now),'simulation'))
        self.assertIs(self.engine.frame,frame);self.engine.step()
        self.assertFalse(self.engine.enabled)

    def test_reordered_frame_is_ignored(self):
        self.data['seq']-=1
        self.assertFalse(self.engine.receive(parse_frame(self.data,self.now),'simulation'))

    def test_tracking_loss_latches_both_arms(self):
        self.grip();self.grip('right')
        self.data['controllers']['left']['tracked']=False;self.feed()
        held={s:q.copy() for s,q in self.engine.q.items()}
        self.assertFalse(self.engine.enabled)
        self.data['controllers']['left']['tracked']=True;self.feed()
        self.assertFalse(self.engine.enabled)
        self.engine.arm(); self.engine.step()
        self.assertFalse(any(self.engine.active.values()))
        self.assertTrue(all(self.engine.trigger_release_required.values()))
        for s in ('left','right'):
            np.testing.assert_allclose(self.engine.q[s],held[s]);self.data['controllers'][s]['grip']=0
        self.feed();self.engine.arm();self.assertTrue(self.engine.enabled)

    def test_recenter_session_focus_and_clock_loss_pause(self):
        for key,value in [('reference',1),('session','new-session'),('focused',False)]:
            with self.subTest(key=key):
                self.setUp();self.data[key]=value;self.feed();self.assertFalse(self.engine.enabled)
        self.setUp();self.data['seq']+=1;self.data['captured_ms']-=1
        self.engine.receive(parse_frame(self.data,self.now),'simulation')
        self.assertFalse(self.engine.enabled)

    def test_old_queued_data_cannot_reenable(self):
        self.data['seq']+=1;self.data['captured_ms']+=1
        self.now+=.2;self.engine.receive(parse_frame(self.data,self.now),'simulation')
        self.assertFalse(self.engine.enabled)
        with self.assertRaises(ValueError):self.engine.arm()
        self.feed();self.engine.arm();self.assertTrue(self.engine.enabled)

    def test_wrong_source_does_not_replace_frame(self):
        old=self.engine.frame
        self.assertFalse(self.engine.receive(parse_frame(self.data,self.now),'quest'))
        self.assertIs(self.engine.frame,old)

    def test_reselecting_current_source_preserves_clock_and_calibration(self):
        old=self.engine.frame;offset=self.engine.clock_offset
        self.engine.set_source('simulation')
        self.assertIs(self.engine.frame,old)
        self.assertEqual(self.engine.clock_offset,offset)
        self.assertTrue(self.engine.enabled)
        self.engine.set_source('quest')
        self.assertIsNone(self.engine.frame)
        self.assertIsNone(self.engine.clock_offset)
        self.assertFalse(self.engine.enabled)

    def test_jump_pauses_without_changing_joints(self):
        self.grip();old=self.engine.q['left'].copy()
        self.data['controllers']['left']['position'][0]+=1;self.feed()
        self.assertFalse(self.engine.enabled)
        np.testing.assert_allclose(self.engine.q['left'],old)

    def test_failed_right_ik_cannot_partially_commit_left(self):
        self.grip();self.grip('right');old={s:q.copy() for s,q in self.engine.q.items()}
        self.kine.fail_side='right'
        for c in self.data['controllers'].values():c['position'][2]-=.02
        self.feed();self.assertFalse(self.engine.enabled)
        for s in ('left','right'):np.testing.assert_allclose(self.engine.q[s],old[s])

    def test_joint_velocity_bound(self):
        self.grip();old=self.engine.q['left'].copy()
        self.data['controllers']['left']['position'][2]-=.1;self.feed()
        self.assertLessEqual(np.max(np.abs(self.engine.q['left']-old)),.900001)

    def test_a_calibrates_at_press_time_before_first_grip(self):
        initial = self.engine.actual['left'].copy()
        anchor = self.engine.anchors['left'][0].copy()
        self.data['controllers']['left']['position'][2] -= .04
        self.data['controllers']['left']['quaternion'] = Rotation.from_euler('y',5,degrees=True).as_quat().tolist()
        for _ in range(10): self.feed()
        # Moving with the grip released must preserve the A sample and hold the arm.
        np.testing.assert_allclose(self.engine.anchors['left'][0],anchor)
        np.testing.assert_allclose(self.engine.actual['left'],initial)
        self.grip()
        for _ in range(60): self.feed()
        self.assertTrue(self.engine.enabled,self.engine.reason)
        np.testing.assert_allclose(self.engine.target['left'][:3,3],initial[:3,3]+[.04*self.engine.scale,0,0],atol=1e-10)
        np.testing.assert_allclose(self.engine.target['left'][:3,:3],Rotation.from_euler('z',5,degrees=True).as_matrix()@initial[:3,:3],atol=1e-10)

    def test_a_recalibrates_current_robot_pose_without_moving(self):
        self.grip();self.data['controllers']['left']['position'][2]-=.04
        for _ in range(60): self.feed()
        self.data['controllers']['left']['grip']=0;self.feed()
        held = self.engine.actual['left'].copy()
        self.data['controllers']['left']['position']=[1,1.5,-1]
        self.data['controllers']['left']['quaternion']=Rotation.from_euler('y',20,degrees=True).as_quat().tolist()
        self.feed();revision=self.engine.calibration_revision;self.engine.calibrate()
        self.assertEqual(self.engine.calibration_revision,revision+1)
        np.testing.assert_allclose(self.engine.actual['left'],held)
        np.testing.assert_allclose(self.engine.target['left'],held)
        np.testing.assert_allclose(self.engine.anchors['left'][0][:3,3],[1,1.5,-1])
        self.grip();self.feed()
        np.testing.assert_allclose(self.engine.actual['left'],held,atol=1e-12)

    def test_b_restores_both_initial_poses_and_invalidates_calibration(self):
        self.grip();self.grip('right')
        for c in self.data['controllers'].values(): c['position'][2]-=.04
        for _ in range(60): self.feed()
        for side in ('left','right'):
            self.assertFalse(np.allclose(self.engine.q[side],self.engine.config['arms'][side]['initial_joints_deg']))
        # Homing this virtual model also works after pose tracking is lost.
        self.data['controllers']['right']['tracked']=False;self.feed()
        revision=self.engine.reset_revision;self.engine.reset()
        self.assertEqual(self.engine.reset_revision,revision+1)
        self.assertFalse(self.engine.enabled);self.assertEqual(self.engine.anchors,{})
        self.assertTrue(self.engine.snapshot()['resetting'])
        with self.assertRaisesRegex(ValueError,'正在恢复'): self.engine.calibrate()
        # Fresh frames without optical tracking must still allow the requested return.
        for _ in range(500):
            self.feed()
            if not self.engine.snapshot()['resetting']: break
        self.assertFalse(self.engine.snapshot()['resetting'])
        self.data['controllers']['right']['tracked']=True;self.feed()
        for side in ('left','right'):
            np.testing.assert_allclose(self.engine.q[side],self.engine.config['arms'][side]['initial_joints_deg'])
            np.testing.assert_allclose(self.engine.target[side],self.engine.actual[side])
        self.engine.calibrate()  # A succeeds, but held triggers cannot start motion.
        self.feed()
        self.assertFalse(any(self.engine.active.values()))
        self.assertTrue(all(self.engine.trigger_release_required.values()))
        for c in self.data['controllers'].values(): c['grip']=0
        self.feed();self.engine.calibrate();self.assertTrue(self.engine.enabled)

    def test_home_is_continuous_monotonic_synchronized_and_bounded(self):
        initial={s:np.array(self.engine.config['arms'][s]['initial_joints_deg']) for s in ('left','right')}
        # Exercise a large return, requiring longer than the configured minimum duration.
        for s,offset in [('left',80.),('right',-40.)]:
            self.engine.q[s]=initial[s]+offset
            self.engine.actual[s]=self.kine.fk(s,self.engine.q[s])
        before={s:q.copy() for s,q in self.engine.q.items()}
        self.engine.reset();started=self.now
        for s in initial: np.testing.assert_array_equal(self.engine.q[s],before[s])
        previous={s:q.copy() for s,q in before.items()};previous_velocity={s:np.zeros(7) for s in initial}
        for _ in range(1000):
            self.feed()
            progress=[]
            for s in initial:
                current=self.engine.q[s].copy()
                velocity=(current-previous[s])/.01
                self.assertLessEqual(float(np.max(np.abs(velocity))),30.+1e-6)
                self.assertLessEqual(float(np.max(np.abs(velocity-previous_velocity[s]))/.01),60.+1e-4)
                self.assertTrue(np.all(np.abs(current-initial[s])<=np.abs(previous[s]-initial[s])+1e-9))
                progress.append((current-before[s])/(initial[s]-before[s]))
                previous[s]=current;previous_velocity[s]=velocity
            np.testing.assert_allclose(progress[0],progress[1],atol=1e-12)
            if not self.engine.snapshot()['resetting']: break
        self.assertFalse(self.engine.snapshot()['resetting'])
        self.assertGreater(self.now-started,self.engine.config['reset_duration_s'])
        for s in initial: np.testing.assert_array_equal(self.engine.q[s],initial[s])

    def test_pause_and_timeout_abort_home_without_snapping_or_resuming(self):
        for interrupt in ('pause','timeout'):
            with self.subTest(interrupt=interrupt):
                self.setUp();self.grip();self.data['controllers']['left']['position'][2]-=.04
                for _ in range(60):self.feed()
                self.engine.reset()
                for _ in range(30):self.feed()
                held={s:q.copy() for s,q in self.engine.q.items()}
                if interrupt=='pause':self.engine.pause()
                else:self.now+=.3;self.engine.step()
                self.assertFalse(self.engine.snapshot()['resetting'])
                for _ in range(300):self.feed()
                for s in held:np.testing.assert_array_equal(self.engine.q[s],held[s])
                self.assertFalse(self.engine.enabled)

    def test_repeated_b_does_not_restart_return_and_slow_ticks_are_bounded(self):
        self.grip();self.data['controllers']['left']['position'][2]-=.04
        for _ in range(60):self.feed()
        self.engine.reset()
        for _ in range(30):self.feed()
        revision=self.engine.reset_revision;progress=self.engine.reset_progress
        before=self.engine.q['left'].copy();self.engine.reset()
        self.assertEqual(self.engine.reset_revision,revision)
        self.assertEqual(self.engine.reset_progress,progress)
        # Fresh frame after an event-loop stall; interpolation advances by at most 50 ms.
        self.feed(dt=.5)
        self.assertLessEqual(np.max(np.abs(self.engine.q['left']-before)),30.*.05+1e-9)


class QuestHeadingTests(unittest.TestCase):
    def setUp(self):
        self.now=10.;self.data=packet()
        self.engine=TeleopEngine(load_config(),LinearKine(),clock=lambda:self.now)
        self.engine.set_source('quest')

    def feed(self):
        self.now+=.01;self.data['seq']+=1;self.data['captured_ms']=self.now*1000
        self.engine.receive(parse_frame(copy.deepcopy(self.data),self.now),'quest');self.engine.step()

    def test_reported_diagonals_become_forward_and_right_after_a(self):
        # A synthetic -45deg room heading reproduces the two reported diagonals.
        # This is a regression fixture, not a measurement of the user's session.
        facing=Rotation.from_euler('y',-45,degrees=True)
        self.data['head']['quaternion']=facing.as_quat().tolist()
        for direction,expected in [([0,0,-1],[1,0,0]),([1,0,0],[0,-1,0]),([0,1,0],[0,0,1])]:
            for c in self.data['controllers'].values():c['grip']=0
            self.feed();before={s:t.copy() for s,t in self.engine.actual.items()}
            self.engine.calibrate()
            for s in before:np.testing.assert_array_equal(self.engine.actual[s],before[s])
            for c in self.data['controllers'].values():
                c['grip']=1;c['position']=(np.asarray(c['position'])+.04*facing.apply(direction)).tolist()
            self.feed()
            self.assertTrue(self.engine.enabled,self.engine.reason)
            for s in before:
                np.testing.assert_allclose(self.engine.target[s][:3,3]-before[s][:3,3],
                                           .04*self.engine.scale*np.asarray(expected),atol=1e-12)
        self.assertAlmostEqual(self.engine.calibrated_heading_deg,-45)

    def test_head_turns_do_not_remap_translation_or_rotation_until_new_a(self):
        facing=Rotation.from_euler('y',45,degrees=True)
        self.data['head']['quaternion']=facing.as_quat().tolist();self.feed();self.engine.calibrate()
        basis=self.engine.basis.copy();initial=self.engine.actual['left'].copy()
        self.data['head']['quaternion']=Rotation.from_euler('y',-80,degrees=True).as_quat().tolist()
        c=self.data['controllers']['left'];c['grip']=1
        c['position']=(np.asarray(c['position'])+.04*facing.apply([0,0,-1])).tolist()
        turn=facing*Rotation.from_euler('x',10,degrees=True)*facing.inv()
        c['quaternion']=turn.as_quat().tolist();self.feed()
        np.testing.assert_array_equal(self.engine.basis,basis)
        np.testing.assert_allclose(self.engine.target['left'][:3,3]-initial[:3,3],[.04*self.engine.scale,0,0],atol=1e-12)
        expected=Rotation.from_rotvec([0,-np.deg2rad(10),0]).as_matrix()@initial[:3,:3]
        np.testing.assert_allclose(self.engine.target['left'][:3,:3],expected,atol=1e-12)
        c['grip']=0;self.feed();held=self.engine.actual['left'].copy();self.engine.calibrate()
        self.assertAlmostEqual(self.engine.calibrated_heading_deg,-80)
        np.testing.assert_array_equal(self.engine.actual['left'],held)
        self.engine.set_source('simulation')
        self.engine.receive(parse_frame(packet(),self.now),'simulation');self.engine.calibrate()
        np.testing.assert_allclose(self.engine.basis,XR_TO_WORLD,atol=1e-12)

    def test_missing_untracked_or_vertical_head_cannot_partially_recalibrate(self):
        self.feed();self.engine.calibrate();revision=self.engine.calibration_revision
        basis=self.engine.basis.copy();anchors=self.engine.anchors
        for head in [None,{'quaternion':[0,0,0,1],'tracked':False},
                     {'quaternion':Rotation.from_euler('x',90,degrees=True).as_quat().tolist(),'tracked':True}]:
            with self.subTest(head=head):
                self.data['head']=head;self.feed()
                with self.assertRaises(ValueError):self.engine.calibrate()
                self.assertEqual(self.engine.calibration_revision,revision)
                self.assertIs(self.engine.anchors,anchors)
                np.testing.assert_array_equal(self.engine.basis,basis)
        self.engine.reset();self.assertIsNone(self.engine.calibrated_heading_deg)


@unittest.skipUnless((SDK_ROOT/'SDK_PYTHON'/('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform vendor SDK required')
class NativeKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config();cls.kine=Kinematics(cls.config)

    def test_vendor_roundtrip_and_rendered_chain(self):
        for side in ('left','right'):
            q=np.array(self.config['arms'][side]['initial_joints_deg'])
            for offset in (0,1,-1):
                probe=q+offset
                target=self.kine.fk(side,probe)
                np.testing.assert_allclose(self.kine.chain(side,probe)[-1],target,atol=1e-6)
                recovered=self.kine.fk(side,self.kine.solve(side,target,probe))
                np.testing.assert_allclose(recovered,target,atol=1e-5)

    def test_real_ik_tracks_small_left_motion_and_holds_right(self):
        now=10.;engine=TeleopEngine(self.config,self.kine,clock=lambda:now);data=packet()
        def feed():
            nonlocal now
            now+=1/90;data['seq']+=1;data['captured_ms']=now*1000
            engine.receive(parse_frame(copy.deepcopy(data),now),'simulation');engine.step()
        feed();engine.arm();data['controllers']['left']['grip']=1;feed()
        left=engine.actual['left'].copy();right=engine.q['right'].copy()
        for i in range(120):
            data['controllers']['left']['position'][2]=-.4-min(i/60,1)*.02;feed()
            self.assertTrue(engine.enabled,engine.reason)
        self.assertAlmostEqual(engine.actual['left'][0,3]-left[0,3],.02*engine.scale,delta=.0005)
        np.testing.assert_allclose(engine.q['right'],right)


if __name__=='__main__':unittest.main()
