import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from quest_teleop.config import load_config, SDK_ROOT
from quest_teleop.kinematics import Kinematics
from quest_teleop.math3d import rigid_inverse, rotation_angle
from quest_teleop.urdf_model import VISUAL_ID, SDK_MODEL


@unittest.skipUnless((SDK_ROOT/'SDK_PYTHON'/('libKine.dll' if sys.platform == 'win32' else 'libKine.so')).exists(),
                     'Platform vendor SDK required')
class UrdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.kine = Kinematics(cls.config)

    def test_lite_assets_and_configuration(self):
        self.assertEqual(self.config['model'], SDK_MODEL)
        self.assertEqual(self.config['visual_model'], VISUAL_ID)
        visual = self.kine.visual
        self.assertEqual(len(visual.meshes), 18)
        for side in ('left', 'right'):
            data = visual.data['arms'][side]
            self.assertIn('M6S-Lite', data['urdf'])
            self.assertEqual([n['chain_index'] for n in data['links']], list(range(9)))
            for path in (p for key,p in visual.meshes.items() if key.startswith(side)):
                self.assertTrue(path.is_file())
                self.assertEqual(path.suffix.upper(), '.STL')

    def test_urdf_tcp_matches_sdk_across_joint_configurations(self):
        rng = np.random.default_rng(71)
        for side in ('left','right'):
            arm = self.kine.visual.arms[side]
            for _ in range(20):
                q = np.asarray(self.config['arms'][side]['initial_joints_deg']) + rng.uniform(-15,15,7)
                urdf_chain = [self.kine.arms[side]['base'] @ t for t in arm.chain(q)]
                for urdf, sdk in zip(urdf_chain, self.kine.chain(side,q), strict=True):
                    self.assertLess(np.linalg.norm(urdf[:3,3]-sdk[:3,3]), 5e-5)
                    self.assertLess(rotation_angle(urdf[:3,:3],sdk[:3,:3]),5e-5)
                flange = self.kine.fk(side,q)
                np.testing.assert_allclose(urdf_chain[-1], flange, atol=5e-5)
                # The terminal mesh is attached to the TCP, not to joint 7.
                relative = rigid_inverse(urdf_chain[-2]) @ urdf_chain[-1]
                np.testing.assert_allclose(relative[:3,3],[0,-.087,0],atol=1e-10)

    def test_wrong_arm_dimensions_are_rejected(self):
        context = SimpleNamespace(arms=copy.deepcopy({side:{'dh':arm['dh']} for side,arm in self.kine.arms.items()}))
        context.arms['left']['dh'][2][2] = 272  # Old M3 model has a different upper-arm length.
        with self.assertRaisesRegex(ValueError,'不匹配'):
            self.kine.visual.validate_frames('left',self.config,context)

    def test_missing_or_external_mesh_references_are_rejected(self):
        arm = self.kine.visual.arms['left']
        for uri in ('https://example.org/link.stl','package://arm/../../README.md','package://arm/meshes/missing.STL'):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                arm.mesh_path(uri)

    def test_joint_limits_match_stricter_vendor_bounds(self):
        for side in ('left','right'):
            limits = self.kine.arms[side]['limits']
            self.assertAlmostEqual(limits[1,0],-118,places=2)
            self.assertAlmostEqual(limits[3,1],45,places=2)
            self.assertAlmostEqual(limits[0,1],173,places=2)


if __name__ == '__main__': unittest.main()
