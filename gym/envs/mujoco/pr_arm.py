import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

eps = 10**-7
def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class PR2ArmEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0,0,0])
        mujoco_env.MujocoEnv.__init__(self, 'pr_arm.xml', 5)
        utils.EzPickle.__init__(self)
        

    def set_target(self, target):
        self.target = np.array(target)

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               data.site_xpos.flat])

    def _step(self, a):
        
        data = self.model.data
        site = self.model.site_pose("lower_box")[0]
        end_effector = self.model.geom_pose("ee")[0]
        qpos = data.qpos
        reward = - np.linalg.norm(site - end_effector) - np.linalg.norm(site - self.target) - np.sum(qpos*qpos)
    
        done = bool(np.linalg.norm(site - end_effector) < eps)
        return self._get_obs(), reward, done, dict()

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
