import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import im_mujoco_env
from gym import error

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class ReacherEnv(mujoco_env.MujocoEnv, im_mujoco_env.ImMujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        im_mujoco_env.ImMujocoEnv.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)

    def standard_reward(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        return [reward, reward_dist, reward_ctrl]

    def zero_reward(self, a):
        return [0, 0, 0]

    def _step(self, a):
        reward, reward_dist, reward_ctrl = self.reward_function(a)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2: break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        state_obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
        self.render()
        im = self._get_viewer().get_image()
        self.store_image(im)
        return state_obs

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(fixed_cam_coordinates=self.fixed_camera, fixed_cam=True,
                                             init_height=self.camera_height, init_width=self.camera_width)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer
