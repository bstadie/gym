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


class ImPointEnv(mujoco_env.MujocoEnv, im_mujoco_env.ImMujocoEnv, utils.EzPickle):
    def __init__(self):
        im_mujoco_env.ImMujocoEnv.__init__(self)
        self.reward_function = self.standard_reward
        #self.reward_function = self.zero_reward
        self.cam_azs = [90.9, None]
        self.cam_elevations = [-3.3, None]
        self.cam_distance_multipliers = [1.1, 1.1]
        self.generation_index = 0
        mujoco_env.MujocoEnv.__init__(self, 'im_point.xml', 5)
        utils.EzPickle.__init__(self)

    def standard_reward(self, qpos):
        reward = np.abs(qpos[0]) + np.abs(qpos[1])
        #print reward
        return -reward

    def zero_reward(self, qpos):
        return 0

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ball_location = np.copy(self.get_body_com("torso"))
        ob = self._get_obs()
        reward = self.reward_function(ball_location)

        done = False
        return ob, reward, done, dict()

    def _get_obs(self):
        obs_vec = np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])
        self.render()
        im = self._get_viewer().get_image()
        self.store_image(im)
        return obs_vec

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 2
        #+ self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
        #+ self.np_random.randn(self.model.nv) * .01
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _get_viewer(self):
        if self.viewer is None or self.reset_viewer is True:
            self.viewer = mujoco_py.MjViewer(fixed_cam=True,
                                             cam_distance_multiplier=self.cam_distance_multipliers[self.generation_index],
                                             init_height=self.camera_height, init_width=self.camera_width,
                                             fixed_cam_coordinates=self.fixed_camera,
                                             cam_azenith=self.cam_azs[self.generation_index],
                                             cam_elevation=self.cam_elevations[self.generation_index])
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
            if self.reset_viewer is True:
                self.reset_viewer = False
        return self.viewer
