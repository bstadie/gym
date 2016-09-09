import numpy as np


class ImMujocoEnv:
    def __init__(self):
        self.reward_function = None

        self.image_history = None
        self.save_steps = 5000
        self.burn_in_steps = np.inf
        self.fixed_camera = [0.0, 0.0, 0.0]
        self.image_history_step = 0
        self.camera_width = 300
        self.camera_height = 300
        self.reset_viewer = False

        self.gan_discriminator = None

        self.save_dir = '/Users/TheMaster/Desktop/Current_Work/irl/experiments/two_link_arm/domain_one_failure'

    def zero_reward(self, qpos):
        return 0

    def gan_reward(self, qpos):
        pass

    def standard_reward(self, qpos):
        raise NotImplementedError('must be implemented in subclass')

    def store_image(self, im):
        self.image_history_step += 1
        true_step = self.image_history_step - self.burn_in_steps
        if true_step == self.save_steps:
                np.save(self.save_dir, self.image_history)
                print 'saved all data'

        elif 0 < true_step < self.save_steps:
            if self.image_history_step % self.save_steps/10 == 0:
                print 'saved image at step ' + str(self.image_history_step)
            data, width, height = im
            one_frame = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
            if self.image_history is None:
                self.image_history = np.zeros(shape=(self.save_steps, 2*self.camera_width, 2*self.camera_height, 3))
            self.image_history[true_step] = one_frame
