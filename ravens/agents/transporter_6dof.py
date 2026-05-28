# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transporter Agent (hybrid 6-DoF / 3D placement baseline)."""

import os

import numpy as np
from ravens.agents.transporter import TransporterAgent
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.models.transport_6dof import TransportHybrid6DoF
from ravens.utils import utils
import tensorflow as tf
from transforms3d import quaternions


class Transporter6dAgent(TransporterAgent):
  """Transporter variant with planar pick and 6-DoF place prediction."""

  def __init__(self, name, task, root_dir='.', n_rotations=36):
    super().__init__(name, task, root_dir, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Transport(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)
    self.transport_6d = TransportHybrid6DoF(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)
    self.six_dof = True

  def _pose_to_matrix(self, pose):
    position, rotation = pose
    quat_wxyz = (rotation[3], rotation[0], rotation[1], rotation[2])
    transform = np.eye(4)
    transform[:3, :3] = quaternions.quat2mat(quat_wxyz)
    transform[:3, 3] = np.asarray(position)
    return transform

  def get_six_dof(self, transform_params, heightmap, pose0, pose1,
                  augment=True):
    """Adjust SE(3) labels after image-space augmentation."""

    if augment and transform_params is not None:
      t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
          *transform_params, heightmap, self.bounds, self.pix_size)
      t_worldnew_world = t_world_centernew @ np.linalg.inv(t_world_center)
    else:
      t_worldnew_world = np.eye(4)

    t_world_p1 = self._pose_to_matrix(pose1)
    t_worldnew_p1 = t_worldnew_world @ t_world_p1

    t_world_p0 = self._pose_to_matrix(pose0)
    t_worldnew_p0 = t_worldnew_world @ t_world_p0

    # Picking uses suction with rotational symmetry around the tool axis,
    # so we only regress the placing pose relative to a zero-yaw pick frame.
    t_worldnew_p0theta0 = t_worldnew_p0.copy()
    t_worldnew_p0theta0[:3, :3] = np.eye(3)

    t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
    t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

    quat_wxyz = quaternions.mat2quat(t_worldnew_p1theta0[:3, :3])
    quat_xyzw = (quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    p1_euler = utils.quatXYZW_to_eulerXYZ(quat_xyzw)

    p0_theta = 0.0
    p1_theta = -np.float32(p1_euler[2])
    z = np.float32(t_worldnew_p1theta0[2, 3])
    roll = np.float32(p1_euler[0])
    pitch = np.float32(p1_euler[1])
    return p0_theta, p1_theta, z, roll, pitch

  def get_sample(self, dataset, augment=True):
    (obs, act, _, _), _ = dataset.sample()
    img = self.get_image(obs)

    p0_xyz, p0_xyzw = act['pose0']
    p1_xyz, p1_xyzw = act['pose1']
    p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
    p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)

    if augment:
      img, _, (p0, p1), transform_params = utils.perturb(img, [p0, p1])
    else:
      transform_params = None

    p0_theta, p1_theta, z, roll, pitch = self.get_six_dof(
        transform_params, img[:, :, 3], (p0_xyz, p0_xyzw), (p1_xyz, p1_xyzw),
        augment=augment)

    return img, p0, p0_theta, p1, p1_theta, z, roll, pitch

  def train(self, dataset, writer=None):
    """Train on a dataset sample for 1 iteration."""
    tf.keras.backend.set_learning_phase(1)
    img, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample(dataset)

    step = self.total_steps + 1
    loss0 = self.attention.train(img, p0, p0_theta)
    loss1 = self.transport.train(img, p0, p1, p1_theta)
    loss2 = self.transport_6d.train(img, p0, p1, p1_theta, z, roll, pitch)

    if writer is not None:
      with writer.as_default():
        tf.summary.scalar('train_loss/attention', loss0, step)
        tf.summary.scalar('train_loss/transport', loss1, step)
        tf.summary.scalar('train_loss/transport_6d', loss2, step)
        tf.summary.scalar(
            'train_loss/transport_6d_z',
            self.transport_6d.z_metric.result(), step)
        tf.summary.scalar(
            'train_loss/transport_6d_roll',
            self.transport_6d.roll_metric.result(), step)
        tf.summary.scalar(
            'train_loss/transport_6d_pitch',
            self.transport_6d.pitch_metric.result(), step)

    print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f} {loss2:.4f}')
    self.total_steps = step

  def validate(self, dataset, writer=None):
    """Run lightweight validation without updating weights."""
    tf.keras.backend.set_learning_phase(0)
    n_iter = 10
    loss0, loss1, loss2 = 0, 0, 0
    for _ in range(n_iter):
      img, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample(
          dataset, augment=False)
      loss0 += self.attention.train(img, p0, p0_theta, backprop=False)
      loss1 += self.transport.train(img, p0, p1, p1_theta, backprop=False)
      loss2 += self.transport_6d.train(
          img, p0, p1, p1_theta, z, roll, pitch, backprop=False)

    loss0 /= n_iter
    loss1 /= n_iter
    loss2 /= n_iter

    if writer is not None:
      with writer.as_default():
        tf.summary.scalar('test_loss/attention', loss0, self.total_steps)
        tf.summary.scalar('test_loss/transport', loss1, self.total_steps)
        tf.summary.scalar('test_loss/transport_6d', loss2, self.total_steps)
    print(f'Validation Loss: {loss0:.4f} {loss1:.4f} {loss2:.4f}')

  def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
    """Run inference and return best action given visual observations."""
    tf.keras.backend.set_learning_phase(0)

    img = self.get_image(obs)

    pick_conf = self.attention.forward(img)
    pick_argmax = np.argmax(pick_conf)
    pick_argmax = np.unravel_index(pick_argmax, shape=pick_conf.shape)
    p0_pix = pick_argmax[:2]
    p0_theta = pick_argmax[2] * (2 * np.pi / pick_conf.shape[2])

    place_conf = self.transport.forward(img, p0_pix)
    _, z_tensor, roll_tensor, pitch_tensor = self.transport_6d.forward(
        img, p0_pix)

    place_argmax = np.argmax(place_conf)
    place_argmax = np.unravel_index(place_argmax, shape=place_conf.shape)
    p1_pix = place_argmax[:2]
    p1_theta = place_argmax[2] * (2 * np.pi / place_conf.shape[2])

    z_value = tf.reshape(
        z_tensor[:, place_argmax[0], place_argmax[1], place_argmax[2]], (1, 1))
    roll_value = tf.reshape(
        roll_tensor[:, place_argmax[0], place_argmax[1], place_argmax[2]],
        (1, 1))
    pitch_value = tf.reshape(
        pitch_tensor[:, place_argmax[0], place_argmax[1], place_argmax[2]],
        (1, 1))

    z_best = float(self.transport_6d.z_regressor(z_value)[0, 0])
    roll_best = float(self.transport_6d.roll_regressor(roll_value)[0, 0])
    pitch_best = float(self.transport_6d.pitch_regressor(pitch_value)[0, 0])

    z_best = float(np.clip(z_best, self.bounds[2, 0], self.bounds[2, 1]))
    roll_best = float(np.clip(roll_best, -np.pi / 2, np.pi / 2))
    pitch_best = float(np.clip(pitch_best, -np.pi / 2, np.pi / 2))

    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
    p1_xyh = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
    p1_xyz = (p1_xyh[0], p1_xyh[1], z_best)

    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((roll_best, pitch_best, -p1_theta))

    return {
        'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
    }

  def load(self, n_iter):
    """Load a pre-trained 3D transporter model."""
    print(f'Loading pre-trained model at {n_iter} iterations.')
    attention_fname = os.path.join(
        self.models_dir, f'attention-ckpt-{n_iter}.h5')
    transport_fname = os.path.join(
        self.models_dir, f'transport-ckpt-{n_iter}.h5')
    transport_6d_fname = os.path.join(
        self.models_dir, f'transport-6d-ckpt-{n_iter}.h5')
    self.attention.load(attention_fname)
    self.transport.load(transport_fname)
    self.transport_6d.load(transport_6d_fname)
    self.total_steps = n_iter

  def save(self):
    """Save 3D transporter checkpoints."""
    if not tf.io.gfile.exists(self.models_dir):
      tf.io.gfile.makedirs(self.models_dir)
    attention_fname = os.path.join(
        self.models_dir, f'attention-ckpt-{self.total_steps}.h5')
    transport_fname = os.path.join(
        self.models_dir, f'transport-ckpt-{self.total_steps}.h5')
    transport_6d_fname = os.path.join(
        self.models_dir, f'transport-6d-ckpt-{self.total_steps}.h5')
    self.attention.save(attention_fname)
    self.transport.save(transport_fname)
    self.transport_6d.save(transport_6d_fname)
