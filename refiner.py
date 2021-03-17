"""
Train synthesized captches with real unlabeled captchas via CycleGAN
"""
from utils import parse_args as base_parse_args, normalize, get_synthesizer, plot_images, get_refiner_custom_objects
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Conv2D, Activation, add, UpSampling2D, Conv2DTranspose, Flatten, AveragePooling2D
from tensorflow_addons.layers import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model, load_model
# from keras.utils import plot_model
from keras.engine.topology import Network

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import random
import datetime
import time
import json
import csv
import sys
import os
from os.path import exists, join, abspath, dirname

import keras.backend as K
import tensorflow as tf

from reflection_padding_2D import ReflectionPadding2D

# sys.path.append('../')
# import load_data


# sys.path.insert(1, dirname(abspath(__file__)) + '/../SimGAN-Captcha/')

np.random.seed(seed=12345)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CycleGAN():
    def __init__(self, flags):
        self.flags = flags
        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '_test'

        # ================ prepare directories ==============
        self.base_folder = f'refiner/{self.date_time}'
        self.image_folder = join(self.base_folder, 'images')
        if not exists(join(self.image_folder, 'A')):
            os.makedirs(join(self.image_folder, 'A'))
            os.makedirs(join(self.image_folder, 'B'))

        # Create folder to save model architecture and weights
        self.model_folder = join(self.base_folder, 'saved_models')
        if not exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.data_generator = DataGenerator(self.flags.synthesizer, self.flags.real_generator, self.flags.batch_size)

    def init_models(self):
        self.img_shape = self.flags.img_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = self.flags.lr_D
        self.learning_rate_G = self.flags.lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = self.flags.batch_size
        self.epochs = self.flags.epochs  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 50

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = False
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        # self.use_data_generator = False
        # self.use_data_generator = True

        # Tweaks
        self.REAL_LABEL = 0.9  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        if self.use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5]  # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        D_A.summary()

        # Discriminator builds
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')
        self.D_A.summary()
        self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use Networks to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')

        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        self.G_A2B.summary()
        self.G_B2A.summary()

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        self.writeMetaDataToJSON()

# ===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1, 1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

# ===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        # x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        # out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        # x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

# ===============================================================================
# Training
    def train(self):
        matplotlib.use("Agg")

        self.init_models()

        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
            # Generate batch of synthetic images
            # import pdb; pdb.set_trace()
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', self.epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % 20 == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (self.batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (self.batch_size,) + self.D_A.output_shape[1][1:]
            # label_shape4 = (self.batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            # ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            # zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (self.batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            loop_index = 0
            for images in self.data_generator:
                real_images_A = images[0]
                real_images_B = images[1]

                # Run all training steps
                run_training_iteration(loop_index, self.data_generator.count)

                loop_index += 1

            # ================== within epoch loop end ==========================

            if epoch % self.save_interval == 0:
                print('\n', '\n', '-------------------------Saving results for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

                # self.saveModel(self.G_model)
                # self.saveModel(self.D_A, epoch)
                # self.saveModel(self.D_B, epoch)
                self.G_A2B.save(f'{self.model_folder}/{self.G_A2B.name}_weights_epoch_{epoch}')
                self.G_B2A.save(f'{self.model_folder}/{self.G_B2A.name}_weights_epoch_{epoch}')

            # save the last one as formal
            print('Finishing...')
            G_A2B = self.modelGenerator(name='G_A2B_model')
            G_A2B.load_weights(self.flags.refiner_forward_model_path)
            G_A2B.save(self.flags.refiner_forward_model_path)
            G_B2A = self.modelGenerator(name='G_B2A_model')
            G_B2A.load_weights(self.flags.refiner_backward_model_path)
            G_B2A.save(self.flags.refiner_backward_model_path)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

# ===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        # if real_ is not None:
        #     if len(real_.shape) > 4:
        #         real_ = real_[0]
        #     image = np.hstack((real_[0], real, synthetic, reconstructed))
        # else:
        image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        plt.imshow(image, cmap='gray')
        plt.axis(False)
        plt.savefig(path_name)

    def saveImages(self, epoch, real_image_A, real_image_B):
        if len(real_image_A.shape) < 4:
            real_image_A = np.expand_dims(real_image_A, axis=0)
            real_image_B = np.expand_dims(real_image_B, axis=0)

        synthetic_image_B = self.G_A2B.predict(real_image_A)
        synthetic_image_A = self.G_B2A.predict(real_image_B)
        reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
        reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

        self.truncateAndSave(real_image_A, synthetic_image_B, reconstructed_image_A,
                             f'{self.image_folder}/A/epoch{epoch}_sample.png')
        self.truncateAndSave(real_image_B, synthetic_image_A, reconstructed_image_B,
                             f'{self.image_folder}/B/epoch{epoch}_sample.png')

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(real_images, synthetic_images, reconstructed_images,
                                 f'{self.image_folder}/tmp.png')
        except Exception:  # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        # if self.use_data_generator:
        max_nr_images = len(self.data_generator)
        # else:
        #     max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

# ===============================================================================
# Save and load

    def saveModel(self, model, filepath):
        return
        model.save_weights(filepath + '.hdf5')
        json_string = model.to_json()
        with open(filepath + '.json', 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in {}'.format(model.name, dirname(filepath)))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open(f'{self.base_folder}/loss_output.csv', 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of steps': len(self.data_generator)
        })

        with open(f'{self.base_folder}/meta_data.json', 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def test(self):
        G_A2B = load_model(self.flags.refiner_forward_model_path, custom_objects=get_refiner_custom_objects())
        G_B2A = load_model(self.flags.refiner_backward_model_path, custom_objects=get_refiner_custom_objects())

        A_test, B_test = next(self.data_generator)

        synthetic_images_B = G_A2B.predict(A_test)
        synthetic_images_A = G_B2A.predict(B_test)

        plots = []
        for i in range(len(synthetic_images_A)):
            test_A = A_test[i]
            test_B = B_test[i]
            synt_A = synthetic_images_A[i]
            synt_B = synthetic_images_B[i]
            img1 = np.concatenate((test_A, synt_B), 1)
            img2 = np.concatenate((test_B, synt_A), 1)
            plots.append(np.concatenate((img1, img2)))

        plot_images(np.squeeze(plots), 4)


class DataGenerator():
    def __init__(self, synthesizer, real_generator, batch_size=32):
        self.synthesizer = synthesizer
        self.real_generator = real_generator
        self.batch_size = batch_size
        self.count = int(self.real_generator.count / batch_size)

    def __len__(self):
        return self.count

    def __iter__(self):
        return self

    def __next__(self):
        return normalize(self.synthesizer.get_batch(no_label=True)), self.real_generator.get_batch()


class RealGenerator():
    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.count = len(os.listdir(join(flags.dataset_dir, flags.class_dir)))
        self.iter_count = 0

        self.datagen = ImageDataGenerator(
            preprocessing_function=normalize
        )

        self.flow_from_directory_params = {
            'directory': flags.dataset_dir,
            'target_size': flags.img_shape[:2],
            'color_mode': 'grayscale',
            'class_mode': None,
            'classes': [flags.class_dir]
        }

        self.batch_gen = self.datagen.flow_from_directory(
            **self.flow_from_directory_params,
            batch_size=flags.batch_size
        )

    def reset(self):
        self.iter_count = 0
        self.batch_gen.reset()

    def get_batch(self):
        next_batch = next(self.batch_gen)
        self.iter_count += len(next_batch)
        if len(next_batch) < self.batch_size or self.iter_count > self.count:  # stop loop over
            self.reset()
            raise StopIteration()

        return next_batch


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


def parse_args(args=None):
    flags = base_parse_args([
        (
            ('-e', '--epochs'),
            {
                'default': 200,
                'type': int,
            }
        ),
        (
            ('--lr-D',),
            {
                'default': 2e-4,
                'type': float,
                'help': 'learning rate of discriminator',
            }
        ),
        (
            ('--lr-G',),
            {
                'default': 2e-4,
                'type': float,
                'help': 'learning rate of generator',
            }
        ),
        (
            ('--class-dir',),
            {
                'type': str,
                'required': True,
                'help': 'inner folder where real images reside',
            }
        ),
        (('cmd', ), {'nargs': '?', 'help': 'train, test'}),
    ], args)[0]
    synthesizer, img_shape = get_synthesizer(flags)
    flags.synthesizer = synthesizer
    flags.img_shape = img_shape
    flags.real_generator = RealGenerator(flags)
    return flags


def test(flags=None):
    CycleGAN(flags).test()


def train(flags=None):
    CycleGAN(flags).train()


if __name__ == '__main__':
    FLAGS = parse_args()
    globals()[FLAGS.cmd](FLAGS)
