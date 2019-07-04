import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import zipfile
import time
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import random
import imageio

N_TRAIN = 6499
N_EPOCHS = 100
MB_SIZE = 12
GEN_INPUT_SIZE = 512
N_FILTERS = GEN_INPUT_SIZE

Z_FIXED = np.random.uniform(-1, 1, size = (6, GEN_INPUT_SIZE))
np.save('z_fixed.npy', Z_FIXED)
#Z_FIXED = np.load('z_fixed.npy')

def grid(imgs, rows, columns, x_size, y_size, name):
	background = Image.new('RGB',(x_size * rows, y_size * columns), (255, 255, 255))
	pil_imgs = []
	for l in range(rows * columns):
		pil_imgs.append(Image.fromarray(imgs[l, :, :, :]))
	for i in range(0, x_size * rows, x_size):
		for j in range(0, y_size * columns, y_size):
			background.paste(pil_imgs[columns * int(i) // x_size + int(j) // y_size], (i, j))
	background.save("{}.jpg".format(name))

def closest(img, data, n):
	idxs = np.argsort(np.sum(np.reshape((data.astype(np.float32) - img.astype(np.float32))**2, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])), axis = 1))
	return np.stack([data[idxs[i], :, :, :] for i in range(n)])

def pixelwise_normalization(x):
	return x / tf.expand_dims(K.sqrt(K.mean(K.square(x), axis = -1) + 1e-8), axis = -1)
	
def minibatch_std(x, mb_size, x_size, y_size): #ISSUE: MINIBATCH SIZE CANNOT CHANGE BETWEEN TRAINING SESSIONS
	m = tf.reduce_mean(K.std(x, axis = 0))
	append = tf.fill([mb_size, x_size, y_size, 1], m)
	return tf.concat([x, append], axis = -1)

def save_img(img, img_name, pil = False):
	if pil:
		image = Image.fromarray(img)
		image.save("fake_cats/{}.jpg".format(img_name))
	else:
		plt.clf()
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img)
		plt.savefig("fake_cats/{}".format(img_name), bbox_inches = 'tight')
		plt.clf()
	
class WGAN:
	
	def __init__(self, data_path, input_size_x, input_size_y, input_size_g, n_filters, kernel_size = 3, grad_penalty = 1, l2_penalty = .001, lr = .00001, data_size_x = 128, data_size_y = 128):
		self.data_path = data_path #must be zipfile
		self.n_blocks = 0
		self.count_train = 0
		self.alpha = K.variable(value = 0) 
		self.input_size_x = input_size_x
		self.input_size_y = input_size_y
		self.data_size_x = data_size_x
		self.data_size_y = data_size_y
		self.grad_penalty = grad_penalty
		self.l2_penalty = l2_penalty
		self.lr = lr
		
		self.loss_d = lambda y_true, y_pred: y_true * y_pred + self.l2_penalty * y_pred ** 2
		input1 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3), name = "input_0_1")
		input2 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3), name = "input_0_2")
		input3 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3), name = "input_0_3")
		from_RGB = keras.layers.Conv2D(n_filters, kernel_size = 1, strides = 1, padding = 'same', 
			activation = tf.nn.leaky_relu, name = "from_RGB_0")#, kernel_initializer = 'he_normal')
		conv1 = keras.layers.Conv2D(n_filters, kernel_size = kernel_size, strides = 1, padding = 'same', 
			activation = tf.nn.leaky_relu, name = "conv_0_1")#, kernel_initializer = 'he_normal')
		conv2 = keras.layers.Conv2D(n_filters, kernel_size = (self.input_size_x, self.input_size_y), 
			strides = 1, padding = 'valid', activation = tf.nn.leaky_relu, name = "conv_0_2")#, kernel_initializer = 'he_normal')
		flatten = keras.layers.Flatten(name = "flatten")
		mb_std = keras.layers.Lambda(lambda x: minibatch_std(x, MB_SIZE, input_size_x, input_size_y), name = "mb_std")
		dense = keras.layers.Dense(1, name = "dense")#, kernel_initializer = 'he_normal')
		output1 = dense(flatten(conv2(conv1(mb_std(from_RGB(input1))))))
		output2 = dense(flatten(conv2(conv1(mb_std(from_RGB(input2))))))
		output3 = dense(flatten(conv2(conv1(mb_std(from_RGB(input3))))))
		self.optimizer = keras.optimizers.Adam(lr = self.lr, beta_1 = 0, beta_2 = .99)
		self.D = keras.models.Model(inputs = input1, outputs = output1)
		self.penalty = lambda y_true, y_pred: self.grad_penalty * (tf.norm(K.gradients(y_pred, input3)) - 1) ** 2
		self.D_train = keras.models.Model(inputs = [input1, input2, input3], outputs = [output1, output2, output3])
		self.D_train.compile(optimizer = self.optimizer, loss = [self.loss_d, self.loss_d, self.penalty])
		
		self.input_size = input_size_g
		input = keras.layers.Input(shape = (self.input_size, ), name = "input")
		reshape = keras.layers.Reshape((1, 1, self.input_size), name = "reshape")
		conv_tr1 = keras.layers.Conv2DTranspose(n_filters, kernel_size = (self.input_size_x, 
			self.input_size_y), strides = 1, padding = 'valid', activation = tf.nn.leaky_relu, name = "convtr_0_1")#, kernel_initializer = 'he_normal')
		norm1 = keras.layers.Lambda(lambda x: pixelwise_normalization(x), name = "norm_0_1")
		conv_tr2 = keras.layers.Conv2DTranspose(n_filters, kernel_size = kernel_size, 
			strides = 1, padding = 'same', activation = tf.nn.leaky_relu, name = "convtr_0_2")#, kernel_initializer = 'he_normal')
		norm2 = keras.layers.Lambda(lambda x: pixelwise_normalization(x), name = "norm_0_2")
		to_RGB = keras.layers.Conv2DTranspose(3, kernel_size = 1, strides = 1, padding = 'same', activation = 'tanh',
			name = "to_RGB_0")#, kernel_initializer = 'he_normal')
		output = to_RGB(norm2(conv_tr2(norm1(conv_tr1(reshape(input))))))
		self.loss_g = lambda y_true, y_pred: -self.D(y_pred)
		self.G = keras.models.Model(inputs = input, outputs = output)
		self.G.compile(optimizer = self.optimizer, loss = self.loss_g)
		
	def generate_images(self, n_imgs):
		z = np.random.uniform(-1, 1, size = (n_imgs, self.input_size))
		imgs = ((self.G.predict(z) * 128) + 128).astype(np.uint8)
		return imgs
        
	def generate_grid(self, rows, columns, name): #generates random grid and saves (no return)
		grid(self.generate_images(rows * columns), rows, columns, self.input_size_x, self.input_size_y, name)

	def find_closest(self, imgs, name, n = 5, shuffle = False): #saves multiple closest files
		data = np.zeros(shape = (N_TRAIN, self.input_size_x, self.input_size_y, 3))
		with zipfile.ZipFile("cats_128.zip") as f:
			name_list = f.namelist()[1:]		
			k = 0
			for img_name in name_list[ : N_TRAIN]: 
				img = Image.open(f.open(img_name))
				img = img.resize(size = (self.input_size_y, self.input_size_x))
				data[k, :, :, :] = img
				k += 1
		data = data.astype(np.uint8)
		for k in range(imgs.shape[0]):
			close = closest(imgs[k, :, :, :], data, n)
			all = np.concatenate([np.reshape(imgs[k, :, :, :], (1, imgs.shape[1], imgs.shape[2], imgs.shape[3])), close])
			if shuffle: np.random.shuffle(all)
			background = Image.new('RGB',(self.input_size_x * (n + 1), self.input_size_y), (255, 255, 255))
			pil_imgs = []
			for l in range(n + 1):
				pil_imgs.append(Image.fromarray(all[l, :, :, :]))
			for i in range(0, self.input_size_x * (n + 1), self.input_size_y):
				for j in range(0, self.input_size_x, self.input_size_y):
					background.paste(pil_imgs[int(i) // self.input_size_x], (i, j))
			background.save("{}_{}.jpg".format(name, k))
        
	def morph(self, x, steps = 20, pause = 5, duration = .1, save = False, name = "morph"):
		xs = []
		for _ in range(pause):
			xs.append(x[0, :])
		for i in range(x.shape[0] - 1):
			for a in range(steps):
				xs.append((steps - a) * x[i, :] / (steps - 1) + a * x[i + 1, :] / (steps - 1))
			for _ in range(pause):
				xs.append(x[i + 1, :])
		for a in range(steps):
			xs.append((steps - a) * x[-1, :] / (steps - 1) + a * x[0, :] / (steps - 1))
		frames = ((self.G.predict(np.stack(xs)) * 128) + 128).astype(np.uint8)
		if save:
			gif = []
			for i in range(frames.shape[0]):
				gif.append(frames[i, :, :, :])
			imageio.mimsave('{}.gif'.format(name), gif, duration = duration)
		return frames
		
	def add_block(self, n_filters, kernel_size = 3):
		self.n_blocks += 1
		from_RGB_next = keras.layers.Conv2D(n_filters, kernel_size = 1, strides = 1, padding = 'same', 
			activation = tf.nn.leaky_relu, name = "from_RGB_{}".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		conv1 = keras.layers.Conv2D(n_filters, kernel_size = kernel_size, strides = 1, padding = 'same', 
			activation = tf.nn.leaky_relu, name = "conv_{}_1".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		n_filters_next = self.D.get_layer("conv_{}_1".format(self.n_blocks - 1)).get_output_shape_at(0)[3]
		conv2 = keras.layers.Conv2D(n_filters_next, kernel_size = kernel_size, strides = 1, padding = 'same', 
			activation = tf.nn.leaky_relu, name = "conv_{}_2".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		downsample = keras.layers.AveragePooling2D(name = "downsample_{}".format(self.n_blocks))
		from_RGB = self.D.get_layer("from_RGB_{}".format(self.n_blocks - 1))
		input = self.D_train.input
		self.input_size_x *= 2
		self.input_size_y *= 2
		input1 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3),
			name = "input_{}_1".format(self.n_blocks))
		input2 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3),
			name = "input_{}_2".format(self.n_blocks))
		input3 = keras.layers.Input(shape = (self.input_size_x, self.input_size_y, 3),
			name = "input_{}_3".format(self.n_blocks))
		left = keras.layers.Lambda(lambda x: (1 - self.alpha) * x, name = "left_{}".format(self.n_blocks))
		right = keras.layers.Lambda(lambda x: self.alpha * x, name = "right_{}".format(self.n_blocks))
		add = keras.layers.Add(name = "add_{}".format(self.n_blocks))
		d_left1 = left(from_RGB(downsample(input1)))
		d_right1 = right(downsample(conv2(conv1(from_RGB_next(input1)))))
		d1 = add([d_left1, d_right1])
		for layer in self.D.layers[2:]:
			d1 = layer(d1)
		output1 = d1
		d_left2 = left(from_RGB(downsample(input2)))
		d_right2 = right(downsample(conv2(conv1(from_RGB_next(input2)))))
		d2 = add([d_left2, d_right2])
		for layer in self.D.layers[2:]:
			d2 = layer(d2)
		output2 = d2
		d_left3 = left(from_RGB(downsample(input3)))
		d_right3 = right(downsample(conv2(conv1(from_RGB_next(input3)))))
		d3 = add([d_left3, d_right3])
		for layer in self.D.layers[2:]:
			d3 = layer(d3)
		output3 = d3
		self.D_train = keras.models.Model(inputs = [input1, input2, input3], outputs = [output1, output2, output3])
		self.penalty = lambda y_true, y_pred: self.grad_penalty * (tf.norm(K.gradients(y_pred, input3)) - 1) ** 2
		self.D_train.compile(optimizer = self.optimizer, loss = [self.loss_d, self.loss_d, self.penalty])
		self.D = keras.models.Model(inputs = input1, outputs = output1)
		
		upsample = keras.layers.UpSampling2D(name = "upsample_{}".format(self.n_blocks))
		conv_tr1 = keras.layers.Conv2DTranspose(n_filters, kernel_size = kernel_size, strides = 1, 
			padding = 'same', activation = tf.nn.leaky_relu, name = "convtr_{}_1".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		norm1 = keras.layers.Lambda(lambda x: pixelwise_normalization(x), name = "norm_{}_1".format(self.n_blocks))
		conv_tr2 = keras.layers.Conv2DTranspose(n_filters, kernel_size = kernel_size, strides = 1, 
			padding = 'same', activation = tf.nn.leaky_relu, name = "convtr_{}_2".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		norm2 = keras.layers.Lambda(lambda x: pixelwise_normalization(x), name = "norm_{}_2".format(self.n_blocks))
		to_RGB = self.G.get_layer("to_RGB_{}".format(self.n_blocks - 1))
		to_RGB_next = keras.layers.Conv2DTranspose(3, kernel_size = 1, strides = 1, 
			padding = 'same', activation = 'tanh', name = "to_RGB_{}".format(self.n_blocks))#, kernel_initializer = 'he_normal')
		g = self.G.get_layer("norm_{}_2".format(self.n_blocks - 1)).output
		g = upsample(g)
		left_g = keras.layers.Lambda(lambda x: (1 - self.alpha) * x, name = "left_g_{}".format(self.n_blocks))
		right_g = keras.layers.Lambda(lambda x: self.alpha * x, name = "right_g_{}".format(self.n_blocks))
		g1 = left_g(to_RGB(g))
		g2 = right_g(to_RGB_next(norm2(conv_tr2(norm1(conv_tr1(g))))))
		add_g = keras.layers.Add(name = "add_g_{}".format(self.n_blocks))
		output = add_g([g1, g2])
		self.G = keras.models.Model(inputs = self.G.input, outputs = output)
		self.G.compile(optimizer = self.optimizer, loss = self.loss_g)
	
	def remove_RGB(self):
		input1 = self.D_train.input[0]
		input2 = self.D_train.input[1]
		input3 = self.D_train.input[2]
		downsample = self.D.get_layer("downsample_{}".format(self.n_blocks))
		from_RGB = self.D.get_layer("from_RGB_{}".format(self.n_blocks))
		conv1 = self.D.get_layer("conv_{}_1".format(self.n_blocks))
		conv2 = self.D.get_layer("conv_{}_2".format(self.n_blocks))
		d1 = downsample(conv2(conv1(from_RGB(input1))))
		for layer in self.D.layers[9:]:
			d1 = layer(d1)
		output1 = d1
		d2 = downsample(conv2(conv1(from_RGB(input2))))
		for layer in self.D.layers[9:]:
			d2 = layer(d2)
		output2 = d2
		d3 = downsample(conv2(conv1(from_RGB(input3))))
		for layer in self.D.layers[9:]:
			d3 = layer(d3)
		output3 = d3
		self.D_train = keras.models.Model(inputs = [input1, input2, input3], outputs = [output1, output2, output3])
		self.penalty = lambda y_true, y_pred: self.grad_penalty * (tf.norm(K.gradients(y_pred, input3)) - 1) ** 2
		self.D_train.compile(optimizer = self.optimizer, loss = [self.loss_d, self.loss_d, self.penalty])
		self.D = keras.models.Model(inputs = input1, outputs = output1)
	
		g = self.G.get_layer("norm_{}_2".format(self.n_blocks)).output
		to_RGB = self.G.get_layer("to_RGB_{}".format(self.n_blocks))
		output = to_RGB(g)
		self.G = keras.models.Model(inputs = self.G.input, outputs = output)
		self.G.compile(optimizer = self.optimizer, loss = self.loss_g)
	
	def train(self, n_epochs, n_train, mb_size, n_gen = 6, Z = None):
		self.count_train += 1
		steps = 0
		K.set_value(self.alpha, 0)
		D_losses = deque(maxlen = n_train // mb_size)
		G_losses = deque(maxlen = n_train // mb_size)
		D_losses_epoch = deque(maxlen = n_epochs)
		G_losses_epoch = deque(maxlen = n_epochs)
		with zipfile.ZipFile(self.data_path) as f:
			name_list = f.namelist()[1:]
			random.shuffle(name_list)			
			data = np.zeros(shape = (n_train, self.input_size_x, self.input_size_y, 3))
			k = 0
			for img_name in name_list[ : n_train]: 
				img = Image.open(f.open(img_name))
				img = img.resize(size = (self.input_size_y, self.input_size_x))
				data[k, :, :, :] = img
				k += 1
			data = (data.astype(np.float32) - 128) / 128
			for i in range (n_epochs):
				time_start = time.clock()
				y_ones = np.ones(mb_size)
				for j in range(n_train // mb_size):
					z_D = np.random.uniform(-1, 1, size = (mb_size, self.input_size))
					z_G = np.random.uniform(-1, 1, size = (mb_size, self.input_size))
					epsilon = np.random.uniform(0, 1, size = (mb_size, 1, 1, 1))
					batch = data[MB_SIZE * j : MB_SIZE * (j + 1), :, :, :]
					Gz = self.G.predict(z_D)
					x = epsilon * batch + (1 - epsilon) * Gz
					Dloss = self.D_train.train_on_batch([Gz, batch, x], [y_ones, -y_ones, y_ones])
					D_losses.append(Dloss[0])
					Gloss = self.G.train_on_batch(z_G, y_ones)
					G_losses.append(Gloss)
					steps += 1
					K.set_value(self.alpha, steps / (n_epochs * (n_train // mb_size)))
					print("Epoch {}/{}\t Batch {}/{}\t D loss: {:.5f}\t G loss: {:.5f}".format(
						i + 1, n_epochs, j + 1, n_train // mb_size, np.mean(D_losses), np.mean(G_losses)), end = "\r")
				print("Epoch {}/{}\t Batch {}/{}\t D loss: {:.5f}\t G loss: {:.5f}\t Time: {:.0f} sec".format(
					i + 1, n_epochs, j + 1, n_train // mb_size, np.mean(D_losses), np.mean(G_losses), 
					time.clock() - time_start))
				D_losses_epoch.append(np.mean(D_losses))
				G_losses_epoch.append(np.mean(G_losses))
				imgs = self.generate_images(n_gen)
				for l in range(n_gen):
					img_name = "cats_train{}_epoch{}_{}x{}_{}".format(self.count_train, i + 1, self.input_size_x, 
						self.input_size_y, l + 1)
					save_img(imgs[l, :, :, :], img_name)
				if Z is not None:
					imgs = (self.G.predict(Z) * 128 + 128).astype(np.uint8)
					for l in range(Z.shape[0]):
						img_name = "cats_{}_fixed_train{}_epoch{}_{}x{}".format(l + 1, self.count_train, i + 1, 
							self.input_size_x, self.input_size_y)
						save_img(imgs[l, :, :, :], img_name)
		self.G.save_weights('WGAN_cats_G_{}'.format(self.count_train))
		self.D.save_weights('WGAN_cats_D_{}'.format(self.count_train))
		self.D_train.save_weights('WGAN_cats_D_train_{}'.format(self.count_train))
		print('Models saved')
		plt.clf()
		plt.plot(D_losses_epoch)
		plt.title('D Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train'], loc = 'upper left')
		plt.savefig('D_loss_cats_train{}_{}x{}'.format(self.count_train, self.input_size_x, 
			self.input_size_y))
		plt.clf()
		plt.plot(G_losses_epoch)
		plt.title('G Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train'], loc = 'upper left')
		plt.savefig('G_loss_cats_train{}_{}x{}'.format(self.count_train, self.input_size_x,
			self.input_size_y))
		plt.clf()

wgan = WGAN(data_path = "cats_128.zip", input_size_x = 4, input_size_y = 4, input_size_g = GEN_INPUT_SIZE, n_filters = N_FILTERS)
wgan.D.summary()
wgan.G.summary()

#wgan.D.load_weights('WGAN_cats_D_1')
#wgan.D_train.load_weights('WGAN_cats_D_train_1')
#wgan.G.load_weights('WGAN_cats_G_1')
#wgan.count_train = 1

### training:
print("Starting training...")

wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.add_block(N_FILTERS)
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.remove_RGB()
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.add_block(N_FILTERS)
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.remove_RGB()
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.add_block(N_FILTERS)
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.remove_RGB()
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.add_block(N_FILTERS // 2)
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.remove_RGB()
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.add_block(N_FILTERS // 4)
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

wgan.remove_RGB()
wgan.train(N_EPOCHS, N_TRAIN, MB_SIZE, Z = Z_FIXED)

print("Training complete")

### generate grids
for i in range(10):
	wgan.generate_grid(6, 6, "grid_{}".format(i))

### generate and save images
z = np.random.uniform(-1, 1, size = (200, wgan.input_size))
np.save('z.npy', z)
#z = np.load('z.npy')
imgs = ((wgan.G.predict(z) * 128) + 128).astype(np.uint8)
for l in range(200):
	img_name = "cats_final_z_saved_t{}_{}x{}_n{}".format(wgan.count_train, wgan.input_size_x, wgan.input_size_y, l + 1)
	save_img(imgs[l, :, :, :], img_name, pil = True)

### make morph gif
wgan.morph(Z_FIXED, steps = 100, pause = 20, duration = .03, save = True, name = "cats_morph")

### generate closest
imgs = wgan.G.predict(Z_FIXED)
imgs = ((imgs * 128) + 128).astype(np.uint8)
wgan.find_closest(imgs, name = "closest_new")
