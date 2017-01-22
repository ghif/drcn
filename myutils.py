from PIL import Image, ImageDraw
import numpy as np

def preprocess_images(X, tmin=-1, tmax=1):
	V = X * (tmax - tmin) / 255.
	V += tmin
	return V

def postprocess_images(V, omin=-1, omax=1):
	X = V - omin
	X = X * 255. / (omax - omin)
	return X

def show_images(Xo, padsize=1, padval=0, filename=None, title=None):
	X = np.copy(Xo)
	[n, c, d1, d2] = X.shape
	if c== 1:
		X = np.reshape(X, (n, d1, d2))

	n = int(np.ceil(np.sqrt(X.shape[0])))
	
	padding = ((0, n ** 2 - X.shape[0]), (0, padsize), (0, padsize)) + ((0, 0), ) * (X.ndim - 3)
	canvas = np.pad(X, padding, mode='constant', constant_values=(padval, padval))

	canvas = canvas.reshape((n, n) + canvas.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, canvas.ndim + 1)))
	canvas = canvas.reshape((n * canvas.shape[1], n * canvas.shape[3]) + canvas.shape[4:])

	if title is not None:
		title_canv = np.zeros((50, canvas.shape[1]))
		title_canv = title_canv.astype('uint8')
		canvas = np.vstack((title_canv, canvas)).astype('uint8')
		
		I = Image.fromarray(canvas)
		d = ImageDraw.Draw(I)
		fill = 255
		d.text((10, 10), title, fill=fill, font=fnt)
	else:
		canvas = canvas.astype('uint8')
		I = Image.fromarray(canvas)

	if filename is None:
		I.show()
	else:
		I.save(filename)

	return I


