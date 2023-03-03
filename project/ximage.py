#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
import os

image = np.loadtxt('ximage.dat');
image = np.rot90(image, k = 1)
fig = plt.figure()
ax = plt.imshow(255-image, cmap='gray', origin = 'upper')

def sample(obs, sigma = 2, order = 1, max_iter = 100, display = True, out_dir = None):
	res = np.copy(obs)
	res_mean = np.zeros_like(obs)
	width, height = obs.shape

	if order == 2:
		id_row = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
		id_col = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
	else:
		id_row = np.array([0, 0, 1, -1])
		id_col = np.array([-1, 1, 0, 0])
	
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for i in range(max_iter):
		print("iteration %d" % i)
		# iterate through each pixel
		for r in range(height):
			for c in range(width):
				yi = obs[r, c]
				neighbor_r = id_row + r
				neighbor_c = id_col + c
				neighbor_mask_r = np.logical_and(neighbor_r >= 0, neighbor_r < height)
				neighbor_mask_c = np.logical_and(neighbor_c >= 0, neighbor_c < width)
				neighbor_mask = np.logical_and(neighbor_mask_r, neighbor_mask_c)
				roi = res[neighbor_r[neighbor_mask], neighbor_c[neighbor_mask]]

				vi = roi.size
				mu_i = 1/(vi+1.0)*yi+vi/(vi+1.0)*np.mean(roi)
				sd_i = sigma/math.sqrt(vi+1.0)
				res[r, c] = np.random.normal(mu_i, sd_i)
				
				# plot(res)
				# if display:
				# 	plt.pause(0.05)
				# print("pixel (%d, %d) value %f, mean %f, yi %f, vi %d, mu %f, sd %f" 
				# 	% (r, c, res[r, c], np.mean(roi), yi, vi, mu_i, sd_i))
		res_mean += res
		plot(res)
		if display:
			plt.pause(0.05)
			# plt.show()	
		if out_dir:
			out_img_path = os.path.join(out_dir, "iter-%d.pdf" % i)
			plt.savefig(out_img_path, bbox_inches='tight', pad_inches=0);

	res_mean /= max_iter
	plot(res_mean)
	if out_dir:
		out_img_path = os.path.join(out_dir, "mean.pdf")
		plt.savefig(out_img_path, bbox_inches='tight', pad_inches=0);
		

def plot(img):
	# plt.imshow(255-img)
	ax.set_data(255-img)
	# plt.gray()	
	plt.xlim(-1, 20)
	plt.ylim(20, -1)
	xlabels = ['0', '5', '10', '15', '20']
	ylabels = ['20', '15', '10', '5', '0']
	arr = [-1, 4, 9, 14, 19]
	plt.xticks(arr, xlabels)
	plt.yticks(arr, ylabels)
	# plt.gca().xaxis.tick_top() 
	# plt.show()
		
def main():
	obs = np.loadtxt('ximage.dat');
	obs = np.rot90(obs)
	obs2 = np.full_like(obs, 57.5)
	# plot(obs)
	# plt.show()
	# return
	
	# plot(obs)	
	# plt.savefig("ximage.pdf", bbox_inches='tight', pad_inches=0);
	# plot(obs2)	
	# plt.savefig("ximage-mean.pdf", bbox_inches='tight', pad_inches=0);
	# plt.show()
	# return

	sigmas = [2, 5, 15]
	neighbors = [1, 2]
	for sigma in sigmas:
		for d in neighbors:
			dir_name = "sigma-%d_d-%d" % (sigma, d)
			sample(obs, sigma = sigma, order = d, max_iter = 100, out_dir = dir_name, display = False)

	sample(obs, sigma = 5, order = 1, max_iter = 100, out_dir = "sigma-5_d-1_uniform", display = False)
	# plt.show()
	plt.close()


if __name__ == "__main__":
	main()
