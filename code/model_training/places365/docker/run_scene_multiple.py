import numpy as np
import sys
import caffe
import pickle
import pandas as pd
import os
import logging

def check_id(out_csvPath):
    ids = set()
    if os.path.exists(out_csvPath):
        df = pd.read_csv(out_csvPath, header=None, names=['uuid', 'scene'])
        ls_id = df['uuid'].tolist()
        ids.update(ls_id)
    return ids

def classify_scene(net, fpath_labels, img, dim):

	# load input and configure preprocessing
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
	#transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # TODO - remove hardcoded path
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	# since we classify only one image, we change batch size from 10 to 1
	net.blobs['data'].reshape(1,3,dim,dim)

	# load the image in the data layer
	net.blobs['data'].data[...] = transformer.preprocess('data', img)

	# compute
	out = net.forward()

	# return top 1 prediction
	with open(fpath_labels, 'rb') as f:

		labels = pickle.load(f)
		top_1 = net.blobs['prob'].data[0].flatten().argsort()[-1]
		return labels[top_1]

if __name__ == '__main__':

	model = 'vgg16'
	dim = 224
	if model == 'alexnet':
		dim = 227

	# fetch pretrained models
	fpath_design = 'models_places/deploy_' + model + '_places365.prototxt'
	fpath_weights = 'models_places/' + model + '_places365.caffemodel'
	fpath_labels = 'resources/labels.pkl'

	# initialize net
	net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)
    
	input_filename = 'img_paths.csv' # please modify as needed

	out_Folder = 'insert the local ABSOLUTE PATH to the docker/sample_output folder' # please modify
	out_csvPath = out_Folder + '/places365.csv'
	if os.path.exists(out_csvPath) == False:
		df = pd.DataFrame(columns=['uuid','place'])
		df.to_csv(out_csvPath, index=False)
	already_id = check_id(out_csvPath)

	in_Folder = 'insert the local ABSOLUTE PATH to the docker/input_csvs folder' # please modify
	in_csvPath = in_Folder +'/' + input_filename
	df_imgs = pd.read_csv(in_csvPath)

	in_imgFolder = 'insert the local ABSOLUTE PATH to the docker/input_imgs folder' # please modify

	index = 0

	for i, row in df_imgs.iterrows():
		try:
			uuid = row['uuid']
			if uuid in already_id:
				continue

			index = index + 1
			if index % 10000 == 0:
				print '{} / {}'.format(index, len(df_imgs)-len(already_id))

			img_path = in_imgFolder + '/' + uuid + '.jpeg'
			img = caffe.io.load_image(img_path)
			scene = classify_scene(net, fpath_labels, img, dim)
			data_arr = [uuid,scene]
			df = pd.DataFrame(data_arr).T
			df.to_csv(out_csvPath, mode='a', header=False, index=False)
		except Exception as e:
			print(e)
			print("Failed for: {}".format(uuid))