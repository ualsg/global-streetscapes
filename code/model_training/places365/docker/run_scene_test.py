import numpy as np
import sys
import caffe
import pickle
import pandas as pd
import os

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
	transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # TODO - remove hardcoded path
	print(np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
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
    
	devices = ['test', 'test2']

	for device in devices:

		print 'Running inference for images on storage device:', device

		out_csvPath = '/data/yujun/repo/global-streetscapes-internal/code/model_training/places365/docker/output/{}.csv'.format(device)
		already_id = check_id(out_csvPath)

		in_csvPath = '/data/yujun/repo/global-streetscapes-internal/code/model_training/places365/docker/input_csvs/{}.csv'.format(device)
		df_imgs = pd.read_csv(in_csvPath)

		index = 0

		for _, row in df_imgs.iterrows():
			
			uuid = row['uuid']
			if uuid in already_id:
				continue

			index = index + 1
			if index % 100 == 0:
				print '{} / {}'.format(index, len(df_imgs)-len(already_id))

			img_path = '/data/yujun/repo/global-streetscapes-internal/code/model_training/places365/docker/input_imgs/{}.jpeg'.format(uuid)
			img = caffe.io.load_image(img_path)
			scene = classify_scene(net, fpath_labels, img, dim)
			data_arr = [uuid,scene]
			df = pd.DataFrame(data_arr).T
			df.to_csv(out_csvPath, mode='a', header=False, index=False)