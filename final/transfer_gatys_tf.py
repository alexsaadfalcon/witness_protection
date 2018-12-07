import tensorflow as tf 
import dippykit as dip
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from itertools import combinations
import os, sys 
import argparse
import scipy.misc
import time

import copy

VGG_MEAN = [103.939, 116.779, 123.68]
def build_part_vgg19(img_input,params_dir='vgg19.npy'):
    '''
    input tensor: input image with shape of [N, H, W, C]
    params_dir: Directory of npz file
    '''
    def conv_layer(x, name):
        with tf.variable_scope(name):
            f = tf.constant(params[name][0],dtype='float32')
            b = tf.constant(params[name][1],dtype='float32')

            conv = tf.nn.conv2d(input=x, filter=f,strides=[1,1,1,1],padding='SAME')
            conv_biased = tf.nn.bias_add(conv, b)
            return tf.nn.relu(conv_biased)

    def max_pool(x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    params = np.load(params_dir,encoding='latin1').item()
    print('Params loaded from %s'%params_dir)

    red, green, blue = tf.split(axis=3, value=img_input,num_or_size_splits=3)
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])

    conv1_1 = conv_layer(bgr, 'conv1_1')
    conv1_2 = conv_layer(conv1_1, 'conv1_2')
    pool1   = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, 'conv2_1')
    conv2_2 = conv_layer(conv2_1, 'conv2_2')
    pool2   = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, 'conv3_1')
    conv3_2 = conv_layer(conv3_1, 'conv3_2')
    conv3_3 = conv_layer(conv3_2, 'conv3_3')
    conv3_4 = conv_layer(conv3_3, 'conv3_4')
    pool3   = max_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, 'conv4_1')
    conv4_2 = conv_layer(conv4_1, 'conv4_2')
    conv4_3 = conv_layer(conv4_2, 'conv4_3')
    conv4_4 = conv_layer(conv4_3, 'conv4_4')
    pool4   = max_pool(conv4_4, 'pool4')
    
    conv5_1 = conv_layer(pool4, 'conv5_1')
           # Style                                        # Content
    #return conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2
    return [conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool2, conv3_1, conv3_2, conv3_3, conv3_4, pool3, conv4_1, conv4_2, conv4_3, conv4_4, pool4, conv5_1]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-img',type=str,default='content_img.jpg',help='The content image')
    parser.add_argument('--style-img',type=str,default='style_img.jpg',help='The style image')
    parser.add_argument('--output',type=str, default='faceswap',help='Output image')
    parser.add_argument('--lr-rate',type=float, default=.2,help='Learning rate')
    parser.add_argument('--epoch',type=int, default=1000,help='Epoch number')
    parser.add_argument('--style-weight',type=float, default=1,help='trade-off between content and style')
    parser.add_argument('--content-weight',type=float, default=1,help='trade-off between content and style')

    args = parser.parse_args()

    content_img = Image.open(args.content_img)
    img_width,img_height = content_img.size
    content_img = content_img.resize((img_width,img_height))
    style_img = Image.open(args.style_img).resize((img_width,img_height))

    input_img = copy.copy(content_img)
#    plt.title('Content Image')
#    plt.imshow(content_img)
#    plt.pause(1)

#    plt.title('Style Image')
#    plt.imshow(style_img)
#    plt.pause(1)

    vgg_input =  tf.Variable(initial_value=np.zeros(shape=[1, img_height, img_width, 3],dtype='float32'),name='image') 

    # Style                                       # Content
    # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1,  conv4_2 = build_part_vgg19(vgg_input,params_dir='vgg19.npy')
    # conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool2, conv3_1, conv3_2, conv3_3, conv3_4, pool3, \
    # conv4_1, conv4_2, conv4_3, conv4_4, pool4, conv5_1 = build_part_vgg19(vgg_input,params_dir='vgg19.npy')
    layers = build_part_vgg19(vgg_input,params_dir='vgg19.npy')
    #for l in layers:
    #    print(l.shape)

    # reshape to NHWC
    content_img = np.reshape(content_img,newshape=(-1,img_height,img_width,3))
    print(content_img.dtype, np.max(content_img), np.min(content_img))
    style_img = np.reshape(style_img,newshape=(-1,img_height,img_width,3))
    input_img = np.reshape(input_img,newshape=(-1,img_height,img_width,3))

    # GPU Config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    sess = tf.Session(config=tf_config)
    counter = 0

    with open('results.txt', 'a') as of:
        of.write('RESULTS FILE\n')
    sess.run(tf.global_variables_initializer())
    for lr_rate in np.logspace(-2, 1, 2):
        for style_weight in np.logspace(1, 4, 2):
            for m, n in np.ndindex((3, 3)):
                m += 2; n += 2
                for content_layers in combinations(layers, m):
                    for style_layers in combinations(layers, n):
                        print(content_layers)
                        counter += 1
                        # Get content feature maps
                        sess.run(vgg_input.assign(content_img))
                        if m > 1:
                            content_maps_out = sess.run(content_layers)
                        else:
                            content_maps_out, = sess.run([content_layers])
                        # Get style feature maps
                        sess.run(vgg_input.assign(style_img))
                        if n > 1:
                            style_maps_out = sess.run(style_layers)
                        else:
                            style_maps_out, = sess.run([style_layers])

                        # Loss
                        def gram_matrix(maps):
                            if isinstance(maps,tf.Tensor):
                                maps_vec = tf.transpose(maps,perm=(0,3,1,2))
                                a,b,c,d = maps_vec.shape
                                maps_vec = tf.reshape(maps_vec,(a*b,c*d))
                                return 1/(2* int(a*b*c*d) ) * tf.matmul(maps_vec, maps_vec,transpose_b=True)
                            else:
                                maps_vec = np.array(maps).transpose((0,3,1,2))
                                a,b,c,d = maps_vec.shape
                                maps_vec = maps_vec.reshape(a*b,c*d)
                                return 1/(2*(a*b*c*d) ) * np.matmul(maps_vec,maps_vec.T)

                        # Input
                        if m > 1:
                            content_maps = [tf.constant(content_maps_out[i],dtype='float32') for i in range(len(content_maps_out))]
                        else:
                            content_maps = tf.constant(content_maps_out, dtype='float32')

                        if n > 1:
                            style_maps = [tf.constant(style_maps_out[i], dtype='float32') for i in range(len(style_maps_out))]
                        else:
                            style_maps = tf.constant(style_maps_out, dtype='float32')


                        #def _cal_squaredNM(m):
                        #    m_shape = m.get_shape().as_list()
                        #    return 4*(m_shape[0]*m_shape[1])**2

                        #style_weights = [0.2,0.2,0.2,0.2,0.2]
                        #img_styles = [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]

                        def mse(x,y):
                            return tf.losses.mean_squared_error(labels=y,predictions=x)

                        def cosdist(x,y):
                            return tf.losses.cosine_distance(labels=y,predictions=x,axis=1)

                        loss_content = tf.Variable(initial_value=0, dtype='float32')
                        if m > 1:
                            for l, content_layer in enumerate(content_layers):
                                loss_content = tf.add(loss_content, mse(content_layer,content_maps[l]))
                        else:
                            loss_content = tf.add(loss_content, mse(content_layers,content_maps))

                        loss_style = tf.Variable(initial_value=0, dtype='float32')
                        if m > 1:
                            for l, style_layer in enumerate(style_layers):
                                loss_style = tf.add(loss_style, mse(style_layer,style_maps[l]))
                        else:
                            loss_style = tf.add(loss_style, mse(style_layers,style_maps))

                        print('Added loss_content and style')
                        loss = args.content_weight*loss_content + args.style_weight*loss_style

                        # Train
                        opt = tf.train.AdamOptimizer(args.lr_rate).minimize(loss,var_list=[vgg_input])

                        sess.run(tf.global_variables_initializer())
                        #white_noise = dip.adjustments.image_noise(np.zeros((1, img_height, img_width, 3)))
                        #dip.imshow(white_noise)
                        #dip.show()
                        #sess.run(vgg_input.assign(white_noise))
                        sess.run(vgg_input.assign(input_img))
                        initTime = time.time()
                        for ep in range(args.epoch+1):
                            _, cur_loss,s_loss,c_loss,img = sess.run([opt,loss,loss_style, loss_content, vgg_input])
                            if ep%5==0:
                                print('[*] Epoch %d  total_loss=%f, style_loss=%f, content_loss=%f'%(ep,cur_loss,s_loss,c_loss))
                            if ep%1000==0:
                                saved_img = np.array(img[0])
                                saved_img = np.where(saved_img<=255,saved_img,255)
                                saved_img = np.where(saved_img>=0,saved_img,0)
                                saved_img = Image.fromarray(saved_img.astype(np.uint8),'RGB')
                                output_name = args.output+'_%d'%(counter)+'.jpg'
                                saved_img.save(output_name)
                                print("[!] image saved as %s\n"%output_name)

                        with open('results.txt', 'a') as of:
                            of.write(f'{counter}, learning rate: {lr_rate}, style weight: {style_weight}, '
                                     f'content layers {content_layers}, style layers {style_layers}')

                        print('Done after {} seconds'.format(time.time()-initTime))
    
if __name__=='__main__':
    main()
