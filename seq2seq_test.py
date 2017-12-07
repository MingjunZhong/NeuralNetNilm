############################################################
# This code is to train a neural network to perform energy disaggregation, 
# i.e., given a sequence of electricity mains reading, the algorithm
# separates the mains into appliances.
#
# Inputs: mains windows -- find the window length in params_appliance
# Targets: appliances windows -- 
#
#
# This code is written by Chaoyun Zhang and Mingjun Zhong.
# Reference:
# Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton.
# ``Sequence-to-point learning with neural networks for nonintrusive load monitoring." 
# Thirty-Second AAAI Conference on Articial Intelligence (AAAI-18), Feb. 2-7, 2018.
############################################################

import NetFlowExt as nf
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import DataProvider
import nilm_metric as nm
from matplotlib.pylab import *

def remove_space(string):
    return string.replace(" ","")
    
def get_arguments():
    parser = argparse.ArgumentParser(description='Predict the appliance\
                                     give a trained neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')
    parser.add_argument('--appliance_name', 
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='../data/uk-dale/testdata/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1000,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=1,
                        help='The number of epoches.')
    parser.add_argument('--nosOfWindows',
                        type=int,
                        default=100,
                        help='The number of windows for prediction \
                        for each iteration.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params \
                                    at the end of traing.')
    return parser.parse_args()

#params_appliance = {'kettle':{'windowlength':129,
#                              'on_power_threshold':2000,
#                              'max_on_power':3948},
#                    'microwave':{'windowlength':129,
#                              'on_power_threshold':200,
#                              'max_on_power':3138},
#                    'fridge':{'windowlength':299,
#                              'on_power_threshold':50,
#                              'max_on_power':2572},
#                    'dishwasher':{'windowlength':599,
#                              'on_power_threshold':10,
#                              'max_on_power':3230},
#                    'washingmachine':{'windowlength':599,
#                              'on_power_threshold':20,
#                              'max_on_power':3962}}
                              
params_appliance = {'kettle':{'windowlength':599,
                              'on_power_threshold':2000,
                              'max_on_power':3998,
                             'mean':700,
                             'std':1000,
                             's2s_length':128},
                    'microwave':{'windowlength':599,
                              'on_power_threshold':200,
                              'max_on_power':3969,
                                'mean':500,
                                'std':800,
                                's2s_length':128},
                    'fridge':{'windowlength':599,
                              'on_power_threshold':50,
                              'max_on_power':3323,
                             'mean':200,
                             'std':400,
                             's2s_length':512},
                    'dishwasher':{'windowlength':599,
                              'on_power_threshold':10,
                              'max_on_power':3964,
                                  'mean':700,
                                  'std':1000,
                                  's2s_length':1536},
                    'washingmachine':{'windowlength':599,
                              'on_power_threshold':20,
                              'max_on_power':3999,
                                      'mean':400,
                                      'std':700,
                                      's2s_length':2000}}
    
args = get_arguments()
def load_dataset():
    app = args.datadir + args.appliance_name +'/' +'building2_'+ args.appliance_name
    test_set_x = np.load(app+'_test_x.npy')  
    test_set_y = np.load(app+'_test_y.npy')  
    ground_truth = np.load(app+'_test_gt.npy')  
    print('test set:', test_set_x.shape, test_set_y.shape)
    print('testset path:{}'.format(app+'_test_gt.npy'))
    print('testset path:{}'.format(app+'_test_x.npy'))
    print('testset path:{}'.format(app+'_test_y.npy'))
    
    return test_set_x, test_set_y, ground_truth

test_set_x, test_set_y, ground_truth = load_dataset()

shuffle = False
appliance_name = args.appliance_name
mean = params_appliance[application]['mean']
std = params_appliance[application]['std']
sess = tf.InteractiveSession()


windowlength = params_appliance[args.appliance_name]['windowlength']

offset = int(0.5*(params_appliance[application]['windowlength']-1.0))

test_kwag = {
    'inputs':test_set_x, 
    'targets':  ground_truth,
    'flatten':False}

# val_kwag = {
#     'inputs': val_set_x, 
#     'targets': val_set_y,
#     'flatten':False}

test_provider = DataProvider.MultiApp_Slider(batchsize = batchsize, 
                                                 shuffle = False, offset=offset)
# val_provider = DataProvider.DoubleSourceSlider(batchsize = 5000, 
#                                                  shuffle = False, offset=offset)


x = tf.placeholder(tf.float32, 
                   shape=[None, windowlength],
                   name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')



##### cnn2
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.ReshapeLayer(network, 
                                 shape=(-1, windowlength, 1, 1))
network = tl.layers.Conv2dLayer(network,  
                                act = tf.nn.relu, 
                                shape = [10, 1, 1, 30],
                                strides=[1, 1, 1, 1],  
                                padding='SAME', 
                                name = 'cnn1')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [8, 1, 30, 30],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn2')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [6, 1, 30, 40],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn3')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [5, 1, 40, 50],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn4')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [5, 1, 50, 50],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn5')
# network = tl.layers.Conv2dLayer(network,
#                                 act = tf.nn.relu,
#                                 shape = [5, 1, 50, 60],
#                                 strides=[1, 1, 1, 1],
#                                 padding='SAME',
#                                 name = 'cnn6')
network = tl.layers.FlattenLayer(network,
                                 name='flatten')
network = tl.layers.DenseLayer(network, 
                               n_units=1024, 
                               act = tf.nn.relu, 
                               name='dense2')
network = tl.layers.DenseLayer(network, 
                               n_units=windowlength,
                               act =tf.identity,
                               name='output_layer')

y = network.outputs
param_file = 'cnn'+app2+'_s2s_model_check.npz'
print(param_file)
#params = tl.files.load_npz(path='./', name=param_file)
params = tl.files.load_npz(path='', name=param_file)
tl.files.assign_params(sess, params, network)
print('params done')

test_prediction = nf.custompredict_add(sess=sess,
                                   network=network, 
                                   output_provider = test_provider , 
                                   x = x, 
                                   fragment_size=window_size, 
                                   output_length=windowlength, 
                                   y_op=None, 
                                   out_kwag=test_kwag,
                                  seqlength = test_set_x.size, std = std, mean = mean)

max_power = params_appliance[application]['max_on_power']
threshold = params_appliance[application]['on_power_threshold']


ground_truth = ground_truth[offset:-offset]*std+mean
mains = (test_set_x[offset:-offset])*std+mean

prediction = test_prediction[offset:-offset]
prediction[prediction<=0.0] = 0.0
print(prediction.shape)
print(ground_truth.shape)
# np.save(args.appliance_name.replace(" ","_")+'_prediction', prediction)
# to load results: np.load(args.appliance_name+'_prediction')
sess.close()
sample_second = 6.0 # sample time is 6 seconds
print('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))
print('NDE:{0}'.format(nm.get_nde(ground_truth.flatten(), prediction.flatten())))
print('MAE:{0}'.format(nm.get_abs_error(ground_truth.flatten(), prediction.flatten())))
print('SAE:{0}'.format(nm.get_sae(ground_truth.flatten(), prediction.flatten(), sample_second)))
save_name_y_pred = 'results/'+'pointnet_building2_'+args.appliance_name+'_pred.npy' #save path for mains
save_name_y_gt = 'results/'+'pointnet_building2_'+args.appliance_name+'_gt.npy'#save path for target
save_name_mains = 'results/'+'pointnet_building2_'+args.appliance_name+'_mains.npy'#save path for target
np.save(save_name_y_pred, prediction)
np.save(save_name_y_gt,ground_truth)
np.save(save_name_mains,mains)

