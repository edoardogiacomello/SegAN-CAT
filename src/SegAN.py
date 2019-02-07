import tensorflow as tf
import dataset_helpers as dh
import os, math
from SegANViewer import SeganViewer
import time

class SegAN():

    def __init__(self, mri_shape, seg_shape, config=None, run_name='last'):
        self.params = {
                        'session_config':config,
                        'mri_shape':mri_shape,
                        'seg_shape':seg_shape,
                        'clip': {'C': True, 'S': False, 'constraint':lambda x: tf.clip_by_value(x, clip_value_min=-0.05, clip_value_max=0.05)},
                        'dtype': tf.float32,
                        'learning_rate': 0.00002,
                        'max_iterations': 500000,
                        'visualize_every_itn': 1000,
                        'batch_size': mri_shape[0],
                        'checkpoint_folder': '../models/SegAN/{}_model/'.format(run_name),
                        'visualize_folder': '../models/SegAN/{}_visualize/'.format(run_name),
                        'threshold': 0.5, # Treshold for a segmentation to be considered as 1 or 0
                        'max_labels': 1.0 # Clip all the segmentation pixels to this value
                       }
        # Define a dictionary that will contain the layers for faster access
        self.layers = {'in':{}, # Inputs to the network
                       'S':{}, # Segmentor
                       'C_gt':{}, # Critic layers with ground truth as input
                       'C_s':{}, # Critic layers with S output as input (parameters are shared with C_gt)
                       'train':{}, # Operations for training
                       'view':{}, # Operations for tensorboard visualization
                       'eval':{}, # Tensors containing evaluation metrics
                       'ops': {}, # operators on variables to influence the training phase
                       }

    def build_network(self, input_mri, true_seg):
        '''
        Build a network using as input the tensors specified as parameters.
        The value of "training" must be fed as a feed_dict in order to enable/disable batch normalization learning
        :param input_mri: Tensor for the MRI to use as S and C input
        :param true_seg: Tensor for the ground truth segmentation in input to C
        :return:
        '''
        # This flag is used for enabling/disabling the training mode for batch normalization.
        # When true, BN layers use mean/var for the current batch, otherwise they use the learned ones.
        self.layers['in']['training'] = tf.get_variable("training", dtype=tf.bool, trainable=False, initializer=False)
        self.layers['ops']['enable_training'] = tf.assign(self.layers['in']['training'], True, validate_shape=False)
        self.layers['ops']['disable_training'] = tf.assign(self.layers['in']['training'], False, validate_shape=False)

        # Setting the inputs
        self.layers['in']['mri'] = input_mri
        self.layers['in']['seg'] = true_seg
        self.layers['S']['out'] = self.build_segmentor()
        # Critic output when ground truth is in input
        self.layers['C_gt']['out'] = self.build_critic(self.layers['in']['seg'], layer_idx='C_gt')
        # Critic output when segmentor is in input
        self.layers['C_s']['out'] = self.build_critic(self.layers['S']['out'], layer_idx='C_s', reuse=True)
        self.define_loss()
        self.define_evaluation_metrics()
        # Defining optimizer
        self.define_optimizer()
        # Defining the saver. Every tensor defined afterwards is not included in the model checkpoint
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=0)

    def _conv2d(self, input, kern_size, filters, strides, padding='SAME', name=None, clip=False):
        def get_filters(shape, clip):
            # This initialization is taken from SegAN code
            initializer = tf.random_normal(shape, mean=0.0, stddev=math.sqrt(2./(shape[0]*shape[1]*shape[2])), dtype=self.params['dtype'])
            return tf.Variable(initializer, constraint=self.params['clip']['constraint'] if clip else None, name="W", dtype=self.params['dtype'])
        filter_shape = (kern_size, kern_size, input.shape[-1].value, filters)
        # No need for bias as the batchnorm already account for that
        return tf.nn.conv2d(input, filter=get_filters(filter_shape, clip), strides=[1,strides, strides,1], padding=padding, name=name)

    def _batchnorm(self, x):
        return tf.layers.batch_normalization(x, epsilon=1e-5, momentum = 0.1, training=self.layers['in']['training'])

    def _resize(self, input, factor_wrt_input=2):
        new_shape = [int(self.params['mri_shape'][1]*factor_wrt_input), int(self.params['mri_shape'][1]*factor_wrt_input)]
        return tf.cast(tf.image.resize_bilinear(input, size=new_shape), self.params['dtype'])

    def build_segmentor(self, scope='S', reuse=None):
        '''
        SegAN Segmentor is made of
            Input = 160x160x3
            Conv4x4 N64 S2 / LeakyRelu
            Conv4x4 N128 S2 / BN / LeakyRelu
            Conv4x4 N256 S2 / BN / LeakyRelu
            Conv4x4 N512 S2 / BN / LeakyRelu
            Resize2 / Conv3x3 N256 S1 / BN / Relu
            Resize2 / Conv3x3 N128 S1 / BN / Relu
            Resize2 / Conv3x3 N64  S1 / BN / Relu
            Resize2 / Conv3x3 N3   S1
        First and last layers have BatchNorm neither bias.

        :param scope: Variable scope for this network. If the same name has been defined already a reuse must be explicitly triggered
        :param reuse: Trigger reuse to define a new segmentor sharing the layers with a previously defined one
        :return: None. Created layers are saved in self.layers
        '''
        with tf.variable_scope(scope, reuse=reuse):
            self.layers['S']['enc_1'] = tf.nn.leaky_relu(self._conv2d(self.layers['in']['mri'], kern_size=4, filters=64, strides=2))
            self.layers['S']['enc_2'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_1'], kern_size=4, filters=128, strides=2, clip=self.params['clip']['S'])))
            self.layers['S']['enc_3'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_2'], kern_size=4, filters=256, strides=2, clip=self.params['clip']['S'])))
            self.layers['S']['enc_4'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_3'], kern_size=4, filters=512, strides=2, clip=self.params['clip']['S'])))
            self.layers['S']['dec_3'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(self.layers['S']['enc_4'], factor_wrt_input=1/8), kern_size=3, filters=256, strides=1, clip=self.params['clip']['S'])))
            self.layers['S']['dec_2'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(tf.add(self.layers['S']['dec_3'], self.layers['S']['enc_3']), factor_wrt_input=1/4), kern_size=3, filters=128, strides=1, clip=self.params['clip']['S'])))
            self.layers['S']['dec_1'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(tf.add(self.layers['S']['dec_2'], self.layers['S']['enc_2']), factor_wrt_input=1/2), kern_size=3, filters=64, strides=1, clip=self.params['clip']['S'])))
            self.layers['S']['out'] = tf.sigmoid(self._conv2d(self._resize(tf.add(self.layers['S']['dec_1'], self.layers['S']['enc_1']), factor_wrt_input=1.0), kern_size=3, filters=self.params['seg_shape'][-1], strides=1, clip=self.params['clip']['S'], name='out_unbound'), name='out')
        return self.layers['S']['out']

    def build_critic(self, input_seg, scope='C', layer_idx='C_gt', reuse=None):
        '''

        :param input_seg: Segmentation tensor in input to the critic (ground truth or from segmentor)
        :param layer_idx: Which dictionary index to store the network layers
        :param scope: Variable scope for this network. If the same name has been defined already a reuse must be explicitly triggered
        :param reuse: Trigger reuse to define a new segmentor sharing the layers with a previously defined one
        :return:
        '''
        C = layer_idx
        with tf.variable_scope(scope, reuse=reuse):
            self.layers[C]['mri_masked'] = tf.multiply(self.layers['in']['mri'], input_seg, name='mri_masked')
            self.layers[C]['enc_1'] = tf.nn.leaky_relu(self._conv2d(self.layers[C]['mri_masked'], kern_size=4, filters=64, strides=2, clip=self.params['clip']['C']))
            self.layers[C]['enc_2'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers[C]['enc_1'], kern_size=4, filters=128, strides=2, clip=self.params['clip']['C'])))
            self.layers[C]['enc_3'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers[C]['enc_2'], kern_size=4, filters=256, strides=2, clip=self.params['clip']['C'])))

            # Flattening each 4D activation volume in a 2D
            enc_1_flattened = tf.reshape(self.layers[C]['enc_1'], shape=[self.params['batch_size'], -1]) # Shape: (batchsize, 409600)
            enc_2_flattened = tf.reshape(self.layers[C]['enc_2'], shape=[self.params['batch_size'], -1]) # Shape: (batchsize, 204800)
            enc_3_flattened = tf.reshape(self.layers[C]['enc_3'], shape=[self.params['batch_size'], -1]) # Shape: (batchsize, 102400)

            # Building the output by stacking activations. In the original paper a 2D output is produced (concatenated), we'll keep it as 3D for better understanding of the volume
            self.layers[C]['out'] = tf.stack([enc_1_flattened,
                                                 tf.tile(enc_2_flattened, [1,2]),
                                                 tf.tile(enc_3_flattened, [1,4])], axis=-1)
        return self.layers[C]['out']

    def define_loss(self):
        # Loss is defined as the Mean Absolute Error (L1) between the critic features at each layer for the true and generated segmentation respectively
        # Since the discriminator outputs all the layers in a single volume it is sufficient to apply the L1 norm to it
        def smooth_dice_loss(x, y):
            # DSC is defined as 2*(|intersection(X,Y)|/(|X|+|Y|) where |.| is the cardinality (white pixels)
            # The classical form isn't differentiable so we use an approximated continuous form
            epsilon = tf.constant(0.0000001, name="epsilon", dtype=self.params['dtype']) # This is necessary to avoid division by zero if the two pictures are black
            x = tf.reshape(x, [self.params['batch_size'], -1])
            y = tf.reshape(y, [self.params['batch_size'], -1])
            intersection = tf.reduce_sum(tf.multiply(x, y))
            return tf.divide(2. * intersection + epsilon, tf.reduce_sum(x) + tf.reduce_sum(y) + epsilon, name="dice_loss")

        self.layers['train']['loss_c'] = tf.reduce_mean(tf.abs(self.layers['C_gt']['out']-self.layers['C_s']['out']))
        self.layers['train']['loss_s'] = tf.reduce_mean(tf.abs(self.layers['C_gt']['out']-self.layers['C_s']['out'])) + smooth_dice_loss(self.layers['in']['seg'], self.layers['S']['out'])

    def define_evaluation_metrics(self):
        # Boolean segmentation outputs
        T = tf.greater(self.layers['in']['seg'], self.params['threshold'])  # Ground truth
        P = tf.greater(self.layers['S']['out'], self.params['threshold'])   # Segmentation from S (prediction)

        nT = tf.logical_not(T)
        nP = tf.logical_not(P)

        self.layers['eval']['dice_score'] = 2*(tf.count_nonzero(tf.logical_and(T,P))/(tf.count_nonzero(T)+tf.count_nonzero(P)))
        # True Positive Rate
        self.layers['eval']['sensitivity'] = (tf.count_nonzero(tf.logical_and(T,P))/(tf.count_nonzero(T)))
        self.layers['eval']['specificity'] = (tf.count_nonzero(tf.logical_and(nT,nP))/(tf.count_nonzero(nT)))


    def define_optimizer(self):
        variables_S = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S')
        variables_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C')
        # Also collect moving mean/variances operators for batchnorm
        batchnorm_S = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='S')
        batchnorm_C = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='C')
        self.layers['train']['optimizer_S'] = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate'])
        self.layers['train']['optimizer_C'] = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate'])

        # Loading the last global step (if any).
        # Optimizer needs the corresponding tensor but has to be called after the load() in order to resume its variables
        self._load_last_global_step()
        # We are asking to recalculate moving avg/var of S before each training step
        with tf.control_dependencies(batchnorm_S):
            self.layers['train']['step_S'] = self.layers['train']['optimizer_S'].minimize(self.layers['train']['loss_s'],
                                                                                        global_step=self.layers['train']['global_step_tensor'],
                                                                                        var_list=variables_S)
        with tf.control_dependencies(batchnorm_C):
            self.layers['train']['step_C'] = self.layers['train']['optimizer_C'].minimize(self.layers['train']['loss_c'],
                                                                                    var_list=variables_C)
    def load(self, sess):
        '''
        Loads a network from the 'checkpoint_folder' path. Also loads the current global_step
        :param sess:
        :return:
        '''
        network_loaded = False
        self.layers['train']['global_step'] = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.params['checkpoint_folder'])
        if latest_checkpoint is None:
            # No networks have been saved
            print("No model found, creating a new one...")
            self.layers['train']['global_step'] = 0
            self.layers['train']['global_step_tensor'] = tf.Variable(self.layers['train']['global_step'],
                                                                 name='global_step')
        else:
            self.saver.restore(sess, latest_checkpoint)
            with open(self.params['checkpoint_folder'] + 'global_step', 'r') as gsin:
                self.layers['train']['global_step'] = int(gsin.read())
            network_loaded=True
            print("Loaded model from {} at global step {}".format(self.params['checkpoint_folder'], self.layers['train']['global_step']))
        return network_loaded

    def _load_last_global_step(self):
        try:
            with open(self.params['checkpoint_folder'] + 'global_step', 'r') as gsin:
                self.layers['train']['global_step'] = int(gsin.read())
        except:
            self.layers['train']['global_step'] = 0
        self.layers['train']['global_step_tensor'] = tf.Variable(self.layers['train']['global_step'], name='global_step')

    def save(self, sess):
        os.makedirs(self.params['checkpoint_folder'], exist_ok=True)
        with open(self.params['checkpoint_folder']+'global_step', 'w') as gsout:
            self.layers['train']['global_step'] = sess.run(self.layers['train']['global_step_tensor'])
            gsout.write(str(self.layers['train']['global_step']))
        self.saver.save(sess, self.params['checkpoint_folder']+'model.ckpt', global_step=self.layers['train']['global_step_tensor'])


    def train(self):

        # Load the training dataset and feed the input to the network.
        # NOTICE: Dataset has to be interleaved with as many samples as we intend to call sess.run() with the same data,
        # because every time we call .run the iterator gets incremented
        # The expected behaviour is to call .run for: train_S, train_C, loss_calculation.
        train_dataset = dh.load_dataset('brats2015-Train-all', batch_size=self.params['batch_size'], interleave=3, cast_to=self.params['dtype'])

        # Define an iterator what works for both training and test datasets
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        next_batch = iterator.get_next()

        # Initializers for the iterators
        enable_train_dataset = iterator.make_initializer(train_dataset)

        # Build the model
        self.build_network(next_batch['mri'], next_batch['seg'])

        with tf.Session(config=self.params['session_config']) as sess:
            loaded = self.load(sess)
            if not loaded:
                sess.run(tf.global_variables_initializer())
            self.visualizer = SeganViewer(self, sess, self.params['visualize_folder'])

            global_step=self.layers['train']['global_step']
            while global_step <= self.params['max_iterations']:
                # Initialize the dataset iterator and set the training flag to True
                sess.run([enable_train_dataset, self.layers['ops']['enable_training']])
                print("Training...")
                while True:
                    epoch_start_time = time.time()
                    try:
                        # Training of C and S
                        _ = sess.run(self.layers['train']['step_C'])
                        _ = sess.run(self.layers['train']['step_S'])
                        # Visualizing losses
                        self.visualizer.log(sess, show='train_loss', global_step=global_step, last_time=epoch_start_time)
                        global_step += 1
                    except tf.errors.OutOfRangeError:
                        print("Epoch Ended")
                        break

                # Show a prediction made with training set:
                sess.run([enable_train_dataset, self.layers['ops']['disable_training']])
                self.visualizer.log(sess, show='train', global_step=global_step)
                print("Checkpoint...")
                self.save(sess)



if __name__ == '__main__':
    batchsize = 32

    mri_shape = [batchsize, 240, 240, 1]
    seg_shape = [batchsize, 240, 240, 1]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    segan = SegAN(mri_shape, seg_shape, config=config)
    segan.train()
    print("Done")