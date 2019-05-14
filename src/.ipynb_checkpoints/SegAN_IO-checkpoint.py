import tensorflow as tf
import dataset_helpers as dh
import os, math
from SegANViewer import SeganViewer
import time

class SegAN_IO():

    def __init__(self, mri_shape, seg_shape, config=None, run_name='last'):
        self.params = {
                        'session_config':config,
                        'mri_shape':mri_shape,
                        'seg_shape':seg_shape,
                        'clip': {'C': True, 'S': False, 'constraint':lambda x: tf.clip_by_value(x, clip_value_min=-0.05, clip_value_max=0.05)},
                        'dtype': tf.float32,
                        'learning_rate': 0.00002,
                        'max_iterations': 2000000,
                        'save_every_itn': 492,
                        'batch_size': mri_shape[0],
                        'checkpoint_folder': '../models/SegAN/{}_model/'.format(run_name),
                        'visualize_folder': '../models/SegAN/{}_visualize/'.format(run_name),
                        'profiler_folder': '../models/SegAN/{}_profiler/'.format(run_name),
                        'threshold': 0.5, # Treshold for a segmentation to be considered as 1 or 0
                       }
        # Define a dictionary that will contain the layers for faster access
        self.layers = {'in':{}, # Inputs to the network
                       'S':{}, # Segmentor
                       'C_gt':{}, # Critic layers with ground truth as input
                       'C_s':{}, # Critic layers with S output as input (parameters are shared with C_gt)
                       'train':{}, # Operations for training
                       'view':{}, # Operations for tensorboard visualization
                       'eval':{}, # Tensors containing evaluation metrics
                       }

    def build_network(self, input_mri, true_seg, session, load_checkpoint=None):
        '''
        Build a network using as input the tensors specified as parameters.
        If load_checkpoint is None, the last available is loaded. If no models are available, a new network is initialized

        The value of "training" must be fed as a feed_dict in order to enable/disable batch normalization learning
        :param input_mri: Tensor for the MRI to use as S and C input
        :param true_seg: Tensor for the ground truth segmentation in input to C
        :return:
        '''
        # This flag is used for enabling/disabling the training mode for batch normalization.
        # When true, BN layers use mean/var for the current batch, otherwise they use the learned ones.
        self.layers['in']['training'] = tf.get_variable("training", dtype=tf.bool, trainable=False, initializer=False)

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
        # Loading the global step from checkpoint filename, otherwise the last available.
        self._load_last_global_step(override_with_checkpoint_gs=load_checkpoint)
        # Defining optimizer (need the global_step_tensor to be defined)
        self.define_optimizer()

        # Defining the saver. Every tensor defined afterwards is not included in the model checkpoint
        self.saver = tf.train.Saver(max_to_keep=None)

        # Load the model variables or initialize a new network
        if not self.load(session, checkpoint=load_checkpoint):
            session.run(tf.global_variables_initializer())

    def _conv2d(self, input, kern_size, filters, strides, padding='SAME', name=None, clip=False, reuse=False):
        def get_filters(shape, clip):
            # This initialization is taken from SegAN code
            initializer = tf.random_normal(shape, mean=0.0, stddev=math.sqrt(2./(shape[0]*shape[1]*shape[2])), dtype=self.params['dtype'])
            return tf.get_variable(name="W", initializer=initializer, constraint=self.params['clip']['constraint'] if clip else None, dtype=self.params['dtype'], trainable=True)
        filter_shape = (kern_size, kern_size, input.shape[-1].value, filters)
        with tf.variable_scope(name_or_scope=name, default_name="conv2d", reuse=reuse):
            # No need for bias as the batchnorm already account for that
            return tf.nn.conv2d(input, filter=get_filters(filter_shape, clip), strides=[1,strides, strides,1], padding=padding, name=name)


    def _batchnorm(self, x):
        return tf.layers.batch_normalization(x, epsilon=1e-5, momentum = 0.1, training=self.layers['in']['training'])

    def _resize(self, input, factor_wrt_input=2.0):
        new_shape = [int(self.params['mri_shape'][1]*factor_wrt_input), int(self.params['mri_shape'][1]*factor_wrt_input)]
        return tf.cast(tf.image.resize_bilinear(input, size=new_shape), self.params['dtype'])

    def build_segmentor(self, scope='S', reuse=False):
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
            self.layers['S']['enc_1'] = tf.nn.leaky_relu(self._conv2d(self.layers['in']['mri'], kern_size=4, filters=64, strides=2, reuse=reuse, name='enc_1'))
            self.layers['S']['enc_2'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_1'], kern_size=4, filters=128, strides=2, clip=self.params['clip']['S'], reuse=reuse, name='enc_2')))
            self.layers['S']['enc_3'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_2'], kern_size=4, filters=256, strides=2, clip=self.params['clip']['S'], reuse=reuse, name='enc_3')))
            self.layers['S']['enc_4'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers['S']['enc_3'], kern_size=4, filters=512, strides=2, clip=self.params['clip']['S'], reuse=reuse, name='enc_4')))
            self.layers['S']['dec_3'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(self.layers['S']['enc_4'], factor_wrt_input=1/8), kern_size=3, filters=256, strides=1, clip=self.params['clip']['S'], reuse=reuse, name='dec_3')))
            self.layers['S']['dec_2'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(tf.add(self.layers['S']['dec_3'], self.layers['S']['enc_3']), factor_wrt_input=1/4), kern_size=3, filters=128, strides=1, clip=self.params['clip']['S'], reuse=reuse, name='dec_2')))
            self.layers['S']['dec_1'] = tf.nn.relu(self._batchnorm(self._conv2d(self._resize(tf.add(self.layers['S']['dec_2'], self.layers['S']['enc_2']), factor_wrt_input=1/2), kern_size=3, filters=64, strides=1, clip=self.params['clip']['S'], reuse=reuse, name='dec_1')))
            self.layers['S']['out'] = tf.sigmoid(self._conv2d(self._resize(tf.add(self.layers['S']['dec_1'], self.layers['S']['enc_1']), factor_wrt_input=1.0), kern_size=3, filters=self.params['seg_shape'][-1], strides=1, clip=self.params['clip']['S'], name='out_unbound', reuse=reuse), name='out')
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
            self.layers[C]['mri_masked'] = tf.concat([self.layers['in']['mri'], input_seg], axis=-1, name='mri_masked')
            self.layers[C]['enc_1'] = tf.nn.leaky_relu(self._conv2d(self.layers[C]['mri_masked'], kern_size=4, filters=64, strides=2, clip=self.params['clip']['C'], reuse=reuse, name='enc_1'))
            self.layers[C]['enc_2'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers[C]['enc_1'], kern_size=4, filters=128, strides=2, clip=self.params['clip']['C'], reuse=reuse, name='enc_2')))
            self.layers[C]['enc_3'] = tf.nn.leaky_relu(self._batchnorm(self._conv2d(self.layers[C]['enc_2'], kern_size=4, filters=256, strides=2, clip=self.params['clip']['C'], reuse=reuse, name='enc_3')))

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
            dice = tf.divide(2. * intersection + epsilon, tf.reduce_sum(x) + tf.reduce_sum(y) + epsilon, name="dice_score")
            return 1 - dice/self.params['batch_size']  # from  github.com/YuanXue1993/SegAN/blob/master/train.py

        self.layers['train']['loss_c'] =  -tf.reduce_mean(tf.abs(self.layers['C_gt']['out']-self.layers['C_s']['out']))
        self.layers['train']['loss_s'] =   tf.reduce_mean(tf.abs(self.layers['C_gt']['out']-self.layers['C_s']['out'])) + smooth_dice_loss(self.layers['in']['seg'], self.layers['S']['out'])

    def define_evaluation_metrics(self):


        # Boolean segmentation outputs
        # Condition Positive - real positive cases
        CP = tf.greater(self.layers['in']['seg'], self.params['threshold'])  # Ground truth
        # Predicted Condition Positive - predicted positive cases
        PCP = tf.greater(self.layers['S']['out'], self.params['threshold'])   # Segmentation from S (prediction)
        # Codition Negative
        CN = tf.logical_not(CP)
        # Predicted Condition Negative
        PCN = tf.logical_not(PCP)

        TP = tf.count_nonzero(tf.logical_and(CP, PCP))
        FP = tf.count_nonzero(tf.logical_and(CN, PCP))
        FN = tf.count_nonzero(tf.logical_and(CP, PCN))
        TN = tf.count_nonzero(tf.logical_and(CN, PCN))

        # TPR/Recall/Sensitivity/HitRate, Probability of detection
        self.layers['eval']['sensitivity'] = TP/(TP+FN)
        # TNR/Specificity/Selectivity, Probability of false alarm
        self.layers['eval']['specificity'] = TN/(TN+FP)
        # False Positive Rate / fall-out
        self.layers['eval']['false_positive_rate'] = 1 - self.layers['eval']['specificity']
        # Precision/ Positive predictive value
        self.layers['eval']['precision'] = TP/(TP+FP)
        # Dice score (Equivalent to F1-Score)
        self.layers['eval']['dice_score'] = 2*(tf.count_nonzero(tf.logical_and(CP,PCP))/(tf.count_nonzero(CP)+tf.count_nonzero(PCP)))
        # (Balanced) Accuracy - Works with imbalanced datasets
        self.layers['eval']['balanced_accuracy'] = (self.layers['eval']['sensitivity'] + self.layers['eval']['specificity'])/2


    def define_optimizer(self):
        variables_S = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S')
        variables_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C')
        # Also collect moving mean/variances operators for batchnorm
        batchnorm_S = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='S')
        batchnorm_C = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='C')
        self.layers['train']['optimizer_S'] = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate'])
        self.layers['train']['optimizer_C'] = tf.train.RMSPropOptimizer(learning_rate=self.params['learning_rate'])

        # We are asking to recalculate moving avg/var of S before each training step
        with tf.control_dependencies(batchnorm_S):
            self.layers['train']['step_S'] = self.layers['train']['optimizer_S'].minimize(self.layers['train']['loss_s'],
                                                                                        global_step=self.layers['train']['global_step_tensor'],
                                                                                        var_list=variables_S)
        with tf.control_dependencies(batchnorm_C):
            self.layers['train']['step_C'] = self.layers['train']['optimizer_C'].minimize(self.layers['train']['loss_c'],
                                                                                    var_list=variables_C)
    def load(self, sess, checkpoint=None):
        '''
        Loads a network from the 'checkpoint_folder' path (if any).
        This function expect the the current self.layers['train']['global_step'] has been already set by _load_global_step()
        :param sess:
        :param checkpoint: Try to load the corresponding checkpoint. If None, load the latest in the folder.
        :return: True if the network has been loaded successfully. False if no checkpoint has been found and a new network has been created
        '''
        network_loaded = False
        latest_checkpoint = checkpoint if checkpoint is not None else tf.train.latest_checkpoint(self.params['checkpoint_folder'])
        if latest_checkpoint is None:
            # No networks have been saved before
            print("No model found, creating a new one...")
        else:
            self.saver.restore(sess, latest_checkpoint)
            network_loaded=True
            print("Loaded model from {} at global step {}".format(self.params['checkpoint_folder'], self.layers['train']['global_step']))
        return network_loaded

    def _load_last_global_step(self, override_with_checkpoint_gs=None):
        '''
        Loads the last global step from the global_step file (if any) and creates the corresponding tensor.
        If override_with_checkpoint_gs is not None, then the global step provided by the checkpoint name is used as global step.
        This is useful if you are loading a network from a previous state that is not the last one.
        :return: None. Has side effects.
        '''
        if override_with_checkpoint_gs is not None:
            self.layers['train']['global_step'] = int(os.path.basename(override_with_checkpoint_gs).split('-')[-1])
        else:
            try:
                with open(self.params['checkpoint_folder'] + 'global_step', 'r') as gsin:
                    self.layers['train']['global_step'] = int(gsin.read())
            except:
                print("Failed to load last globalstep. Initializing a new run...")
                self.layers['train']['global_step'] = 0
        self.layers['train']['global_step_tensor'] = tf.get_variable(name='global_step', initializer=self.layers['train']['global_step'])

    def save(self, sess, best=False):
        os.makedirs(self.params['checkpoint_folder'], exist_ok=True)
        with open(self.params['checkpoint_folder']+'global_step', 'w') as gsout:
            self.layers['train']['global_step'] = sess.run(self.layers['train']['global_step_tensor'])
            gsout.write(str(self.layers['train']['global_step']))

        print("Saving at global step {}".format(self.layers['train']['global_step']))
        if not best:
            self.saver.save(sess, self.params['checkpoint_folder']+'model.ckpt', global_step=self.layers['train']['global_step_tensor'])
        else:
            self.saver.save(sess, self.params['checkpoint_folder']+'model-best.ckpt', global_step=self.layers['train']['global_step_tensor'])

    def train(self, seed=None):

        # Load the training dataset and feed the input to the network.
        # NOTICE: Dataset has to be interleaved with as many samples as we intend to call sess.run() with the same data,
        # because every time we call .run the iterator gets incremented
        # The expected behaviour is to call .run for: train_S, train_C, loss_calculation.

        # Since BRATS does contain GT for testing, we train on "tcia" data and test on 2013
        
        train_dataset = dh.load_dataset('../datasets/brats2015-Train-all_training_crop_mri',
                                        mri_type=['MR_T1c', 'MR_T2', 'MR_Flair'],
                                        random_crop=[160,160,3],
                                        batch_size=self.params['batch_size'],
                                        prefetch_buffer=3,
                                        cast_to=self.params['dtype'],
                                        clip_labels_to=1.0,
                                        infinite=True,
                                        interleave=3
                                        )

        validation_dataset = dh.load_dataset('../datasets/brats2015-Train-all_validation_crop_mri',
                                             mri_type=['MR_T1c', 'MR_T2', 'MR_Flair'],
                                             center_crop=[160,160,3],
                                             batch_size=self.params['batch_size'],
                                             prefetch_buffer=1,
                                             cast_to=self.params['dtype'],
                                             clip_labels_to=1.0,
                                             infinite = True,
                                             interleave=1
                                        )
        
        tensorboard_datasets = dh.load_dataset('../datasets/brats2015-Train-all_validation_crop_mri',
                                             mri_type=['MR_T1c', 'MR_T2', 'MR_Flair'],
                                             center_crop=[160,160,3],
                                             batch_size=self.params['batch_size'],
                                             prefetch_buffer=1,
                                             cast_to=self.params['dtype'],
                                             clip_labels_to=1.0,
                                             interleave=1,
                                             take_only=self.params['batch_size'],
                                             shuffle=False,
                                             infinite=True
                                        )

        # Define a "feedable" iterator of a string handle that selects which dataset to use
        use_dataset=tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(use_dataset, train_dataset.output_types, train_dataset.output_shapes)
        next_batch = iterator.get_next()

        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = validation_dataset.make_initializable_iterator()
        tboard_iterator = tensorboard_datasets.make_initializable_iterator()

        # Initializers for the datasets
        reset_train_iter = train_iterator.initializer
        reset_validation_iter = valid_iterator.initializer
        reset_tboard_iter = tboard_iterator.initializer


        with tf.Session(config=self.params['session_config']) as sess:
            # Setting the Graph-level seed
            if seed is not None:
                tf.set_random_seed(seed)

            # Build the model
            self.build_network(next_batch['mri'], next_batch['seg'], session=sess)

            self.visualizer = SeganViewer(self, sess, self.params['visualize_folder'])

            # Handles to switch between datasets
            use_train_dataset = sess.run(train_iterator.string_handle())
            use_valid_dataset = sess.run(valid_iterator.string_handle())
            use_tboard_dataset = sess.run(tboard_iterator.string_handle())
            # Handle to switch training mode for BatchNorm
            is_training = self.layers['in']['training']

            while self.layers['train']['global_step'] <= self.params['max_iterations']:
                # Initialize the dataset iterators and set the training flag to True (for BatchNorm)
                sess.run([reset_train_iter, reset_validation_iter, reset_tboard_iter])
                print("Training...")

                for i in range(self.params['save_every_itn']):
                    epoch_start_time = time.time()
                    # Training of C and S
                    _ = sess.run(self.layers['train']['step_C'], feed_dict={use_dataset: use_train_dataset, is_training: True})
                    _ = sess.run(self.layers['train']['step_S'], feed_dict={use_dataset: use_train_dataset, is_training: True})
                    # Logging batch loss [Stored in SeganViewer]
                    self.visualizer.log(sess, show='train_metrics', feed_dict={use_dataset: use_train_dataset, is_training: True}, last_time=epoch_start_time)
                    self.visualizer.log(sess, show='test_metrics',  feed_dict={use_dataset: use_valid_dataset, is_training: False}, last_time=epoch_start_time)
                    self.layers['train']['global_step'] += 1

                
                
                # Show a prediction made with a reference sample and plot the epoch metrics:
                # Show the differences by using trained and current BatchNorm weights.
                self.visualizer.log(sess, show='train', feed_dict={use_dataset: use_tboard_dataset, is_training: True})
                best_metric = self.visualizer.log(sess, show='test', feed_dict={use_dataset: use_tboard_dataset, is_training: False}, monitor={'dice_score': 0.7, 'balanced_accuracy': 0.7})
                print("Epoch Ended")
                has_to_save = any([m for m in best_metric.values()])
                if has_to_save:
                    print("Checkpoint...")
                    self.save(sess, best=True)
                




