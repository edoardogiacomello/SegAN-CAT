import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import dataset_helpers as dh
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Setting allow_growth for gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model running on CPU")


class DeepMRI():
    def __init__(self, batch_size, size, mri_channels, model_name='DeepMRI', max_epochs=100000):
        self.batch_size = batch_size
        self.size = size
        self.mri_shape = (size, size, mri_channels)
        self.label_shape = (size, size, 1)
        self.train_dataset = None
        self.max_epochs = max_epochs
        self.ref_sample = None
        self.model_name = model_name
        self.save_path = 'models/{}/'.format(model_name)
        self.current_epoch = 0
        self.save_every = 1 # If you change this and load a saved model, epoch count will be wrong. Update current_epoch before starting the train.
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    
    def build_model(self, load_model, seed, arch=None, current_epoch=None):
        ''' If load_model is None or False, creates a new model, if is 'last' load the most recent checkpoint, otherwise loads the specified one.
            :param current_epoch: If set, import the history of the given load_model checkpoint into this model. Has effect only if a specific checkpoint is given and no other saved checkpoints are found in the model folder.
        '''
        if seed is not None:
            tf.random.set_seed(seed)
        
        if arch is None:
            import SegAN_IO_GAP_arch as arch
        
        print("Using architecture: {}".format(arch.__name__))
        self.arch = arch
        self.generator = arch.build_segmentor(self.mri_shape)
        self.discriminator = arch.build_critic(self.mri_shape, self.label_shape)
        self.g_optimizer = tf.optimizers.RMSprop(learning_rate=0.00002)
        self.d_optimizer = tf.optimizers.RMSprop(learning_rate=0.00002)
        
        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator, g_optimizer=self.g_optimizer, d_optimizer=self.g_optimizer)
        last_ckpt = tf.train.latest_checkpoint(self.save_path)
        
        self.train_metrics = Logger(self.save_path+'log_train.csv')
        self.valid_metrics = Logger(self.save_path+'log_valid.csv')
        
        if load_model=='last' and last_ckpt is not None:
            print("Latest Checkpoint is: {}".format(last_ckpt))
            self.ckpt.restore(last_ckpt)
            self.current_epoch = self.train_metrics.get_next_epoch()
            print("Loaded model from: {}, next epoch: {}".format(load_model, self.current_epoch))
        else:
            if load_model != 'last':
                print("Loading", load_model)
                self.ckpt.restore(load_model)
                
                if last_ckpt is None:
                    print("W: Your save path/modelname is different from the one you used to save the model.")
                    if current_epoch is not None:
                        self.train_metrics.load_from(os.path.dirname(load_model)+'/log_train.csv', at=current_epoch)
                        self.valid_metrics.load_from(os.path.dirname(load_model)+'/log_valid.csv', at=current_epoch)
                        self.current_epoch = self.train_metrics.get_next_epoch()
                    else:
                        print("W: You are importing a checkpoint with a different model name without indicating a starting epoch. Training history will be reset.")
                    
                else:
                    self.current_epoch = self.train_metrics.get_next_epoch()
                print("Loaded model from: {}, next epoch: {}".format(load_model, self.current_epoch))
            else:
                print("Created new model")
            
        
    
   
    
    def load_dataset(self, dataset, mri_types):
        if dataset not in ['brats', 'bd2decide']:
            raise NotImplementedError("Failed to load dataset {} with modalities {}".format(dataset, ','.join(mri_types)))
        
        if any([d is not None for d in [self.train_dataset, self.validation_dataset, self.test_dataset]]):
            print("Unloading previous dataset")
            del self.train_dataset
            del self.validation_dataset
            del self.test_dataset
            
        
        if dataset == 'bd2decide':
            print("Loading dataset {} with modalities {}".format(dataset, ','.join(mri_types)))
            self.train_dataset = lambda: dh.load_dataset('../datasets/BD2Decide-T1T2_training_crop_mri',
                                    mri_type=mri_types,
                                    random_crop=list(self.mri_shape),
                                    batch_size=self.batch_size,
                                    prefetch_buffer=1,
                                    clip_labels_to=1.0,
                                    infinite=False, 
                                    cache=False
                                    )

            self.validation_dataset = lambda: dh.load_dataset('../datasets/BD2Decide-T1T2_validation_crop_mri',
                                    mri_type=mri_types,
                                    center_crop=list(self.mri_shape),
                                    batch_size=self.batch_size,
                                    prefetch_buffer=1,
                                    clip_labels_to=1.0,
                                    infinite=False,
                                    cache=False
                                    ) 
            
            self.test_dataset =  lambda: dh.load_dataset('../datasets/BD2Decide-T1T2_testing_crop_mri',
                                    mri_type=mri_types,
                                    center_crop=list(self.mri_shape),
                                    batch_size=self.batch_size,
                                    prefetch_buffer=1,
                                    clip_labels_to=1.0,
                                    infinite=False,
                                    cache=False
                                    )
        else:
            if dataset == 'brats':
                print("Loading dataset {} with modalities {}".format(dataset, ','.join(mri_types)) )
                self.train_dataset = lambda: dh.load_dataset('../datasets/brats2015_training_crop_mri',
                                        mri_type=mri_types,
                                        random_crop=list(self.mri_shape),
                                        batch_size=self.batch_size,
                                        prefetch_buffer=1,
                                        clip_labels_to=1.0,
                                        infinite=False, 
                                        cache=False
                                        )

                self.validation_dataset = lambda: dh.load_dataset('../datasets/brats2015_validation_crop_mri',
                                                mri_type=mri_types,
                                                center_crop=list(self.mri_shape),
                                                batch_size=self.batch_size,
                                                prefetch_buffer=1,
                                                clip_labels_to=1.0,
                                                infinite=False, 
                                                cache=False
                                                )
                self.test_dataset = lambda: dh.load_dataset('../datasets/brats2015_testing_crop_mri',
                                                mri_type=mri_types,
                                                center_crop=list(self.mri_shape),
                                                batch_size=self.batch_size,
                                                prefetch_buffer=1,
                                                clip_labels_to=1.0,
                                                infinite=False, 
                                                cache=False
                                                )
            
        self.train_dataset_length = None
        self.validation_dataset_length = None
        self.test_dataset_length = None
        
        # Selecting one random sample from validation set as reference for the predictions
        for row in self.validation_dataset():
            if row['seg'].numpy()[0].any():
                self.ref_sample = row['mri'].numpy()[0], row['seg'].numpy()[0]
                print("Done.")
                break
        
    
    
    @tf.function
    def compute_metrics(self, y_true, y_pred, g_loss, d_loss, threshold=0.5):
        # Boolean segmentation outputs
        # Condition Positive - real positive cases
        CP = tf.greater(y_true, threshold)  # Ground truth
        # Predicted Condition Positive - predicted positive cases
        PCP = tf.greater(y_pred, threshold)   # Segmentation from S (prediction)
        # Codition Negative
        CN = tf.math.logical_not(CP)
        # Predicted Condition Negative
        PCN = tf.math.logical_not(PCP)

        TP = tf.math.count_nonzero(tf.math.logical_and(CP, PCP))
        FP = tf.math.count_nonzero(tf.math.logical_and(CN, PCP))
        FN = tf.math.count_nonzero(tf.math.logical_and(CP, PCN))
        TN = tf.math.count_nonzero(tf.math.logical_and(CN, PCN))

        # TPR/Recall/Sensitivity/HitRate, Probability of detection
        sensitivity = TP/(TP+FN)
        # TNR/Specificity/Selectivity, Probability of false alarm
        specificity = TN/(TN+FP)
        # False Positive Rate / fall-out
        false_positive_rate = 1 - specificity
        # Precision/ Positive predictive value
        precision = TP/(TP+FP)
        # Dice score (Equivalent to F1-Score)
        dice_score = 2*(tf.math.count_nonzero(tf.math.logical_and(CP,PCP))/(tf.math.count_nonzero(CP)+tf.math.count_nonzero(PCP)))
        # (Balanced) Accuracy - Works with imbalanced datasets
        balanced_accuracy = (sensitivity + specificity)/2.0
        
        
        # For debugging the loss..
        smooth_dice_loss = self.arch.smooth_dice_loss(y_true, y_pred)
        mae_distance = tf.reduce_mean(tf.metrics.mae(y_true,y_pred))
        
        
        
        # When editing this also edit the Logger class accordingly
        return [g_loss, d_loss, sensitivity, specificity, false_positive_rate, precision, dice_score, balanced_accuracy, smooth_dice_loss, mae_distance]
        
        
    @tf.function
    def train_step(self, x, y, train_g=True, train_d=True):
        '''
        Performs a training step.
        :param x: batch of training data
        :param y: batch of target data
        :train_g: A tf.constant (Bool) telling if g has to be trained [True]
        :train_d: A tf.constant (Bool) telling if d has to be trained [True]
        '''
        # FIXME: Here Pix2Pix example uses 2 tapes
        #with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        with tf.GradientTape(persistent=True) as tape:
            g_output = self.generator(x, training=train_g)
            d_real, d_real_pred, d_real_gap = self.discriminator([x, y], training=train_d)
            d_fake, f_fake_pred, d_fake_gap = self.discriminator([x, g_output], training=train_d)
            
            # Loss for training gap weights
            d_gap_loss = tf.reduce_mean(tf.abs(1.0 - d_real)) + tf.reduce_mean(tf.abs(d_fake))
            
            g_loss = self.arch.loss_g(d_real, d_fake, g_output, y)
            d_loss = self.arch.loss_d(d_real, d_fake) + d_gap_loss
        if train_g == True:
            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        if train_d == True:
            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        del tape

        return self.compute_metrics(y, g_output, g_loss, d_loss)
    
    @tf.function
    def validation_step(self, x, y):
        g_output = self.generator(x, training=False)
        d_real, d_real_pred, d_real_gap = self.discriminator([x, y], training=False)
        d_fake, f_fake_pred, d_fake_gap = self.discriminator([x, g_output], training=False)
        g_loss = self.arch.loss_g(d_real, d_fake, g_output, y)
        # Loss for training gap weights
        d_gap_loss = tf.reduce_mean(tf.abs(1.0 - d_real)) + tf.reduce_mean(tf.abs(d_fake))
        d_loss = self.arch.loss_d(d_real, d_fake) + d_gap_loss
        return self.compute_metrics(y, g_output, g_loss, d_loss)
        
    def alternated_training(self, n_gen, n_disc, start_with='d'):
        '''
        Iterator returning a tuple (train_g, train_d) of Bools indicating if g or d has to be trained this step, in an alternated fashion.
        :param n_gen: how many times g has to be trained before training d        
        :param n_disc: how many times d has to be trained before training g
        :start_with: String, can be either 'g' or 'd'.
        '''        
        switch = start_with
        c = 0
        tg, td = True, True
        while True:
            if switch=='d':
                tg, td = False, True
                c += 1
                if c >= n_disc:
                    switch = 'g'
                    c = 0
            elif switch=='g':
                tg, td = True, False
                c += 1
                if c >= n_gen:
                    switch = 'd'
                    c = 0
            yield tg, td

    
    def train(self, alternating_steps=None, tracked_metrics=['dice_score']):
        ''' 
            Train the network, saving metrics every epoch and saving the best model on the tracked metrics.
            Supports alternating training. 
            :param alternating_steps: (steps_g, steps_d) or None. How many steps each network has to be trained before starting training the other. \
            If None, both network are trained each step on the same batch of data (train happens independently on G and D in any case).
            Ie. None: G is trained on X1, then D on X1, G on X2, D on X2...
            If (1, 1), G is trained on X1, then D on X2, G on X3, D on X4...
            
        '''
        net_switch = self.alternated_training(alternating_steps[0], alternating_steps[1]) if alternating_steps is not None else None
        
        for e in range(self.current_epoch, self.max_epochs):
            self.train_progress = tk.utils.Progbar(self.train_dataset_length, stateful_metrics=self.train_metrics.metrics_names)
            # Training Step
            for i, row in enumerate(self.train_dataset()):
                # Alternated training
                if net_switch is None or (e == self.current_epoch and i==0):
                    # The fist step we train both g and d for initializing the needed tensors
                    train_g, train_d = True, True
                else:
                    train_g, train_d = next(net_switch)
                # Train step and metric logging
                train_metrics = self.train_step(row['mri'], row['seg'], train_g=train_g, train_d=train_d)
                self.train_metrics.update(train_metrics, self.train_progress, i)
            self.train_dataset_length = i+1
            
            
            
            # Validation Step
            self.valid_progress = tk.utils.Progbar(self.validation_dataset_length, stateful_metrics=self.valid_metrics.metrics_names)
            for i, row in enumerate(self.validation_dataset()):
                valid_metrics = self.validation_step(row['mri'], row['seg'])
                self.valid_metrics.update(valid_metrics, self.valid_progress, i)
            self.validation_dataset_length = i+1
            # Updating the logs
            epoch_train_metrics = self.train_metrics.on_epoch_end(e)
            epoch_valid_metrics = self.valid_metrics.on_epoch_end(e)
            
            # Saving the model if it's the best according some metrics
            is_best = self.valid_metrics.is_best_model(tracked_metrics)
            for m, best in zip(tracked_metrics, is_best):
                if best:
                    print("Found new best model for {}, saving...".format(m))
                    if self.ref_sample is not None:
                        self.show_prediction(self.ref_sample, save_to=self.save_path+self.model_name+"best_{}_{}.png".format(m, e))
                    self.ckpt.save(self.save_path+"best_{}_{}".format(m, e))
               
    def evaluate(self, csv_path, dataset='validation'):
        ''' 
        Evaluate the laoded model on the given dataset ('validation' or 'testing'). Dataset must be loaded beforehand.
        :param csv_path: csv to save the results 
        '''
        assert dataset in ['validation', 'testing']
        eval_logger = Logger(csv_path)
        eval_dataset = self.validation_dataset if dataset == 'validation' else self.test_dataset
        self.eval_progress = tk.utils.Progbar(None, stateful_metrics=eval_logger.metrics_names)
        for i, row in enumerate(eval_dataset()):
            eval_metrics = self.validation_step(row['mri'], row['seg'])
            eval_logger.update(eval_metrics, self.eval_progress, i)
        return eval_logger.on_epoch_end(0)
        

    def show_prediction(self, ref_sample, save_to, mri_index=0):
        x, y = ref_sample
        y_pred = self.generator([x[np.newaxis, :]], training=False)
        
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x[..., mri_index])
        plt.subplot(1, 3, 2)
        plt.imshow(y_pred[0,..., 0])
        plt.subplot(1, 3, 3)
        plt.imshow(y[..., 0])
        plt.show()
        if save_to is not None:
            plt.savefig(save_to)
        

        
class Logger():
    ''' Class for keeping running average of the statistics and returns the metrics for the current epoch'''
    def __init__(self, csv_path, start_from=None):
        self.metrics_names = ['loss_g','loss_d','sensitivity','specificity','false_positive_rate','precision','dice_score','balanced_accuracy', 'smooth_dice_loss', 'mae']
        # Criteria for determining under what condition a given metric value x is better then y.
        self.criteria = {'loss_g': 'max',
                         'loss_d': 'min',
                         'sensitivity': 'max', # TP rate
                         'specificity': 'max', # TN rate
                         'false_positive_rate': 'min',
                         'precision': 'max',
                         'dice_score': 'max',
                         'balanced_accuracy': 'max',
                         'smooth_dice_loss':'min',
                         'mae':'min'}
        
        self.batch_means = [tk.metrics.Mean() for name in self.metrics_names]
        
        self.csv_path = csv_path
        if os.path.dirname(csv_path) != '':
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if os.path.isfile(csv_path):
            self.history = pd.read_csv(csv_path)
            print("Loaded history from {}".format(csv_path))
        else:
            self.history = pd.DataFrame(columns=['epoch']+self.metrics_names)
        
    
    def update(self, current_metrics, progressbar, pbar_step):
        '''
        Updates the metrics for the current epoch. Also updates a progressbar if provided.
        current_metrics: a list of metrics. Keys have to match self.metrics_names
        '''
        # Update epoch averages
        for name, mean, current in zip(self.metrics_names, self.batch_means, current_metrics):
            mean.update_state(current)
    
        if progressbar is not None:
            bar_metrics = list(zip(self.metrics_names, current_metrics))
            progressbar.update(pbar_step, bar_metrics)
        
    def on_epoch_end(self, current_epoch):
        """
        Update the history with the averages of the last epoch and initializes a new one.
        """
        results = [mean.result().numpy() for mean in self.batch_means]
        for mean in self.batch_means:
            mean.reset_states()
        # Update and save the history
        update_dict = dict(zip(self.metrics_names, results))
        update_dict['epoch'] = current_epoch
        update_dict['datetime'] = datetime.datetime.now()
        self.history = self.history.append(update_dict, ignore_index=True)
        self.save()
        return results
    
    def is_best_model(self, metrics):
        ''' Returns whether the last registered epoch is the best according to the given list of metrics.'''
        results = list()
        for m in metrics:
            last_entry = self.history[m].tail(1)
            past_entries = self.history[m].iloc[:len(self.history[m])-1]
            if self.criteria[m] == 'min':
                best = (last_entry <= past_entries.min()).values.item()
            elif self.criteria[m] == 'max':
                best = (last_entry >= past_entries.max()).values.item()
            results.append(best)
        return results
        
    def get_next_epoch(self):
        '''Returns the last logged epoch index. 
        This is needed since TF checkpoint keeps tracks of the epoch only when saving the best model, but we keep track of every epoch '''
        if len(self.history) > 0:
            return int(self.history.tail(1)['epoch']) + 1
        else:
            return 0
        
        
    def load_from(self, csv_path, at):
        ''' Imports an history log from csv truncating it after <at> epochs. Use this function when loading a checkpoint into a new model and you want to keep previous history'''
        self.history = pd.read_csv(csv_path)
        self.history = self.history[self.history['epoch'] <= at]
        print("Imported history from {} at epoch {}".format(csv_path, at))
        
        
    def save(self, csv_path=None):
        path = csv_path or self.csv_path
        self.history.to_csv(self.csv_path, index=False)
        
        
        
        
        
        
        