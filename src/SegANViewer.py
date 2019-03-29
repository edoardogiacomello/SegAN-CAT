import tensorflow as tf
import time
class SeganViewer():
    def __init__(self, segan, sess, output_folder, max_img_outputs=12):
        '''
        Helper for Tensorboard visualization and logging the metrics on disk
        :param segan:
        :param sess:
        :param output_folder:
        '''
        self.output_folder = output_folder
        self.segan = segan
        self.summaries = {'train_metrics':[],  # Only logs the training loss and metrics (faster)
                          'train_img':[], # Shows the images during training (slower)
                          'test_metrics':[], # Only logs the test loss and metrics (faster)
                          'test_img':[], # Shows the images during testing (slower)
                          }

        # Creating accumulators for displaying the average loss
        self.current_batch_stats = {run:{
                                    'loss_c': list(),
                                    'loss_s': list(),
                                    'sensitivity': list(),
                                    'specificity': list(),
                                    'false_positive_rate': list(),
                                    'precision': list(),
                                    'dice_score': list(),
                                    'balanced_accuracy':list()
                                    } for run in ['train_metrics', 'test_metrics']}

        # These are the tensors that gets evaluated for each batch
        self.scalar_metrics = {
                                    'loss_c': self.segan.layers['train']['loss_c'],
                                    'loss_s': self.segan.layers['train']['loss_s'],
                                    'sensitivity': self.segan.layers['eval']['sensitivity'],
                                    'specificity': self.segan.layers['eval']['specificity'],
                                    'false_positive_rate': self.segan.layers['eval']['false_positive_rate'],
                                    'precision': self.segan.layers['eval']['precision'],
                                    'dice_score': self.segan.layers['eval']['dice_score'],
                                    'balanced_accuracy':self.segan.layers['eval']['balanced_accuracy']
        }
        
        # We build some placeholders that will be evaulated after each epoch to display the averages in TB
        self.average_tensors = {'avg_loss_c': tf.placeholder(tf.float32),
                                'avg_loss_s': tf.placeholder(tf.float32),
                                'avg_sensitivity': tf.placeholder(tf.float32),
                                'avg_specificity': tf.placeholder(tf.float32),
                                'avg_false_positive_rate': tf.placeholder(tf.float32),
                                'avg_precision': tf.placeholder(tf.float32),
                                'avg_dice_score': tf.placeholder(tf.float32),
                                'avg_balanced_accuracy': tf.placeholder(tf.float32)
                               }
        
        # Creating scalar summaries for each metric (and calculating the mean)
        self.avg_summaries = [tf.summary.scalar(k, tf.reduce_mean(self.average_tensors[k])) for k in self.average_tensors.keys()]
        

        mri_input = tf.summary.image('mri_input', self.segan.layers['in']['mri'], max_outputs=max_img_outputs)
        seg_input = tf.summary.image('seg_input', self.segan.layers['in']['seg'], max_outputs=max_img_outputs)
        s_output = tf.summary.image('s_output', self.segan.layers['S']['out'], max_outputs=max_img_outputs)
        d_input = tf.summary.image('mri_masked', self.segan.layers['C_s']['mri_masked'], max_outputs=max_img_outputs)

        # Defining summaries for the network weights
        s_vars = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S') if 'W' in w.name]
        c_vars = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C') if 'W' in w.name]
        w_as_img_s = [tf.summary.image('weights/'+w.name, self._rearrange_filters(w)) for w in s_vars]
        w_as_img_c = [tf.summary.image('weights/'+w.name, self._rearrange_filters(w)) for w in c_vars]



        # Defining the merged summaries that get printed in TB
        self.summaries['train_metrics'] = self.avg_summaries
        self.summaries['test_metrics'] = self.avg_summaries
        
        self.summaries['train_img'].append(mri_input)
        self.summaries['train_img'].append(seg_input)
        self.summaries['train_img'].append(s_output)
        self.summaries['train_img'].append(d_input)
        self.summaries['train_img'].append(w_as_img_s)
        self.summaries['train_img'].append(w_as_img_c)

        self.summaries['test_img'].append(mri_input)
        self.summaries['test_img'].append(seg_input)
        self.summaries['test_img'].append(s_output)
        self.summaries['test_img'].append(d_input)

        # Merge the summaries into writers
        self.merged = {run: tf.summary.merge(self.summaries[run]) for run in self.summaries.keys()}
        self.train_writer = tf.summary.FileWriter(output_folder+'train/', sess.graph)
        self.test_writer = tf.summary.FileWriter(output_folder+'test/', sess.graph)
        #self.predict_writer = tf.summary.FileWriter(output_folder+'prediction/', sess.graph)

        
        
    def _rearrange_filters(self, input):
        '''
        Given a tensor of weights of shape (K, K, C, N)
        where K is the filter size, C are the input channels and N are the number of filters,
        returns a 2D tensor by concatenating every KxK block on C columns and N rows.
        :return: Image representing the filters
        '''

        W, H, C, R = input.shape
        return tf.reshape(tf.transpose(input, perm=[3, 1, 2, 0]), (1, W * C, H * R, 1))


    def log(self, session, show, feed_dict, last_time=None):
        '''
        Logs network loss or ouputs, depending on the chosen run and values fed.
        :param session: the current tensorflow session
        :param show: What type of run has to be logged. 
        Can be one of:
        - train_metrics: Evaulate and keep the train metrics for the current batch
        - test_metrics: Evaulate and keep the test metrics for the current batch
        - train: writes the result to tensorboard. Also displays image data
        - test: writes the result to tensorboard. Also displays image data
        :param feed_dict: feed dictionary to use when evaluating the models, if any
        :return: None
        '''

        assert show in ['train_metrics', 'test_metrics', 'test', 'train'], "Run type not found. Please select one from {}".format(self.summaries.keys())
        if show in ['train_metrics', 'test_metrics']:
            # This get called after each batch
            
            last_values = session.run(self.scalar_metrics, feed_dict=feed_dict)
            # Adding last values to the list
            for k in last_values.keys():
                self.current_batch_stats[show][k].append(last_values[k])

            time_log = "| time={}".format(time.time() - last_time) if last_time is not None else ""
            text=" | ".join(["{}={}".format(k, last_values[k]) for k in last_values.keys()])
            print("{} {} time:{}: {} - ".format(self.segan.layers['train']['global_step'], show, time_log, text))
            return
        if show is 'train':
            writer = self.train_writer
        if show is 'test':
            writer = self.test_writer
        
        # Write the summary
        # Show the images
        img_summary = session.run(self.merged[show+'_img'], feed_dict=feed_dict)
        writer.add_summary(img_summary, global_step=self.segan.layers['train']['global_step'])

        # Feed the average of batches epochs into tensorflow
        metric_log = self.current_batch_stats[show+'_metrics']
        print(list(metric_log.keys()))
        feed_dict.update({self.average_tensors['avg_{}'.format(metric)]: metric_log[metric] for metric in metric_log.keys()})
        
        avg_summary = session.run(self.merged[show+'_metrics'], feed_dict=feed_dict)
        writer.add_summary(avg_summary, global_step=self.segan.layers['train']['global_step'])

        # Reset the accumulators
        for k in self.current_batch_stats[show+'_metrics']:
            self.current_batch_stats[show+'_metrics'][k] = list()
        



