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
        self.summaries = {'train_loss':[],  # Only logs the training loss and metrics (faster)
                          'train':[], # Shows the images during training (slower)
                          'test_loss':[], # Only logs the test loss and metrics (faster)
                          'test':[], # Shows the images during testing (slower)
                          'predict':[], # Shows the images during prediction. Only requires an input MRI
                          }

        # Creating summaries to show
        loss_c = tf.summary.scalar('loss_c', self.segan.layers['train']['loss_c'])
        loss_s = tf.summary.scalar('loss_s', self.segan.layers['train']['loss_s'])
        dice_score = tf.summary.scalar('dice_score', self.segan.layers['eval']['dice_score'])
        sensitivity = tf.summary.scalar('sensitivity', self.segan.layers['eval']['sensitivity'])
        specificity = tf.summary.scalar('specificity', self.segan.layers['eval']['specificity'])

        mri_input = tf.summary.image('mri_input', self.segan.layers['in']['mri'], max_outputs=max_img_outputs)
        seg_input = tf.summary.image('seg_input', self.segan.layers['in']['seg'], max_outputs=max_img_outputs)
        s_output = tf.summary.image('s_output', self.segan.layers['S']['out'], max_outputs=max_img_outputs)
        d_input = tf.summary.image('mri_masked', self.segan.layers['C_s']['mri_masked'], max_outputs=max_img_outputs)

        # Defining summaries for the network weights
        s_vars = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S') if 'W' in w.name]
        c_vars = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C') if 'W' in w.name]
        w_as_img_s = [tf.summary.image('weights/'+w.name, self._rearrange_filters(w)) for w in s_vars]
        w_as_img_c = [tf.summary.image('weights/'+w.name, self._rearrange_filters(w)) for w in c_vars]



        # Defining which summaries to calculate for each case
        self.summaries['train_loss'].append(loss_c)
        self.summaries['train_loss'].append(loss_s)
        self.summaries['train_loss'].append(dice_score)
        self.summaries['train_loss'].append(sensitivity)
        self.summaries['train_loss'].append(specificity)

        self.summaries['train'].append(mri_input)
        self.summaries['train'].append(seg_input)
        self.summaries['train'].append(s_output)
        self.summaries['train'].append(d_input)
        self.summaries['train'].append(w_as_img_s)
        self.summaries['train'].append(w_as_img_c)

        self.summaries['test_loss'].append(loss_c)
        self.summaries['test_loss'].append(loss_s)
        self.summaries['test_loss'].append(dice_score)
        self.summaries['test_loss'].append(sensitivity)
        self.summaries['test_loss'].append(specificity)

        self.summaries['test'].append(mri_input)
        self.summaries['test'].append(seg_input)
        self.summaries['test'].append(s_output)
        self.summaries['test'].append(d_input)

        self.summaries['predict'].append(mri_input)
        self.summaries['predict'].append(s_output)
        self.summaries['predict'].append(d_input)
        # TODO: Add D output

        # Merge the summaries
        self.merged = {run: tf.summary.merge(self.summaries[run]) for run in self.summaries.keys()}
        self.train_writer = tf.summary.FileWriter(output_folder+'train/', sess.graph)
        self.test_writer = tf.summary.FileWriter(output_folder+'test/', sess.graph)
        self.predict_writer = tf.summary.FileWriter(output_folder+'prediction/', sess.graph)

    def _rearrange_filters(self, input):
        '''
        Given a tensor of weights of shape (K, K, C, N)
        where K is the filter size, C are the input channels and N are the number of filters,
        returns a 2D tensor by concatenating every KxK block on C columns and N rows.
        :return: Image representing the filters
        '''

        W, H, C, R = input.shape
        return tf.transpose(tf.reshape(tf.transpose(input, perm=[3, 1, 2, 0]), (W * C, H * R)))



    def get_ops(self, show):
        '''
        Returns the tensors that have to be run to compute and visualize the training loss. These loss have to be passed to the log function after evaluation
        :param show: What type of run has to be logged. Can be one of train_loss, train, test_loss, test, predict.
        :return:
        '''
        if show in ['train_loss', 'test_loss']:
            return [self.merged[show], self.segan.layers['train']['loss_c'], self.segan.layers['train']['loss_s']]
        else:
            return self.merged[show]

    def log(self, session, show, global_step, last_time=None):
        '''
        Logs network loss or ouputs, depending on the chosen run and values fed.
        :param session: the current tensorflow session
        :param show: What type of run has to be logged. Can be one of train_loss, train, test_loss, test, predict.
        :param global_step: (Number) current iteration index of the network.
        :return: None
        '''
        assert show in self.summaries.keys(), "Run type not found. Please select one from {}".format(self.summaries.keys())
        if show in ['train_loss', 'test_loss']:
            summary, loss_c, loss_s = session.run([self.merged[show], self.segan.layers['train']['loss_c'], self.segan.layers['train']['loss_s']])
            writer = self.train_writer if show=='train_loss' else self.test_writer
            writer.add_summary(summary, global_step)

            time_log = "| time={}".format(time.time() - last_time) if last_time is not None else ""
            print("{}: {} | c={} | s={} {}".format(global_step, show, loss_c, loss_s, time_log))
            return
        if show is 'train':
            writer = self.train_writer
        if show is 'test':
            writer = self.test_writer
        if show is 'predict':
            writer = self.predict_writer
        summary = session.run(self.merged[show])
        writer.add_summary(summary, global_step=global_step)




