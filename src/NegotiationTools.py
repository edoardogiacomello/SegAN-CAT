import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


from scipy.ndimage import convolve



class StatisticsLogger():
    def __init__(self):
        self.pd = pd.DataFrame()
        self.tools = NegTools()

    def log_step(self, sample, png_path, seg_path, step, method, status, agreement, proposals, ground_truth, max_steps, agent_names=None, label_names=None,
                 binary_strategy='maximum'):
        '''
        Calculates metrics and logs them in an internal DataFrame.
        If status is 'consensus', then replicates the last row up to "max_steps", for consistent visualization.
        :param sample: index of the current sample that is being analyzed
        :param png_path: path of the image, used for logging purposes
        :param seg_path: path of the segmentation, used for logging purposes
        :param step: current negotiation step. Must start at zero for the first iteration.
        :param method: name of the current negotiation method that is being used
        :param status: current status of the negotiation. Can be 'negotiation' if there's still no consensus, 'consensus' for the only step for which consensus is reached.
        When 'consensus' is found, this function automatically fills the log with new records having the 'padding' status until <max_step> is reached.
        This is necessary for having the same number of rows for each method.
        :param agreement: Binary vector of agreements. Must have shape (H, W, Label)
        :param proposals: Float vector of proposals. Must have shape (Agent, H, W, Label)
        :param ground_truth: Float vector of ground truth. Must have shape (H, W, Label). Typically composed of 0.0 and 1.0 values.
        :param max_steps: Max steps of negotiation to log. Negotiations that reach consensus in a previous step gets padded up to this number of records.
        :param agent_names: Human readeable names for the agents. If None, it will be inferred from the first axis of proposals and they will be called "Agent 0", "Agent 1", etc.
        :param label_names: Human readeable names for the labels. If None, it will be inferred from the first axis of ground_truth and they will be called "Label 0", "Label 1", etc.
        :param binary_strategy: Binarization strategy used for converting float proposals to binary. Use 'maximum' if the proposals sums to 1.0 (softmax output of the network), otherwise 'threshold' (sigmoid output).
        :return: None
        '''
        self.consensus = self.tools.get_consensus(proposals)
        self.mask = np.logical_not(self.consensus)

        if step == 0:
            # Computation to be performed just once per sample
            self.consensus_initial = self.consensus
            self.consensus_start = np.count_nonzero(self.mask)
            self.ground_truth = ground_truth
            self.ground_truth_bool = ground_truth.astype(np.bool)

        # Computation to be performed once per step (as proposals and agreement are different at each call)
        binary_proposals = self.tools.binarize(x=proposals, strategy=binary_strategy)
        binary_agreement = self.tools.binarize(x=agreement, strategy=binary_strategy)

        step_row = {  # This data is the same for each agent performing the same step
            'sample': sample,
            'png_path': png_path,
            'seg_path': seg_path,
            'method': method,
            'step': step,
            'status': status,
            'consensus_start': self.consensus_start,
            'consensus_current': np.count_nonzero(np.logical_not(self.consensus))
            # TODO: Check current consensus
        }

        self.agr_stats = self.compute_statistics(self.ground_truth_bool, binary_agreement,
                                                 prefix='agr_vs_gt_'.format(method), mask=self.mask, label_names=label_names)
        
        # Computation to be performed for each agent (there is one proposal per agent)
        agent_names = ['Agent {}'.format(n) for n in range(proposals.shape[0])] if agent_names is None else agent_names
        for ag_id in range(proposals.shape[0]):
            agent_row = {
                'agent': agent_names[ag_id],
            }

            self.prop_stats = self.compute_statistics(self.ground_truth_bool, binary_proposals[ag_id],
                                                      prefix='prop_vs_gt_'.format(method), mask=self.mask, label_names=label_names)

            # Collecting data
            agent_row.update(step_row)
            agent_row.update(self.agr_stats)
            agent_row.update(self.prop_stats)

            # Logging
            self.pd = self.pd.append(agent_row, ignore_index=True)
        
        if status == 'consensus' and step + 1 < max_steps:
            repetitions = range(step + 1, max_steps)
            filling = pd.concat([self.pd.tail(n=len(AGENT_NAMES))] * len(repetitions), ignore_index=True)
            filling['status'] = 'padding'
            filling['step'] = [l for l in repetitions for k in range(len(agent_names))]
            self.pd = self.pd.append(filling, ignore_index=True)

    def compute_statistics(self, ground_truth, predictions, prefix, label_names=None, mask=None):
        '''
        Compute statistics (using sklearn classification report), eventually masking the inputs.

        :param ground_truth: vector of 
        :param predictions:
        :param prefix: prefix to give to the returned dictionary values
        :param mask:
        :return:
        '''
        labels = range(ground_truth.shape[-1])
        label_names = ['Label {}'.format(n) for n in labels] if label_names is None else label_names
        
        if mask is None or np.all(np.logical_not(mask)):
            mask = np.full(ground_truth.shape[0:2], fill_value=True)

        ground_truth = ground_truth[np.where(mask)].argmax(axis=-1)
        predictions = predictions[np.where(mask)].argmax(axis=-1)

        report = classification_report(y_true=ground_truth,
                                       y_pred=predictions,
                                       labels=labels,
                                       target_names=label_names,
                                       output_dict=True
                                       )
        # Flattening the report
        stats = dict()
        for metric_type, metric_dict in report.items():
            if not isinstance(metric_dict, dict):
                # sometimes classification_report returns an 'accuracy' float instead of 'micro_*' dictionary
                stats[prefix + str(metric_type)] = metric_dict
            else:
                for metric, value in metric_dict.items():
                    stats[prefix + str(metric_type) + '_' + metric] = value

        return stats

    def save(self):
        self.pd.to_csv('results/run_{}.csv'.format(str(datetime.datetime.now())))


class NegTools():
    # This has to be here to support multithreading
    import tensorflow as tf        
        
        
    def binarize(self, x, strategy, treshold=0.5, axis=-1):
        '''
        Returns a binary vector from a floating point represenatation. The strategy control how it is computed.
        :param x:
        :param strategy:
        :param treshold: can be either 'threshold' (only pixels > 0.5 are considered positive) or 'maximum' (only the maximum amongst the prediction is considered as True)
        :param axis:
        :return:
        '''
        assert strategy in ['treshold', 'maximum'], 'Specify a strategy for prediction binarization'
        if strategy == 'treshold':
            return np.greater(x, treshold)
        if strategy == 'maximum':
            return np.equal(x, x.max(axis=axis)[..., np.newaxis])

    def get_consensus(self, proposals_np):
        binary_predictions = np.equal(proposals_np, proposals_np.max(axis=-1)[..., np.newaxis])
        # For each pixel check if there's any label fow which every agent proposes True
        return np.all(binary_predictions, axis=0).any(axis=-1)

    def compute_majority_voting(self, proposals_np, binary_strategy, weights=None):
        '''
        Calculates the Majority voting between the given agent proposals. Ties are resolved by random sampling.
        :param proposals_np: Proposals of the agents. Must have shape (agents, H, W, labels).
        :param binary_strategy: see function "binarize". Use "maximum" if the labels for each prediction sum to one, otherwise "treshold".
        :param weights: Optional array of weights, must have the same shape as the proposals. Each vote is multiplied by its weight before tie breaking.
        :return: A binary vector of shape (H, W, Labels)
        '''
        assert weights is None or weights.shape == proposals_np.shape
        binary_predictions = self.binarize(proposals_np, binary_strategy, axis=-1)
        if weights is None:
            votes = np.count_nonzero(binary_predictions, axis=0)
        else:
            votes = np.sum(weights*binary_predictions.astype(np.float32), axis=0)
        majority = np.equal(votes, votes.max(axis=-1, keepdims=True))
        majority = self.tie_breaking(majority)
        return majority

    def tie_breaking(self, majority, axis=-1):
        random = np.random.uniform(size=majority.shape)
        random_masked = majority.astype(np.uint8) * random
        solution = np.equal(random_masked, np.max(random_masked, axis=axis)[..., np.newaxis])
        # Fixing the cases in which all elements are "False" in the input vector
        coords = np.where(np.all(np.logical_not(majority), axis=axis))
        for coord in zip(*(coords)):
            solution[coord] = np.full_like(majority[coord], fill_value=False)
        return solution

    def mean_proposal(self, proposals_np, binary_strategy):
        mean_float = proposals_np.mean(axis=0)
        return self.binarize(mean_float, strategy=binary_strategy)

    def max_proposal(self, proposals_np, binary_strategy='maximum'):
        ''' Agreement is computed by taking the maximum prediction over all the agents and binarizing the result taking the maximum label for each pixel. Ties are resolved by random sampling'''
        maxagr = np.max(proposals_np, axis=0)
        # Normalization
        maxagr = maxagr / np.sum(maxagr, axis=-1, keepdims=True)
        if binary_strategy is not None:
            return self.tie_breaking(self.binarize(maxagr, strategy=binary_strategy))
        else:
            return maxagr
        
    @tf.function
    def get_local_mean(self, pred, scope):
        '''
        Perform convolution across the pixels of a prediction.
        :param pred:
        :param scope: filter size.
        :return:
        '''
        import tensorflow as tf 
        if scope % 2 == 0:
            print("Warning: Even filter size could result in asymmetrical convolution")
        pred = tf.cast(pred, tf.float32)
        filters = tf.fill((scope, scope, pred.shape[-1], 1), 1.0/(scope*scope))
        convs =  tf.nn.depthwise_conv2d(pred, filters, strides=[1,1,1,1], padding='SAME')
        return convs
    
    @tf.function
    def normalize_softmax(self, x):
        softmax = lambda x: tf.math.softmax(x, axis=-1)
        if len(x.shape) >= 4:
            return tf.map_fn(softmax, x)
        else:
            return tf.math.softmax(x, axis=-1)
    
    def is_normalized(self, predictions):
        return np.all(np.equal(np.around(np.sum(predictions, axis=-1), decimals=3), 1.0))
    
    def get_confidence_convolution(self, pred, scope, normalize):
        convs = self.get_local_mean(pred, scope)
        if not self.is_normalized(convs) and normalize:
            return self.normalize_softmax(convs).numpy()
        else:
            return convs.numpy()
    
    def entropy_per_pixels(self, proposal):
        '''
        Get the entropy over the labels of a proposal
        :param proposal - proposals of shape [H, W, labels] 
        '''
        n_labels = proposal.shape[-1]
        entr = lambda x, base=n_labels, eps=10e-16: -np.sum(x*np.log(x+eps)/np.log(base),axis=-1)
        entr_over_pixels = entr(proposal)
        return np.expand_dims(entr_over_pixels, axis=-1)

    def get_confidence(self, proposal, method, convolution_size=3):
        ''' 
        Get the confidence for the current proposal.
        :param proposal - An array of shape [H, W, labels]
        :param method - can be either 'pixelwise_entropy', 'mean_entropy' or 'convolution_entropy'.
        :param convolution_size [Default: 3] size of the convolution filter for 'convolution_entropy'.        
        '''
        assert method in ['pixelwise_entropy', 'mean_entropy', 'convolution_entropy']
        assert len(proposal.shape) == 3, "Proposal must have shape [H, W, Labels]"
        
        entropy = self.entropy_per_pixels(proposal)
        if method == 'pixelwise_entropy':
            return 1.0 - entropy
        if method == 'mean_entropy':
            return 1.0 - np.full_like(entropy, np.mean(entropy))
        if method == 'convolution_entropy':
            return 1.0 - self.get_confidence_convolution(entropy[np.newaxis, ...], convolution_size, normalize=False)[0]
        
    def weighted_average(self, proposals, weights, binary_strategy='maximum'):
        '''
        Computes an agreement based on the weighted average of the proposals. Applies Tie breaking if the binarization is active.
        :param proposals - An array of shape [Agents, H, W, Labels]
        :param weights - An array of shape [Agents, H, W]
        :param binary_strategy [Default: 'maximum'] binarization strategy. Applies Tie Breaking. If None, a float result is returned.
        '''
        
        agreement = np.divide(np.sum(proposals*weights, axis=0), np.sum(weights, axis=0))
        if binary_strategy is None:
            return agreement
        else:
            return self.tie_breaking(self.binarize(agreement, strategy=binary_strategy))

        
    def add_noise(self, proposal, std, mean=0.0, normalize=True):
        assert len(proposal.shape) == 3, "Please input proposals for a single agent, in format (h, w, labels)"
        noise = np.random.normal(loc=mean, scale=std, size=proposal.shape)
        proposal = np.clip(proposal + noise, 0.0, 1.0)
        if normalize:
            return proposal / proposal.sum(axis=-1, keepdims=True)
        else:
            return proposal
        
        
        
        

        
        
        
        
        
# Use these carefully, they account for the full volume in input and may be misleading
#     def masked_mae(self, x, y, mask=None, axis=None):
#         error = np.abs(x - y)
#         if mask is not None:
#             error = np.where(mask.astype(np.bool), error, np.nan)
#         return np.nanmean(error, axis=axis)

#     def dice_score(self, pred, true, mask=None, axis=None):
#         tp, tn, fp, fn = self.confusion_matrix(true, pred, mask=mask, axis=axis)
#         return 2. * tp / (2 * tp + fp + fn)




    
    
    
    
# OTER UTILITIES

def softmax(X, theta=1.0, axis=None):
    """
    from: https://nolanbconaway.github.io/blog/2017/softmax-numpy

    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


# Feature scaling for visualization
def feature_scaling(x, A, B):
    ''' Perform linear scaling of an array of data between x.min() and x.max() to a new range [A,B]'''
    return A + (B - A) * (x - x.min()) / (x.max() - x.min())


def numpy_to_pandas_series(data, index_prefix=None, index_values=None):
    '''Convert a multidimensional numpy array into a panda series having as many indices as the data dimensions and a single value column.
    Either index_prefix (a list of strings of length dim) or index_values (see MultiIndex.from_product()) should be defined. '''
    assert index_prefix or index_values
    assert not (index_prefix and index_values)
    if index_values:
        builder = index_values
    else:
        builder = [[pref + str(i) for i in range(dim)] for dim, pref in zip(data.shape, index_prefix)]

    assert len(data.shape) == len(builder), "Data is shape " + str(data.shape) + " but index builder is long " + str(
        len(builder))

    import pandas as pd
    indices = pd.MultiIndex.from_product(builder)
    return pd.Series(data=data.flatten(), index=indices)


def plot(proposals=None, input_sample=None, ground_truth=None, agreement=None, agreement_title='Agr', size=(10, 10),
         dpi=200, agent_names=None, label_names=None):  
    
    if input_sample is not None:
        sub = plt.figure(figsize=tuple((int(f / 4) for f in size)), dpi=dpi)
        plt.axis('off')
        plt.imshow(input_sample)
        plt.title("Input Sample")

    if proposals is not None:
        agent_names = ['Agent {}'.format(n) for n in range(proposals.shape[0])] if agent_names is None else agent_names
        label_names = ['Label {}'.format(n) for n in range(proposals.shape[-1])] if label_names is None else label_names
        plt.figure(figsize=size, dpi=dpi)
        for a, agent in enumerate(agent_names):
            for l, label in enumerate(label_names):
                sub = plt.subplot(len(agent_names), len(label_names), 1 + len(label_names) * a + l)
                plt.axis('off')
                plt.imshow(proposals[a, ..., l], cmap='Greys_r', vmin=0.0, vmax=1.0)
                sub.set_title("Proposal \n" + agent + ":" + label)

    if ground_truth is not None:
        label_names = ['Label {}'.format(n) for n in range(ground_truth.shape[-1])] if label_names is None else label_names
        fig = plt.figure(figsize=size, dpi=dpi)
        for l, label in enumerate(label_names):
            sub = plt.subplot(1, len(label_names), 1 + l)
            plt.axis('off')
            plt.imshow(ground_truth[..., l], cmap='Greys_r', vmin=0.0, vmax=1.0)
            sub.set_title("GT: " + label)

    if agreement is not None:
        label_names = ['Label {}'.format(n) for n in range(agreement.shape[-1])] if label_names is None else label_names
        plt.figure(figsize=size, dpi=dpi)
        for l, label in enumerate(label_names):
            sub = plt.subplot(1, len(label_names), 1 + l)
            plt.axis('off')
            plt.imshow(agreement[..., l], cmap='Greys_r', vmin=0.0, vmax=1.0)
            sub.set_title(str(agreement_title) + ": " + label)

