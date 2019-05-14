

def test_filter_stacking():
    import tensorflow as tf
    import numpy as np
    from skimage.io import imshow
    import matplotlib.pyplot as plt
    H, W, R, C = 2, 3, 4, 5

    # This is a test for checking that the stacking of filters for visualization works as intended

    get_random_block = lambda: np.full((H, W), fill_value=np.random.randint(low=0, high=255))
    get_random_filter = lambda: np.stack([get_random_block() for r in range(R)], axis=-1)
    get_random_weights = lambda: np.stack([get_random_filter() for c in range(C)], axis=-1)
    weights = get_random_weights()

    print(weights)
    print(weights.shape)

    # weights = np.transpose(weights, [2,3,0,1])
    # weights = np.split(weights, axis=-1)
    solution = np.reshape(np.transpose(weights, [3, 1, 2, 0]), (W * C, H * R)).transpose()
    print(weights)
    with tf.Session() as sess:
        w = tf.constant(weights)
        w = tf.transpose(w, [3, 2, 0, 1])  # C R H W
        # INEFFICIENT SOLUTION
        # patches = [tf.unstack(c, axis=-1) for c in tf.unstack(w, axis=-1)]
        # rebuilt = tf.concat([tf.concat(col, axis=0) for col in patches], axis=1)
        w = tf.reshape(w, (C, W, H, R))

        rebuilt = tf.concat(w, axis=0)
        res = sess.run(rebuilt)
        pass
    print(res)
    print(res.shape)
    # imshow(res)
    # plt.show()

def test_dataset_performance():
    import dataset_helpers as dh
    import tensorflow as tf
    import time





    tf.set_random_seed(123456789)
    train_dataset = dh.load_dataset('brats2015-Train-all',
                                    #filter=lambda x: tf.logical_not(tf.equal(x['dataset_version'], "2013")),
                                    batch_size=32,
                                    prefetch_buffer=3,
                                    shuffle_buffer=64,
                                    shuffle=False,
                                    cache=False,
                                    mri_type="MR_T1",
                                    interleave=3,
                                    cast_to=tf.float32,
                                    clip_labels_to=1.0,
                                    infinite=False,
                                    n_threads=8)

    validation_dataset = dh.load_dataset('brats2015-Train-all',
                                         #filter=lambda x: tf.equal(x['dataset_version'], "2013"),
                                         batch_size=32,
                                         prefetch_buffer=6,
                                         shuffle_buffer=64,
                                         shuffle=False,
                                         cache=False,
                                         mri_type="MR_T1",
                                         cast_to=tf.float32,
                                         clip_labels_to=1.0,
                                         infinite=False,
                                         n_threads=8)

    tensorboard_datasets = dh.load_dataset('brats2015-Train-all',
                                           filter=lambda x: tf.equal(x['dataset_version'], "2013"),
                                           batch_size=32,
                                           prefetch_buffer=1,
                                           shuffle_buffer=1,
                                           mri_type="MR_T1",
                                           cast_to=tf.float32,
                                           clip_labels_to=1.0,
                                           take_only=32,
                                           shuffle=False,
                                           infinite=True,
                                           cache=False)

    # Define a "feedable" iterator of a string handle that selects which dataset to use
    use_dataset = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(use_dataset, train_dataset.output_types, train_dataset.output_shapes)
    next_batch = iterator.get_next()

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = validation_dataset.make_one_shot_iterator()
    tboard_iterator = tensorboard_datasets.make_one_shot_iterator()

    # Initializers for the datasets
    #reset_train_iter = train_iterator.initializer
    #reset_validation_iter = valid_iterator.initializer
    #reset_tboard_iter = tboard_iterator.initializer

    steps_to_profile = []
    profiler_builder = tf.profiler.ProfileOptionBuilder
    profiler_opts = profiler_builder(profiler_builder.time_and_memory()).order_by('micros').build()
    with tf.contrib.tfprof.ProfileContext("/home/DeepMRI/models/SegAN/profiled_test/",
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:

        with tf.Session() as sess:

            # Handles to switch between datasets
            use_train_dataset = sess.run(train_iterator.string_handle())
            use_valid_dataset = sess.run(valid_iterator.string_handle())
            use_tboard_dataset = sess.run(tboard_iterator.string_handle())

            global_step=1
            while global_step <= 1500:
                # Initialize the dataset iterators and set the training flag to True (for BatchNorm)
                last_time = time.time()
                # sess.run([reset_train_iter, reset_validation_iter, reset_tboard_iter])
                print("Training...")

                for i in range(100):
                    print("{}: INIT took {}".format(global_step, time.time() - last_time))
                    last_time = time.time()
                    if global_step in steps_to_profile:
                        pctx.trace_next_step()
                        pctx.dump_next_step()
                    # Training of C and S
                    _ = sess.run(next_batch, feed_dict={use_dataset: use_train_dataset})
                    print("{}: C_data took {}".format(global_step, time.time()-last_time))
                    last_time = time.time()
                    if global_step in steps_to_profile:
                        pctx.profiler.profile_operations(options=profiler_opts)
                        pctx.trace_next_step()
                        pctx.dump_next_step()

                    _ = sess.run(next_batch,feed_dict={use_dataset: use_train_dataset})
                    print("{}: S_data took {}".format(global_step, time.time() - last_time))
                    last_time = time.time()
                    # Visualizing losses
                    if global_step in steps_to_profile:
                        pctx.profiler.profile_operations(options=profiler_opts)
                        pctx.trace_next_step()
                        pctx.dump_next_step()
                    _ = sess.run(next_batch, feed_dict={use_dataset: use_train_dataset})
                    print("{}: C_loss took {}".format(global_step, time.time() - last_time))
                    last_time = time.time()
                    if global_step in steps_to_profile:
                        pctx.profiler.profile_operations(options=profiler_opts)
                        pctx.trace_next_step()
                        pctx.dump_next_step()
                    _ = sess.run(next_batch, feed_dict={use_dataset: use_valid_dataset})
                    print("{}: S_loss took {}".format(global_step, time.time() - last_time))
                    last_time = time.time()
                    global_step += 1

                print("Checkpoint...")
                # Show a prediction made with a reference sample:
                # Show the differences by using trained and current BatchNorm weights.
                _ = sess.run(next_batch, feed_dict={use_dataset: use_tboard_dataset})
                print("{}: V_train took {}".format(global_step, time.time() - last_time))
                last_time = time.time()
                _ = sess.run(next_batch, feed_dict={use_dataset: use_tboard_dataset})
                print("{}: V_test took {}".format(global_step, time.time() - last_time))
                last_time = time.time()

if __name__ == '__main__':
    test_dataset_performance()