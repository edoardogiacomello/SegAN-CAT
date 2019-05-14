class DummyDataLoader():
    def __init__(self, shape_mri, shape_gt, maxsize = 300):
        import queue
        from multiprocessing.dummy import Pool
        import os
        self.shape_mri = shape_mri
        self.shape_gt = shape_gt
        self.Q = queue.Queue(maxsize=maxsize)
        # Start generating objects
        self.stop_signal = False
        # Starting 32 threads to load data
        self.pool = Pool(os.cpu_count())
        self.pool.apply_async(self.worker)
        print("DataLoader has started loading objects")

    def next(self):
        if self.Q.empty():
            print("Dataset buffer is empty and it's impacting on performance.")
        return self.Q.get(block=True)


    def worker(self):
        while not self.stop_signal:
            mri, gt = self.generate()
            self.Q.put((mri, gt), block=True)

    def stop(self):
        self.stop_signal = True
        self.pool.close()
        self.pool.join()

    def generate(self):
        batch_mri = list()
        batch_gt = list()
        for i in range(self.shape_mri[0]):
            # As dummy data we generate some squares and triangles:
            # MRI has both, while gt is the segmentation done on triangles only
            rectangles, _ = draw.random_shapes(self.shape_mri[1:], max_shapes=5, multichannel=False, shape='rectangle')
            rectangles = util.invert(rectangles)
            triangles, _ = draw.random_shapes(self.shape_mri[1:], max_shapes=5, multichannel=False, shape='triangle')
            triangles = util.invert(triangles)
            # Impose triangles over rectangles
            mri = np.where(triangles>0, triangles, rectangles)
            binary_triangles = np.where(triangles>0, 1.0, 0.0)
            batch_mri.append(util.img_as_float32(mri))
            batch_gt.append(util.img_as_float32(binary_triangles))
        batch_mri, batch_gt = np.stack(batch_mri, axis=0), np.stack(batch_gt, axis=0)
        if self.shape_mri[-1] == 1:
            batch_mri = np.expand_dims(batch_mri, axis=-1)
            batch_gt = np.expand_dims(batch_gt, axis=-1)
        return batch_mri, batch_gt