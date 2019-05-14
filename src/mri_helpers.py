import SimpleITK as sitk
import itertools
from ipywidgets import interact
import matplotlib.pyplot as plt

def myshow(img, title=None, margin=0.05, dpi=200 ):
    '''
    Function for showing a MRI in jupyter
    '''
    import matplotlib.pyplot as plt
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if (slicer):
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]


    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    def callback(z=None):
        import matplotlib.pyplot as plt
        extent = (0, xsize*spacing[1], ysize*spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

        plt.set_cmap("gray")

        if z is None:
            ax.imshow(nda,extent=extent,interpolation=None)
        else:
            ax.imshow(nda[z,...],extent=extent,interpolation=None)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0,nda.shape[0]-1))
    else:
        callback()

def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    import matplotlib.pyplot as plt
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))


    img_null = sitk.Image([0,0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        #TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            img = sitk.Compose(img_comps)

def show_mri(file_path):
    myshow(sitk.ReadImage(file_path))
    
def resample_to(mha_in, mha_ref, is_label):
    ''' Resample an image from an input space (in_size, in_spacing, in_origin) to a reference space.'''
    # Adapted from https://gist.github.com/zivy/79d7ee0490faee1156c1277a78e4a4c4
    b2b = sitk.ReadImage(mha_in)
    ref = sitk.ReadImage(mha_ref)
    s_size, s_spacing, s_origin = b2b.GetSize(), b2b.GetSpacing(), b2b.GetOrigin()
    t_size, t_spacing, t_origin = ref.GetSize(), ref.GetSpacing(), ref.GetOrigin()

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
    # spacing will not yield the correct coordinates resulting in a long debugging session. 
    t_center = np.array(ref.TransformContinuousIndexToPhysicalPoint(np.array(ref.GetSize())/2.0))

    dimension = b2b.GetDimension()

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(b2b.GetDirection())
    transform.SetTranslation(np.array(b2b.GetOrigin()) - t_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(b2b.TransformContinuousIndexToPhysicalPoint(np.array(b2b.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - t_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth 
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that 
    # no new labels are introduced.
    interpolator = sitk.sitkLinear if not is_label else sitk.sitkNearestNeighbor
    return sitk.Resample(b2b, ref, centered_transform, interpolator, 0.0)
    