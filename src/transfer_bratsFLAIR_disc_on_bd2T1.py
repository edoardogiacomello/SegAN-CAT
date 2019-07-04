from DeepMRI import DeepMRI
import SegAN_IO_arch as arch

gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Transfer_BratsFLAIR_disc_Bc2T1")
gan.build_model(load_model='last', seed=1234567890, arch=arch)

for i, l in enumerate(gan.discriminator.layers):
    if i > 4:
        l.trainable = False
gan.load_dataset('bd2decide', ['T1'])
gan.train()
