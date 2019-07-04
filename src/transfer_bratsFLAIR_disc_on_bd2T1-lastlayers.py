from DeepMRI import DeepMRI
import SegAN_IO_arch as arch

gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Transfer_BratsFLAIR_disc_Bc2T1_lastlayers")
gan.build_model(load_model='last', seed=1234567890, arch=arch)

for l in gan.discriminator.layers[-7:]:
    l.trainable = False
gan.load_dataset('bd2decide', ['T1'])
gan.train()
