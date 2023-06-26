import os


class Options:
    def __init__(self, checkpoints, gpu_ids):
        self.serial_batches = True  # no shuffle
        self.no_flip = True  # no flip
        self.label_nc = 0
        self.n_downsample_global = 3
        self.mc = 64
        self.k_size = 4
        self.start_r = 1
        self.mapping_n_block = 6
        self.map_mc = 512
        self.no_instance = True
        self.checkpoints_dir = checkpoints
        self.gpu_ids = gpu_ids
        self.mapping_net_dilation = 1
        self.use_segmentation_model = False
        self.feat_dim = -1
        self.spatio_size = 64
        self.resize_or_crop = 'scale_width'
        self.isTrain = False
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.norm = "instance"
        self.load_pretrain = ""
        self.which_epoch = "latest"
        self.load_pretrain = ""
        self.no_load_VAE = False
        self.use_vae_which_epoch = "latest"

        self.NL_res = True
        self.use_SN = True
        self.correlation_renormalize = True
        self.NL_use_mask = True
        self.NL_fusion_method = "combine"
        self.non_local = "Setting_42"
        # self.name = "mapping_scratch"
        self.load_pretrainA = os.path.join(self.checkpoints_dir, "VAE_A_quality")
        self.load_pretrainB = os.path.join(self.checkpoints_dir, "VAE_B_scratch")

        self.mapping_exp = 1
        self.inference_optimize = True
        self.mask_dilation = 3
        self.name = "mapping_Patch_Attention"
