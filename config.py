
PRETRAINING = 0
FINE_TUNING = 1


class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i" % mode

        self.mode = mode

        if self.mode == PRETRAINING:
            self.batch_size = 6
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 0
            self.nb_epochs = 10
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 1.  # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"
            self.model = "UNet"

            # Paths to the data
            self.data_train = "D:\mri\dataset_121_121_145"
            self.label_train = None

            self.data_val = None
            self.label_val = None

            self.input_size = (1, 121, 121, 145)
            self.label_name = "age"

            self.num_classes = 2

            self.checkpoint_dir = "./workspace"

        elif self.mode == FINE_TUNING:
            # We assume a classification task here
            self.batch_size = 16
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 10
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            self.tf = None

            # Paths to the data
            self.data_train = "D:\mri\dataset_121_121_145"
            self.label_train = None

            self.data_val = None
            self.label_val = None

            self.input_size = (1, 121, 121, 145)
            self.label_name = "age"

            self.pretrained_path = "workspace/y-Aware_Contrastive_MRI_epoch_7.pth"
            self.num_classes = 2
            self.model = "UNet"
