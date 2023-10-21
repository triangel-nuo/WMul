##################################################
# Training Config
##################################################
workers = 1                 # number of Dataloader workers
epochs = 160                # number of epochs
batch_size = 4            # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'inceptionR'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'aircraft'                # 'aircraft',  'car'

# saving directory of .ckpt models
save_dir = './FGVC/aircraft/inceptionR/test'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name
