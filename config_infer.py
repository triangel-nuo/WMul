##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 4                 # number of Dataloader workers
epochs = 160                # number of epochs
batch_size = 4            # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
net = 'inceptionR'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

visual_path = None  

##################################################
# Dataset/Path Config
##################################################
tag = 'car'                # 'aircraft', car'

# checkpoint model for resume training
ckpt = './FGVC/car/test/model_bestacc.pth'
