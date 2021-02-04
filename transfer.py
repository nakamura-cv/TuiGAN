from options.config import get_arguments
from utils.manipulate import *
from models.TuiGAN import *
import utils.functions as functions
from utils.imresize import imresize
from utils.imresize import imresize_to_shape


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_name', help='load model name', required=True) # moving-gif_00058_paint
    parser.add_argument('--model_param', help= 'load model param', default='scale_factor=0.750,min_size=100,niter=4000,lambda_cyc=1.000,lambda_idt=1.000')
    parser.add_argument('--input_dir', help='input image dir', required=True) 
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='transfer')
    parser.add_argument('--start_scale', help='injection scale', type=int, default='0')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    Gs2 = []
    dir2save = functions.generate_dir2save(opt)

    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real_in = functions.read_image(opt)
        functions.adjust_scales2image(real_in, opt)
    
        real_ = functions.read_image(opt)
        real = imresize(real_,opt.scale1,opt)
        reals = functions.creat_reals_pyramid(real,reals,opt)
        Gs,Zs,NoiseAmp,Gs2= functions.load_model(opt)
        TuiGAN_transfer(Gs,Zs,reals,NoiseAmp,Gs2,opt)