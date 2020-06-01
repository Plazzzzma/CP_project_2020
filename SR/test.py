import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#from model import RCAN, SRFBN, RRDB, SAN, RDN, EDSR
import skimage
import skimage.measure

from model.RRDB import RRDBNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--epoch", type=int, default=3, help="Number of aining epochs")
parser.add_argument("--net", type=str, default="ours", help="RCAN, ESRGAN, SAN, RDN, EPSR, SRFAN, ours")
parser.add_argument("--denoiser", type=str, default="None", help="previous denoisor")
parser.add_argument("--avg", type=str, default="avg1", help="if no previous denoisor, which noise level folder to use")
parser.add_argument("--test_path", type=str, default='../../data/test/test_images/', help="Directory where test images are stored")
parser.add_argument("--gt_path", type=str, default='../../data/test/test_gt/', help="Directory where GT images are stored")
opt = parser.parse_args()


def inference(model, test_path, gt_path, results_dir):
    files_input = glob.glob(test_path+'/*.png')
    files_input.sort()
    files_gt = glob.glob((gt_path+'/*.png'))
    files_gt.sort()
    
    print("files_gt: ", files_gt)
    print("files_input: ", files_input)
    
    mse_results=np.zeros(len(files_input))
    psnr_results = np.zeros(len(files_input))
    ssim_results = np.zeros(len(files_input))
    
    print("Loop over images")
    for idx in range(len(files_input)):
        lr_img = cv2.imread(files_input[idx], cv2.IMREAD_GRAYSCALE)
        (w,h) = np.shape(lr_img)
        img_ans = np.zeros((w,h))
        x = 0
        y = 0
        #print("Image: " + str(idx))
        while (x < w):
            #print("x: "+str(x)+" w: "+str(w))
            y = 0
            while (y < h):
                #print("y: "+str(y)+" h: "+str(h))
                tmp_patch = lr_img[x:x+128,y:y+128] / 255
                tmp_patch = np.expand_dims(tmp_patch, 0)
                tmp_patch = torch.tensor([tmp_patch]).float()
                #print("before model")
                sr_patch = model(tmp_patch.cuda())
                #print("after model")
                if isinstance(sr_patch, list):
                    sr_patch = sr_patch[-1]
                sr_patch = sr_patch.cpu().data.numpy()
                sr_patch = np.clip(sr_patch, 0, 1)
                sr_patch = sr_patch[0][0]
                
                #print("sr_patch shape: ", sr_patch.shape)
                #print("img_ans shape: ",img_ans.shape)
                #print("tmp_patch shape: ", tmp_patch.shape)

                #img_ans[x+64:x+128,y+64:y+128] = sr_patch[64:,64:]
                
                #print("img_ans shape: ", img_ans.shape)
                
                #print("x: ", x)
                #print("y: ", y)
                
                #if (x < 64):
                #    img_ans[x+64:x+128,y+64:y+128] = sr_patch[:64,64:]
                #if (y < 64):
                #    img_ans[x+64:x+128,y:y+64] = sr_patch[64:,:64]
                #if (x < 64 and y < 64):
                #    img_ans[x:x+64,y:y+64] = sr_patch[:64,:64]
                #y = y + 64
            #x = x + 64
                img_ans[x+32:x+128,y+32:y+128] = sr_patch[32:,32:]
            
                if (x < 32):
                    img_ans[x:x+32,y+32:y+128] = sr_patch[:32,32:]
                if (y < 32):
                    img_ans[x+32:x+128,y:y+32] = sr_patch[32:,:32]
                if (x < 32 and y < 32):
                    img_ans[x:x+32,y:y+32] = sr_patch[:32,:32]
                y = y + 64
            x = x + 64
        
        print("loop done")
        img_gt = cv2.imread(files_gt[idx], cv2.IMREAD_GRAYSCALE)
        print(skimage.measure.compare_psnr(img_ans, img_gt/255))
        mse_results[idx]=skimage.measure.compare_mse(img_ans,img_gt/255)
        psnr_results[idx] = skimage.measure.compare_psnr(img_ans, img_gt/255)
        ssim_results[idx] = skimage.measure.compare_ssim(img_ans, img_gt/255)

        cv2.imwrite(results_dir + '/' + os.path.basename(files_input[idx]), (255*img_ans).astype('uint8'))
            
    return psnr_results, ssim_results, mse_results


def main():
    
    model_name = opt.net
    model_dir = '../../net_data/trained_srs/ours/' #os.path.join('../../net_data/trained_srs/', opt.net)
    
    if opt.net == 'RCAN':
        net = RCAN()
    elif opt.net == 'SRFBN':
        net = SRFBN()
    elif opt.net == 'ESRGAN':
        net = RRDBNet(nb=23)
    elif (opt.net == 'EPSR'):
        net = EDSR()
    elif (opt.net == 'RDN'):
        net = RDN()
    elif (opt.net == 'SAN'):
        net = SAN()
    elif (opt.net == 'ours'):
        net = RRDBNet()
    else:
        raise NotImplemented('Network model not implemented.')

    #model = nn.DataParallel(net).cuda()
    model = net
    model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch_%d.pth' % (opt.epoch)), map_location='cuda:0')['model'].state_dict())
    #model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch_19.pth'),map_location='cuda:0')['model'].state_dict())#, strict=False
    model.to(torch.device("cuda:0"))
    model.eval()

    if opt.denoiser == 'None':
        test_dir = opt.test_path #modified 06.05 before was opt.path.join(testpath+avg)
        result_dir = os.path.join('../../results/', opt.net)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:    
        test_dir = os.path.join(opt.test_path, opt.denoiser)
        result_dir = os.path.join(opt.test_path, opt.net, opt.denoiser)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    print('Testing with model %s at epoch %d, with %s' %(model_name, opt.epoch, test_dir))

    if opt.denoiser == 'None':
        psnr_results, ssim_results,mse_results = inference(model, test_dir, opt.gt_path, result_dir)
        np.save(os.path.join(result_dir, 'PSNR'), psnr_results)
        np.save(os.path.join(result_dir, 'SSIM'), ssim_results)
        return
    for avg_img_dirs in glob.glob('%s/*'%(test_dir)):
        test_dir = avg_img_dirs
        results_dir = os.path.join(result_dir, os.path.basename(test_dir))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        psnr_results, ssim_results = inference(model, test_dir, opt.gt_path, results_dir)
        print( 'Average %s PSNR: %.2fdB' %(opt.test_path, np.mean(psnr_results)) ) 
        print( 'Average %s MSE: %.2fdB' %(opt.test_path, np.mean(mse_results)) ) 
        np.save( os.path.join(results_dir, 'PSNR'), psnr_results )
        np.save( os.path.join(results_dir, 'SSIM'), ssim_results )
        np.save(os.path.join(results_dir,'MSE'), mse_results)

if __name__ == "__main__":
    main()
