import wandb
import numpy as np
import os
from torch import autocast
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
from kornia.geometry.transform import translate

from datasets.audioset import my_get_test_set, get_full_training_set, get_ft_weighted_sampler
from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from diffeo.transform import Diffeo, temp_from_displacement
from botorch.utils.sampling import sample_hypersphere


preds_url = \
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/passt_enemble_logits_mAP_495.npy"


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x
'''
def sample_sphere(input_tensor, noise_norm):
    # Naive rejection sampling, suuper slow
    rand_direction = torch.torch.rand_like(input_tensor)
    norm = torch.linalg.norm(rand_direction, dim=(i for i in range(1, len(input_tensor.shape))))
    rejected = norm>1
    while rejected.any():
        rand_direction[rejected] = torch.torch.rand_like(input_tensor[rejected])
        norm = torch.linalg.norm(rand_direction[rejected], dim=(i for i in range(1, len(input_tensor.shape))))
        rejected = norm>1
    return rand_direction*noise_norm
'''

def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.binary_cross_entropy_with_logits(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)
    return mAP.mean(), ROC.mean(), losses.mean()


def evaluate(args):
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         fmin=args.fmin,
                         fmax=args.fmax
                        )
    mel.to(device)
    mel.eval()
    ###############
    # DIFFEO
    ###############
    diffeos = {}
    outputs = {}
    norm_diff = {}
    noise_norm = {}
    dims = [128, 1000]
    axis  = args.axis
    for cutoff in args.cutoffs:
        for d in args.displacements:
            temp = temp_from_displacement(d, cutoff, dims[axis])
            diffeos[f"{cutoff}_{d}_{axis}"] = Diffeo(cutoff, temp, axis=[axis,])
            outputs[f"{cutoff}_{d}_{axis}"] = []
            outputs[f"Noise_{cutoff}_{d}_{axis}"] = []
            norm_diff[f"{cutoff}_{d}_{axis}"] = []
    for d in args.displacements:
        outputs[f"noise_{d}"] = []
        outputs[f"translation_{d}_axis_{axis}"] = []
    outputs[f"clean_repeated"] = []
    outputs[f"clean"] = []
    ##########

    dl = DataLoader(dataset= my_get_test_set("/home/chiche/datasets/audioset-kaggle/", resample_rate=args.resample_rate), #get_test_set(resample_rate=args.resample_rate)
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=False)

    print(f"Running AudioSet evaluation for model '{model_name}' on device '{device}'")
    targets = []
    axis_names = ["Time", "Freq"]
    x_diff = []
    for batch in tqdm(dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        with autocast(device_type=device.type) if args.cuda else nullcontext():
            with torch.no_grad():
                #print("x", x.shape)
                x = _mel_forward(x, mel)
                #print(x.shape)
                y_clean, _ = model(x)
                outputs["clean"].append(y_clean.float().cpu().numpy())
                assert y_clean.dtype is torch.float16
                targets.append(y.cpu().numpy())
                for _ in range(args.num_diffeos):
                    outputs["clean_repeated"].append(y_clean.float().cpu().numpy())
                    for cutoff in args.cutoffs:
                        for d in args.displacements:
                            diffeo_x = diffeos[f"{cutoff}_{d}_{axis}"](x)
                            y_hat, _ = model(diffeo_x)
                            assert y_hat.dtype is torch.float16
                            #print("y_diffeo", y_hat.shape)
                            norm_diff[f"{cutoff}_{d}_{axis}"].append(torch.linalg.norm((diffeo_x-x).reshape(x.shape[0], -1), axis=1).cpu().numpy())
                            #print("norm diff", norm_diff[f"{cutoff}_{d}_{axis}"][-1].shape)
                            outputs[f"{cutoff}_{d}_{axis}"].append(y_hat.float().cpu().numpy())
                            #print("outputs", outputs[f"{cutoff}_{d}_{axis}"][-1].shape)
                    for d in args.displacements:
                        y_hat, _ = model(x+d*torch.rand_like(x))
                        outputs[f"noise_{d}"].append(y_hat.float().cpu().numpy())
                        t = torch.zeros((y_hat.shape[0], 2), device=y_hat.device)
                        t[axis] = 2*(torch.rand_like(t[0])>0.5)*d-d
                        y_hat, _ = model(translate(x, t))
                        outputs[f"translation_{d}_axis_{axis}"].append(y_hat.float().cpu().numpy())
        #break
    #print("clean bf", outputs["clean"][0].shape)
    #print("targets bf", targets[0].shape)
    clean_outputs =  np.concatenate(outputs["clean_repeated"])        
    ##########
    #   NOISE
    ##########
    for cutoff in args.cutoffs:
        for d in args.displacements:
            noise_norm[f"{cutoff}_{d}_{axis}"] = np.median(np.concatenate(norm_diff[f"{cutoff}_{d}_{axis}"]))
            #print("noise norm", noise_norm[f"{cutoff}_{d}_{axis}"])
    for batch in tqdm(dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with autocast(device_type=device.type) if args.cuda else nullcontext():
            with torch.no_grad():
                x = _mel_forward(x, mel)  
                for _ in range(args.num_diffeos): 
                    for cutoff in args.cutoffs:
                        for d in args.displacements:
                            with torch.no_grad():
                                x_noisy = x + noise_norm[f"{cutoff}_{d}_{axis}"]*sample_hypersphere(x.shape[2]*x.shape[3], x.shape[0], device=x.device, dtype = x.dtype).reshape(x.shape)
                                y_noisy, _ = model(x_noisy)
                                outputs[f"Noise_{cutoff}_{d}_{axis}"].append(y_noisy.cpu().numpy())
        #break
    for cutoff in args.cutoffs:
        for d in args.displacements:
            output = np.concatenate(outputs[f"Noise_{cutoff}_{d}_{axis}"])
            #print(f"Noise_{cutoff}_{d}_{axis}", output.shape)
            mse_noise = np.median(np.sqrt(np.sum((clean_outputs-output)**2, axis=1) ) )
            #print(f"MSE_Noise_{cutoff}_{d}_{axis}", mse_noise.shape)
            postfix = f"_cutoff_{cutoff}_disp_{d}_{axis_names[axis]}"
            wandb.log({f"MSE_Noise_"+postfix: mse_noise})
            output = np.concatenate(outputs[f"{cutoff}_{d}_{axis}"])
            #print(f"Diffeo_{cutoff}_{d}_{axis}", output.shape)
            mse = np.median(np.sqrt(np.sum((clean_outputs-output)**2, axis=1) ) )
            #print(f"MSE Diffeo_{cutoff}_{d}_{axis}", mse.shape)
            wandb.log({"MSE_Diffeo_"+postfix: mse})
            wandb.log({"Norm_MSE_Diffeo_"+postfix: mse/mse_noise})

    for d in args.displacements:
        output = np.concatenate(outputs[f"translation_{d}_axis_{axis}"])
        mse_trans = np.median(np.abs(clean_outputs-output))
        wandb.log({f"MSE_Trans_disp_{d}_{axis_names[axis]}": mse_trans})


    # Clean metrics
    targets = np.concatenate(targets)
    clean_outputs =  np.concatenate(outputs["clean"]) 
    #print("*"*20)
    #print(targets)
    #print(len(targets))
    #print("clean concat", clean_outputs.shape)
    #print("targets concat", targets.shape)
    mAP = metrics.average_precision_score(targets, clean_outputs, average=None).mean()
    ROC = metrics.roc_auc_score(targets, clean_outputs, average=None).mean()
    wandb.log({"mAP": mAP, "ROC": ROC})

    #print(f"Results on AudioSet test split for loaded model: {model_name}")
    print("  mAP: {:.3f}".format(mAP.mean()))
    print("  ROC: {:.3f}".format(ROC.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=12)

    # evaluation
    # overwrite 'pretrained_name' by 'ensemble' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])
    parser.add_argument('--model_name', type=str, default="mn10_as")

    # training
    parser.add_argument('--pretrained_name', type=str, default=None)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--epoch_len', type=int, default=100000)
    parser.add_argument('--roll', action='store_true', default=False)
    parser.add_argument('--wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=0)

    # lr schedule
    parser.add_argument('--max_lr', type=float, default=0.0008)
    parser.add_argument('--warm_up_len', type=int, default=8)
    parser.add_argument('--ramp_down_start', type=int, default=80)
    parser.add_argument('--ramp_down_len', type=int, default=95)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # knowledge distillation
    parser.add_argument('--teacher_preds', type=str,
                        default=os.path.join("resources", "passt_enemble_logits_mAP_495.npy"))
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--kd_lambda', type=float, default=0.1)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)

    # Diffeo
    parser.add_argument('--cutoffs', nargs=1, default=[20, 2], type=int)
    parser.add_argument('--displacements', nargs=1, default=[1, 2], type=float)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--num_diffeos', type=int, default=10)

    args = parser.parse_args()

    wandb.init(project="Audio Diffeo", entity="alelab", config=args, name=args.experiment_name)
    if args.train:
        train(args)
    else:
        evaluate(args)
