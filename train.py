from __future__ import division
from __future__ import print_function

import argparse
import getpass

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import torchvision
from torch.utils.tensorboard import SummaryWriter

import os, sys, json
import numpy as np
from skimage import io, transform
from datetime import datetime

from utils.utils import *
from utils.mpi_utils import *
from utils.mlp import *
from utils.unet_model import *
from utils.colmap_runner import colmapGenPoses

from einops import rearrange, reduce, repeat
# for ddp
import torch.distributed as dist
import torch.multiprocessing as mp
from models.networks.generator import SPADEGenerator
from timeit import default_timer as timer
import time

parser = argparse.ArgumentParser()

#training schedule
parser.add_argument('-epochs', type=int, default=4000, help='total epochs to train')
parser.add_argument('-steps', type=int, default=-1, help='total steps to train. In our paper, we proposed to use epoch instead.')
parser.add_argument('-tb_saveimage', type=int, default=50, help='write an output image to tensorboard for every <tb_saveimage> epochs')
parser.add_argument('-tb_savempi', type=int, default=200, help='generate MPI (WebGL) and measure PSNR/SSIM of validation image for every <tb_savempi> epochs')
parser.add_argument('-checkpoint', type=int, default=50, help='save checkpoint for every <checkpoint> epochs. Be aware that! It will replace the previous checkpoint.')
parser.add_argument('-tb_toc',type=int, default=500, help="print output to terminal for every tb_toc epochs")

#lr schedule
parser.add_argument('-lrc', type=float, default=10, help='the number of times of lr using for learning rate of explicit basis (k0).')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of a multi-layer perceptron')
parser.add_argument('-decay_epoch', type=int, default=1333, help='the number of epochs for decay learning rate')
parser.add_argument('-decay_rate', type=float, default=0.1, help='ratio of decay rate at every <decay_epoch> epochs')

#network (First MLP)
parser.add_argument('-ray', type=int, default=8000, help='the number of sampled ray that is used to train in each step')
parser.add_argument('-hidden', type=int, default=384, help='the number of hidden node of the main MLP')
parser.add_argument('-mlp', type=int, default=4, help='the number of hidden layer of the main MLP')
parser.add_argument('-pos_level', type=int, default=10, help='the number of positional encoding in terms of image size. We recommend to set 2^(pos_level) > image_height and image_width')
parser.add_argument('-depth_level', type=int, default=8,help='the number of positional encoding in terms number of plane. We recommend to set 2^(depth_level) > layers * subplayers')
parser.add_argument('-lrelu_slope', type=float, default=0.01, help='slope of leaky relu')
parser.add_argument('-sigmoid_offset', type=float, default=5, help='sigmoid offset that is applied to alpha before sigmoid')

#basis (Second MLP)
parser.add_argument('-basis_hidden', type=int, default=64, help='the number of hidden node in the learned basis MLP')
parser.add_argument('-basis_mlp', type=int, default=1, help='the number of hidden layer in the learned basis MLP')
parser.add_argument('-basis_order', type=int, default=3, help='the number of  positional encoding in terms of viewing angle')
parser.add_argument('-basis_out', type=int, default=8, help='the number of coeffcient output (N in equation 3 under seftion 3.1)')

#loss
parser.add_argument('-gradloss', type=float, default=0.05, help='hyperparameter for grad loss')
parser.add_argument('-perceptualloss', type=float, default=0, help='hyperparameter for perceptual loss')
parser.add_argument('-sparsityloss', type=float, default=0, help='hyperparameter for sparsity loss')
parser.add_argument('-tvc', type=float, default=0.03, help='hyperparameter for total variation regularizer')

#training and eval data
parser.add_argument('-scene', type=str, default="", help='directory to the scene')
parser.add_argument('-ref_img', type=str, default="",  help='reference image, camera parameter of reference image is use to create MPI')
parser.add_argument('-runpath', type=str, default="", help='directory save the run data')
parser.add_argument('-dmin', type=float, default=-1, help='first plane depth')
parser.add_argument('-dmax', type=float, default=-1, help='last plane depth')
parser.add_argument('-mask_weight', type=float, default=0, help='whether to add more weights for foreground')
parser.add_argument('-invz', action='store_true', help='place MPI with inverse depth')
parser.add_argument('-scale', type=float, default=-1, help='scale the MPI size')
parser.add_argument('-llff_width', type=int, default=1008, help='if input dataset is LLFF it will resize the image to <llff_width>')
parser.add_argument('-deepview_width', type=int, default=800, help='if input dataset is deepview dataset, it will resize the image to <deepview_width>')
parser.add_argument('-train_ratio', type=float, default=0.875, help='ratio to split number of train/test (in case dataset doesn\'t specify how to split)')
parser.add_argument('-random_split', action='store_true', help='random split the train/test set. (in case dataset doesn\'t specify how to split)')
parser.add_argument('-num_workers', type=int, default=8, help='number of pytorch\'s dataloader worker')
parser.add_argument('-cv2resize', action='store_true', help='apply cv2.resize instead of skimage.transform.resize to match the score in our paper (see note in github readme for more detail) ')
parser.add_argument('-use_appearance', action='store_true', help='whether to apply appearance code to data')
parser.add_argument('-use_learnable_planes', action='store_true', help='whether use learnable planes')
parser.add_argument('-use_learnable_pose', action='store_true', help='whether use learnable poses')

#MPI
parser.add_argument('-offset', type=int, default=200, help='the offset (padding) of the MPI.')
parser.add_argument('-layers', type=int, default=16, help='the number of plane that stores base color')
parser.add_argument('-sublayers', type=int, default=12, help='the number of plane that share the same texture. (please refer to coefficient sharing under section 3.4 in the paper)')

#predict
parser.add_argument('-no_eval', action='store_true', help='do not measurement the score (PSNR/SSIM/LPIPS) ')
parser.add_argument('-no_csv', action='store_true', help="do not write CSV on evaluation")
parser.add_argument('-no_video', action='store_true', help="do not write the video on prediction")
parser.add_argument('-no_webgl', action='store_true', help='do not predict webgl (realtime demo) related content.')
parser.add_argument('-predict', action='store_true', help='predict validation images')
parser.add_argument('-eval_path', type=str, default='runs/evaluation/', help='path to save validation image')
parser.add_argument('-web_path', type=str, default='runs/html/', help='path to output real time demo')
parser.add_argument('-web_width', type=int, default=16000, help='max texture size (pixel) of realtime demo. WebGL on Highend PC is support up to 16384px, while mobile phone support only 4096px')
parser.add_argument('-http', action='store_true', help='serve real-time demo on http server')
parser.add_argument('-render_viewing', action='store_true', help='genereate view-dependent-effect video')
parser.add_argument('-render_nearest', action='store_true', help='genereate nearest input video')
parser.add_argument('-render_depth', action='store_true', help='generate depth')
parser.add_argument('-render_fixed_view', action='store_true', help='generate fixed view video')
parser.add_argument('-render_keypoints_path', type=str, default='keypoints_2d', help='path to render keypoints (choose other keypoints to test)')
parser.add_argument('-render_bullet_time', action='store_true', help='render move with bullet time')

# render path
parser.add_argument('-nice_llff', action='store_true', help="generate video that its rendering path matches real-forward facing dataset")
parser.add_argument('-nice_shiny', action='store_true', help="generate video that its rendering path matches real-forward facing dataset")


#training utility
parser.add_argument('-model_dir', type=str, default="scene", help='model (scene) directory which store in runs/<model_dir>/')
parser.add_argument('-pretrained', type=str, default="", help='location of checkpoint file, if not provide will use model_dir instead')
parser.add_argument('-restart', action='store_true', help='delete old weight and retrain')
parser.add_argument('-clean', action='store_true', help='delete old weight without start training process')

# for generator  
parser.add_argument('-netG', type=str, default='unet', help='selects model to use for netG (pix2pixhd | spade)')
parser.add_argument('-ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('-init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('-init_variance', type=float, default=0.02, help='variance of the initialization distribution')
parser.add_argument('-use_vae', action='store_true', help='enable training with an image encoder.')
parser.add_argument('-norm_G', type=str, default='spectralspadesyncbatch3x3', help='norm G')
parser.add_argument('-num_upsampling_layers', type=str, default='normal', help='whether to add more upsampling layers')
parser.add_argument('-semantic_nc', type=int, default=5, help='# of input channels')
parser.add_argument('-crop_size', type=int, default=0, help='# of input channels')
parser.add_argument('-aspect_ratio', type=int, default=1, help='# of input channels')

parser.add_argument('-use_refinement', action='store_true', help='whether to add refinement network')
parser.add_argument('-use_gaussian', action='store_true', help='whether to use gaussian feature as input')


#miscellaneous
parser.add_argument('-gpus', type=int, default=1, help='number of gpu used')

args = parser.parse_args()

def computeHomographies(sfm, feature, planes, pose_deform = None):
  fx = feature['fx'][0]
  fy = feature['fy'][0]
  px = feature['px'][0]
  py = feature['py'][0]

  if type(pose_deform) == type(None):
    new_r = feature['r'][0] @ sfm.ref_rT.cuda()
    new_t = (-new_r @ sfm.ref_t.cuda()) + feature['t'][0]
  else:
    new_r = (feature['r'][0] + pose_deform[feature['idx'][0], :, :3]) @ sfm.ref_rT.cuda()
    new_t = (-new_r @ sfm.ref_t.cuda()) + feature['t'][0] + pose_deform[feature['idx'][0], :, 3:] 

  n = pt.tensor([[0.0, 0.0, 1.0]]).cuda()
  Ha = new_r.t()
  Hb = Ha @ new_t @ n @ Ha
  Hc = (n @ Ha @ new_t)[0]

  ki = pt.tensor([[fx, 0, px],
                  [0, fy, py],
                  [0, 0, 1]], dtype=pt.float).inverse().cuda()

  tt = sfm.ref_cam
  ref_k = pt.tensor( [[tt['fx'], 0, tt['px']],
                      [0, tt['fy'], tt['py']],
                      [0,        0,       1]]).cuda()

  # planes_mat = pt.Tensor(planes).view(-1, 1, 1).cuda()
  planes_mat = planes.view(-1, 1, 1)
  return (ref_k @ (Ha + Hb / (-planes_mat - Hc))) @ ki

def computeHomoWarp(sfm, input_shape, input_offset,
                    output_shape, selection,
                    feature, planes, inv=False, inv_offset = False, pose_deform = None):

  selection = selection.cuda()
  # coords: (sel, 3)
  coords = pt.stack([selection % output_shape[1], selection // output_shape[1],
                    pt.ones_like(selection)], -1).float()

  # Hs: (n, 3, 3)

  Hs = computeHomographies(sfm, feature, planes, pose_deform = pose_deform)
  if inv: Hs = Hs.inverse()
  if inv_offset:
    coords[:, :2] += input_offset
  prod = coords @ pt.transpose(Hs, 1, 2).cuda()
  scale = pt.tensor([input_shape[1] - 1, input_shape[0] - 1]).cuda()

  ref_coords = prod[:, :, :2] / prod[:, :, 2:]
  if not inv_offset:
    warp = ((ref_coords + input_offset) / scale.view(1, 1, 2)) * 2 - 1
  else:
    warp = ((ref_coords) / scale.view(1, 1, 2)) * 2 - 1
  warp = warp[:, :, None]


  return warp, ref_coords

def totalVariation(images):

  pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
  pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
  sum_axis = [1, 2, 3]

  tot_var = (
      pt.sum(pt.abs(pixel_dif1), dim=sum_axis) +
      pt.sum(pt.abs(pixel_dif2), dim=sum_axis))

  return tot_var / (images.shape[2]-1) / (images.shape[3]-1)

def get_sparsity_loss(mpi_a):
  q_r = pt.sum(mpi_a * mpi_a, dim=1, keepdim=True)
  p_r = mpi_a / (q_r + 1e-7)
  return pt.mean(pt.sum(p_r, dim=1))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(f'cuda:{rank}')
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(f'cuda:{rank}'))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(f'cuda:{rank}'))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def cumprod_exclusive(x):
  cp = pt.cumprod(x, 0)
  cp = pt.roll(cp, 1, 0)
  cp[0] = 1.0
  return cp

def getWarp3d(warp, interpolate = False):
  if not interpolate:
    depths = pt.repeat_interleave(pt.linspace(-1, 1, args.layers), args.sublayers).view(1, -1, 1, 1, 1).cuda()
  else:
    depths = pt.linspace(-1, 1, args.layers * args.sublayers).view(1, -1, 1, 1, 1).cuda()
  warp3d = warp[None] # 1, n, sel, 1, 2
  warp3d = pt.cat([warp3d, pt.ones_like(warp3d[:, :, :, :, :1]) * depths], -1)
  return warp3d

def normalized(v, dim):
  return v / (pt.norm(v, dim=dim, keepdim=True) + 1e-7)

class Basis(nn.Module):
  def __init__(self, shape, out_view):
    super().__init__()
    #choosing illumination model
    self.order = args.basis_order

    # network for learn basis
    self.seq_basis = nn.DataParallel(
      ReluMLP(
        args.basis_mlp, #basis_mlp
        args.basis_hidden, #basis_hidden
        self.order * 4,
        args.lrelu_slope,
        out_node = args.basis_out, #basis_out
      )
    )
    print('Basis Network:',self.seq_basis)

    # positional encoding pre compute
    self.pos_freq_viewing = pt.Tensor([(2 ** i) for i in range(self.order)]).view(1, 1, 1, 1, -1).cuda()

  def forward(self, sfm, feature, ref_coords, warp, planes, coeff = None):
    vi, xy = get_viewing_angle(sfm, feature, ref_coords, planes)
    n, sel = vi.shape[:2]

    # positional encoding for learn basis
    hinv_xy = vi[...,  :2, None] * self.pos_freq_viewing
    big = pt.reshape(hinv_xy, [n, sel, 1, hinv_xy.shape[-2] * hinv_xy.shape[-1]])
    vi = pt.cat([pt.sin(0.5*np.pi*big), pt.cos(0.5*np.pi*big)], -1)

    out2 = self.seq_basis(vi)
    out2 = pt.tanh(out2)

    vi = out2.view(n, sel, 1, 1, -1)

    coeff = coeff.view(coeff.shape[0], coeff.shape[1], coeff.shape[2], 3,  -1)
    coeff = pt.tanh(coeff)

    illumination = pt.sum(coeff * vi,-1).permute([0, 3, 1, 2])

    return illumination

def get_viewing_angle(sfm, feature, ref_coords, planes):
  camera = sfm.ref_rT.t() @ feature["center"][0] + sfm.ref_t

  # (n, rays, 2) -> (n, 2, rays)
  coords = ref_coords.permute([0, 2, 1])
  # (n, 2, rays) -> (n, 3, rays)
  coords = pt.cat([coords, pt.ones_like(coords[:, :1])], 1).cuda()

  # coords: (n, 3, rays)
  # viewed planes: (n, 1, 1)
  # xyz: (n, 3, rays)
  # xyz = coords * pt.Tensor(planes).view(-1, 1, 1).cuda()
  xyz = coords * planes.view(-1, 1, 1)

  ki = pt.tensor([[feature['fx'][0], 0, feature['px'][0]],
                  [0, feature['fy'][0], feature['py'][0]],
                  [0, 0, 1]], dtype=pt.float).inverse().cuda()

  xyz = ki @ xyz

  # camera: (3, 1) -> (1, 3, 1)
  # xyz: (n, 3, rays)
  # viewing_angle: (n, 3, rays)
  # viewing_angle = normalized(camera[None].cuda() - xyz, 1)
  inv_viewing_angle = normalized(xyz - camera[None].cuda(), 1)

  view = inv_viewing_angle.permute([0, 2, 1])
  xyz = xyz.permute([0, 2, 1])
  return view[:,:,None], xyz[:,:,None]

class Network(nn.Module):
  def __init__(self, shape, sfm, use_appearance=False, use_learnable_planes=False, device=0):
    super(Network, self).__init__()
    self.shape = [shape[2], shape[3]]
    if args.use_gaussian:
      in_channels = 127
      args.nc = 127
    else:
      in_channels = 5
      args.nc = 5

    if args.netG == 'unet':
      self.unet = UNet(
          in_channels,
          args.sublayers * args.layers + args.layers * 3
      )
    elif args.netG == 'spade':
      args.crop_size = shape[2]
      args.aspect_ratio = shape[2] / shape[3]
      self.unet = SPADEGenerator(args) 
      self.unet.init_weights()
    elif args.netG == 'unet_s':
      self.unet = UNet_small(
          in_channels,
          args.sublayers * args.layers + args.layers * 3
      )
    
    if use_appearance:
      # five camera, three channels 
      luminance_scale = pt.ones((5, 3)).to(f'cuda:{device}') 
      luminance_shift = pt.zeros((5, 3)).to(f'cuda:{device}') 
      self.luminance_scale = nn.Parameter(luminance_scale)
      self.luminance_shift = nn.Parameter(luminance_shift)
    
    if args.use_refinement:
      self.refinement_net = nn.Sequential(
                              nn.Conv2d(3, 32, 7, padding=3), 
                              nn.Conv2d(32, 3, 3, padding=1)
                            )
    
    if args.render_depth:
      self.rainbow_mpi = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=np.float32)
      for i,s in enumerate(np.linspace(1, 0, shape[0])):
        color = Rainbow(s)
        for c in range(3):
          self.rainbow_mpi[i,c] = color[c]
      self.rainbow_mpi = pt.from_numpy(self.rainbow_mpi).to('cuda:0')
    else:
      self.rainbow_mpi = None

    if sfm.dmin < 0 or sfm.dmax < 0:
      raise ValueError("invalid dmin dmax")

    self.regularize_planes = True 
    self.planes = pt.from_numpy(getPlanes(sfm, args.layers * args.sublayers)).float().to(f'cuda:{device}')
    if use_learnable_planes:
      planes_deform = pt.zeros(args.layers * args.sublayers).to(f'cuda:{device}')
      self.planes_deform = nn.Parameter(planes_deform)
      if self.regularize_planes:
        planes_deform_track = pt.zeros(args.layers * args.sublayers).to(f'cuda:{device}') 
        self.planes_deform_track = nn.Parameter(planes_deform_track)
        self.planes_deform_track.requires_grad = False
    
    if args.use_learnable_pose:
      pose_deform = pt.zeros(10000, 3, 4).to(f'cuda:{device}')
      self.pose_deform = nn.Parameter(pose_deform)
    else:
      self.pose_deform = None
    
    pos = True
    if pos:
      x = np.linspace(0,1,self.shape[1])
      y = np.linspace(0,1,self.shape[0])
      xv, yv = np.meshgrid(x, y)
      xv = np.expand_dims(xv, 2)
      yv = np.expand_dims(yv, 2)
      pos_enc = np.concatenate([xv, yv], axis=2)
      pos_enc = rearrange(pos_enc, 'h w c -> c h w')
      self.pos_enc = pt.from_numpy(pos_enc).to(f'cuda:{device}')[None].float()

      # self.planes = self.planes + self.planes_deform
    
    if args.render_depth:
      self.rainbow_mpi = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=np.float32)
      # self.rainbow_mpi = np.zeros((shape[0]+1, 3, shape[2], shape[3]), dtype=np.float32)
      # depth_cut = 100
      for i,s in enumerate(np.linspace(1, 0, shape[0])):
        color = Rainbow(s)
        for c in range(3):
          self.rainbow_mpi[i,c] = color[c]
        # if i < depth_cut:

      #  a background depth layer
      # for c in range(3):
          # self.rainbow_mpi[shape[0],c] = color[c]
      self.rainbow_mpi = pt.from_numpy(self.rainbow_mpi).to(f'cuda:{device}')
    else:
      self.rainbow_mpi = None
    
    print('All combined layers: {}'.format(args.layers * args.sublayers))
    print(self.planes)
    print('Using inverse depth: {}, Min depth: {}, Max depth: {}'.format(sfm.invz == 1, self.planes[0],self.planes[-1]))


  def forward(self, sfm, feature, output_shape, selection):
    ''' Rendering
    Args:
      sfm: reference camera parameter
      feature: target camera parameter
      output_shape: [h, w]. Desired rendered image
      selection: [ray]. pixel to train
    Returns:
      output: [1, 3, rays, 1] rendered image
    '''
    # get reference view features 
    ref_input = feature['ref_input'].cuda()
    ref_input = F.pad(ref_input, (args.offset, args.offset, args.offset, args.offset))
    ref_input = pt.cat([ref_input, self.pos_enc], axis=1) 

    node = 0
    out = self.unet(ref_input) 
    
    mpi_a = out[:, node:node + args.layers * args.sublayers, :, :]
    if args.render_depth:
      # set max depth
      mpi_a[:, -70:, :, :] = 100 

    
    # n -> number of layers * sublayers
    # (n, sel, 1, 2), (n, sel, 1, 1), (n, sel, 2)
    if args.use_learnable_planes:
      if not self.regularize_planes:
        planes = self.planes + self.planes_deform
      else:
        planes = self.planes + self.planes_deform
        # check whether there are two planes crossed
        diff = planes - torch.roll(planes, -1)
        if not torch.all(diff[:-1] < 0):
          # reset the deformation
          self.planes_deform.data = self.planes_deform_track.data
          planes = self.planes + self.planes_deform
        else:
          # back up the deform
          self.planes_deform_track.data = self.planes_deform.data
        
    else:
      planes = self.planes
    warp, ref_coords = computeHomoWarp(sfm,
                                self.shape,
                                sfm.offset,
                                output_shape, selection,
                                feature, planes, pose_deform=self.pose_deform)

    n = warp.shape[0]
    sel = warp.shape[1]

    mpi_a3d = mpi_a[None]
    warp_a3d = getWarp3d(warp, interpolate=True) 
    samples_a = F.grid_sample(mpi_a3d, warp_a3d, align_corners=True)
    

    node += args.layers * args.sublayers 
    # n, 1, sel, 1
    mpi_a_sig  = pt.sigmoid(samples_a[0].permute([1, 0, 2, 3]) - args.sigmoid_offset)

    if args.render_depth:
      # generate Rainbow MPI instead of real mpi to visualize the depth
      # self.rainbow_mpi: n, 3, h, w   warp: (n, sel, 1, 2)
      # Need: N, C, Din, Hin, Win;  N, Dout, Hout, Wout, 3
      rainbow_3d = self.rainbow_mpi.permute([1, 0, 2, 3])[None]
      warp3d = getWarp3d(warp)
      #samples: N, C, Dout, Hout, Wout
      samples = F.grid_sample(rainbow_3d, warp3d, align_corners=True)
      # (layers, out_node, rays, 1)
      rgb = samples[0].permute([1, 0, 2, 3])
    else:
      # if predicting rgb
      mpi_c = rearrange(out[:, node:, :, :], 'b (n c) w h -> (b n) c w h', c=3)
      mpi_sig = pt.sigmoid(mpi_c)

      # mpi_sig: n_layers(not multiplied by sublayers), 3, mpi_h, mpi_w   warp: (n, sel, 1, 2)
      # mpi_sig3d: 1, 3, n_layers, mpi_h, mpi_w 
      mpi_sig3d = mpi_sig.permute([1, 0, 2, 3])[None]
      # warp3d: 1, n, sel, 1, 3 
      warp3d = getWarp3d(warp)
      # samples: 1, 3, n, sel, 1
      samples = F.grid_sample(mpi_sig3d, warp3d, align_corners=True)
      # rgb:  n, 3, sel, 1
      rgb = samples[0].permute([1, 0, 2, 3])


    weight = cumprod_exclusive(1 - mpi_a_sig)

    output = pt.sum(weight * rgb * mpi_a_sig, dim=0, keepdim=True)

    if args.use_appearance:
      output = output * self.luminance_scale[feature['camera_index']][:,:,None, None] + output * self.luminance_shift[feature['camera_index']][:,:,None, None] 
    
    # end = timer()
    # end = time.time()
    # print(f"rendering:{end - start}")
    # print(output.shape)
    
    if args.use_refinement:
      output = rearrange(output, 'b c (h w) d -> b c h (w d)', h=output_shape[0])
      output = self.refinement_net(output)
      output = rearrange(output, 'b c h (w d) -> b c (h w) d', d=1)

    return output, mpi_a, mpi_a_sig

def getMPI(model, sfm, m = 1, dataloader = None):
  ''' convert from neural network to MPI planes
    Args:
      model: Neural net model
      sfm: reference camera parameter
      m: target camera parameter
      dataloader:
    Returns:
      output: dict({
        'mpi_c':'explicit coefficient k0',
        'mpi_a':'alpha transparentcy',
        'mpi_b':'basis',
        'mpi_v':'Kn coefficient'
      })
    '''
  sh = sfm.ref_cam['height'] + sfm.offset * 2
  sw = sfm.ref_cam['width'] + sfm.offset * 2

  y, x = pt.meshgrid([
    (pt.arange(0, sh, dtype=pt.float)) / (sh-1) * 2 - 1,
    (pt.arange(0, sw, dtype=pt.float)) / (sw-1) * 2 - 1])

  coords = pt.cat([x[:,:,None].cuda(), y[:,:,None].cuda()], -1)
  model.eval()
  sh_v = 400
  sw_v = 400
  rangex, rangey =  0.7, 0.6
  y_v, x_v = pt.meshgrid([
    (pt.linspace(-rangey, rangey, sh_v)),
    (pt.linspace(-rangex, rangex , sw_v))])
  #viewing [sh_v, sh_w, 2]
  viewing = pt.cat([x_v[:,:,None].cuda(), y_v[:,:,None].cuda()], -1)
  #hinv_xy [1, sh_v, sh_w, 2 * pos_lev]
  hinv_xy = viewing.view(1, sh_v, sw_v, 2, 1) * model.specular.pos_freq_viewing
  hinv_xy = hinv_xy.view(1, sh_v, sw_v, -1)
  #pe_view [1, sh_v, sw_v, 2 * 2 * pos_lev]
  pe_view = pt.cat([pt.sin(0.5*np.pi *hinv_xy), pt.cos(0.5*np.pi *hinv_xy)], -1)
  #out2 [1, sh, sw, num_basis]
  out2 = model.specular.seq_basis(pe_view)
  #imgs_b [num_basis, 1, sh_v, sw_v]
  imgs_b = pt.tanh(out2.permute([3, 0, 1, 2])).cpu().detach()

  n = args.layers * args.sublayers
  imgs_c, imgs_a, imgs_v = [], [], []

  with pt.no_grad():
    for i in range(0, n, m):

      #coords [sh, sw, 2] --> [1, sh, sw, 2, 1]
      #vxy [1, sh, sw, 2, pos_lev] -->  [1, sh, sw, 2*pos_lev]
      vxy = coords.view(1, sh, sw, 2, 1) * model.pos_freq
      vxy = vxy.view(1, sh, sw, -1)
      #vz [1, sh, sw, depth_lev]
      vz = pt.ones_like(coords.view(1, sh, sw, -1)[..., :1]) * model.z_coords[i:i+1] * model.depth_freq
      #vxyz [1, sh, sw, 2*pos_lev + depth_lev]
      vxyz = pt.cat([vxy, vz], -1)
      bigcoords = pt.cat([pt.sin(vxyz), pt.cos(vxyz)], -1)
      if sfm.offset > 270:
        out =  [model.seq1(bigy) for bigy in [bigcoords[:, :int(sh/2)], bigcoords[:, int(sh/2):]]]
        out = pt.cat(out, 1)
      else:
        out = model.seq1(bigcoords)
      node = 0

      mpi_a = out[..., node:node + 1].cpu()
      node +=1
      mpi_a = mpi_a.view(mpi_a.shape[0], 1, mpi_a.shape[1], mpi_a.shape[2])
      imgs_a.append(pt.sigmoid(mpi_a[0] - args.sigmoid_offset))

      if i % args.sublayers == 0:
        out = out[..., node:].cpu()
        mpi_v = out.view(out.shape[0], out.shape[1], out.shape[2], 3, -1)
        mpi_v = mpi_v.permute([0, 3, 1, 2, 4])
        mpi_v = mpi_v[0]
        mpi_v =  pt.tanh(mpi_v)
        imgs_v.append(mpi_v)

    mpi_c_sig = pt.sigmoid(model.mpi_c)
    mpi_a_sig = pt.stack(imgs_a, 0)
    mpi_v_tanh = pt.stack(imgs_v, 0)
    info = {
      'mpi_c': mpi_c_sig.cpu(),
      'mpi_a': mpi_a_sig.cpu(),
      'mpi_v' : mpi_v_tanh.cpu(),
      'mpi_b':  imgs_b.cpu()
    }

  pt.cuda.empty_cache()
  return info

def generateAlpha(model, dataset, dataloader, writer, runpath, suffix="", dataloader_train = None):
  ''' Prediction
    Args.
      model.   --> trained model
      dataset. --> valiade dataset
      writer.  --> tensorboard
  '''
  suffix_str = "/%06d" % suffix if isinstance(suffix, int) else "/"+str(suffix)
  # create webgl only when using -predict or finish training
  if not args.no_webgl and suffix =="":
    info = getMPI(model, dataset.sfm, dataloader = dataloader_train)

    outputCompactMPI(info,
                   dataset.sfm,
                   model.planes,
                   runpath + args.model_dir + suffix_str,
                   args.layers,
                   args.sublayers,
                   dataset.sfm.offset,
                   args.invz,
                   webpath=args.web_path,
                   web_width= args.web_width)

  if not args.no_eval and len(dataloader) > 0:
    out = evaluation(model,
                     dataset,
                     dataloader,
                     args.ray,
                     runpath + args.model_dir + suffix_str,
                     webpath=args.eval_path,
                     write_csv = not args.no_csv)
    if writer is not None and isinstance(suffix, int):
      for metrics, score in out.items():
        writer.add_scalar('METRICS/{}'.format(metrics), score, suffix)


def setLearningRate(optimizer, epoch):
  ds = int(epoch / args.decay_epoch)
  lr = args.lr * (args.decay_rate ** ds)

  optimizer.param_groups[0]['lr'] = lr
  if args.lrc > 0:
    optimizer.param_groups[1]['lr'] = lr * args.lrc
    if epoch >= 1:
      optimizer.param_groups[2]['lr'] = lr * args.lrc


def train(gpu, args):
  pt.manual_seed(1)
  np.random.seed(1)

  # torch.backends.cudnn.benchmark = True
  # torch.backends.cudnn.deterministic = False 

  if args.restart or args.clean:
    # os.system("rm -rf " + "runs/" + args.model_dir)
    os.system("rm -rf " + args.runpath + args.model_dir)
  if args.clean:
    exit()

  rank = gpu
  # if args.gpus > 1:
  dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
  dpath = args.scene

  dataset = loadDataset(dpath)
  sampler_train, sampler_val, dataloader_train, dataloader_val = prepareDataloaders(
    dataset,
    dpath,
    random_split = args.random_split,
    train_ratio = args.train_ratio,
    num_workers = args.num_workers,
    ddp=True,
    # ddp=False,
    world_size=args.world_size,
    rank=rank
  )

  mpi_h = int(dataset.sfm.ref_cam['height'] + dataset.sfm.offset * 2)
  mpi_w = int(dataset.sfm.ref_cam['width'] + dataset.sfm.offset * 2)
  args.ray = int((dataset.sfm.ref_cam['width'] / args.llff_width) * dataset.sfm.ref_cam['height']) * args.llff_width 
  model = Network((args.layers,
                 4,
                 mpi_h,
                 mpi_w,
                 ), dataset.sfm, 
                 use_appearance=args.use_appearance,
                 use_learnable_planes=args.use_learnable_planes,
                 device=rank)

  pt.cuda.set_device(gpu)
  model.cuda(gpu)
  # if args.gpus > 1:
  model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

  if args.lrc > 0:
    my_list = [name for name, params in model.named_parameters() if 'unet' in name or 'refinement_net' in name]
    my_list1 = [name for name, params in model.named_parameters() if 'luminance_scale' in name or 'luminance_shift' in name]
    my_list2 = [name for name, params in model.named_parameters() if 'planes_deform' in name]
    mlp_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    app_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list1, model.named_parameters()))))
    planes_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list2, model.named_parameters()))))
    optimizer = pt.optim.Adam([
      {'params': mlp_params, 'lr': 0},
      {'params': app_params, 'lr': 0},
      {'params': planes_params, 'lr': 0},
      ])
  else:
    optimizer = pt.optim.Adam(model.parameters(), lr=0)
  
  if args.use_learnable_pose:
    my_list = [name for name, params in model.named_parameters() if 'pose_deform' in name]
    pose_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    optimizer_pose = pt.optim.Adam([
      {'params': pose_params, 'lr': 0},
      ])


  start_epoch = 0
  runpath = args.runpath
  ckpt = runpath + args.model_dir + "/ckpt.pt"
  if os.path.exists(ckpt):
    start_epoch = loadFromCheckpoint(ckpt, model, optimizer, rank)
  elif args.pretrained != "":
    start_epoch = loadFromCheckpoint(runpath + args.pretrained + "/ckpt.pt", model, optimizer, rank)

  step = start_epoch * len(sampler_train)

  if args.epochs < 0 and args.steps < 0:
    raise Exception("Need to specify epochs or steps")

  if args.epochs < 0:
    args.epochs = int(np.ceil(args.steps / len(sampler_train)))

  if args.predict:
    generateAlpha(model, dataset, dataloader_val, None, runpath, dataloader_train = dataloader_train)
    if not args.no_video:
      if args.render_nearest:
        vid_path = 'video_nearest'
        render_type = 'nearest'
      elif args.render_viewing:
        vid_path = 'viewing_output'
        render_type = 'viewing'
      elif args.render_depth:
        vid_path = 'video_depth'
        render_type = 'depth'
      elif args.render_bullet_time:
        vid_path = 'video_bullet_time'
        render_type = 'bullet_time'
      else:
        vid_path = 'video_output'
        render_type = 'default'
      if args.render_fixed_view:
        if args.render_depth:
          vid_path = 'video_fixed_view_depth'
        else:
          vid_path = 'video_fixed_view'
        render_type = 'render_fixed_view'
      pt.cuda.empty_cache()
      # render_video(model, dataset, args.ray, os.path.join(runpath, vid_path, args.model_dir),
                  # render_type = render_type, dataloader = dataloader_train)
      if args.render_keypoints_path == "keypoints_2d":
        render_video_new(model, dataset, args.ray, os.path.join(runpath, vid_path, args.model_dir),
                    render_type = render_type, dataloader = dataloader_train)
      else:
        render_video_new(model, dataset, args.ray, os.path.join(runpath, vid_path, args.model_dir+"_"+args.render_keypoints_path),
                    render_type = render_type, dataloader = dataloader_train)
    if args.http:
      serve_files(args.model_dir, args.web_path)
    exit()


  if rank==0:
    backupConfigAndCode(runpath)
    ts = TrainingStatus(num_steps=args.epochs * len(sampler_train))
    writer = SummaryWriter(runpath + args.model_dir)
    writer.add_text('command',' '.join(sys.argv), 0)
    ts.tic()
  
  if args.perceptualloss > 0:
    perceptual_loss = VGGPerceptualLoss(rank=rank)

   # shift by 1 epoch to save last epoch to tensorboard
  for epoch in range(start_epoch, args.epochs+1):
    epoch_loss_total = 0
    epoch_mse = 0

    model.train()

    for i, feature in enumerate(dataloader_train):
      #print("step: {}".format(i))
      setLearningRate(optimizer, epoch)
      optimizer.zero_grad()

      if args.use_learnable_pose:
        if epoch >= 3:
          ds = int(epoch / args.decay_epoch)
          lr = args.lr * (args.decay_rate ** ds)
          optimizer_pose.param_groups[0]['lr'] = lr * 0.1
        optimizer_pose.zero_grad()

      output_shape = feature['image'].shape[-2:]
      # print(output_shape)

      #sample L-shaped rays
      # sel = Lsel(output_shape, args.ray)

      # Select all rays for CNN
      sel = Asel(output_shape)

      gt = feature['image']
      gt = gt.view(gt.shape[0], gt.shape[1], gt.shape[2] * gt.shape[3])
      gt = gt[:, :, sel, None].cuda()

      if args.mask_weight > 0:
        mask = feature['mask_input']
        mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2] * mask.shape[3])
        mask = mask[:, :, sel, None].cuda()
        


      output, mpi_a, mpi_a_sig = model(dataset.sfm, feature, output_shape, sel)
      # if args.use_appearance:

      if args.mask_weight > 0:
        mse = pt.mean((output - gt) ** 2 * mask)
      else:
        mse = pt.mean((output - gt) ** 2)

      loss_total = mse

      #tvc regularizer
      # loss_total = loss_total + tvc

      output = rearrange(output, 'b c (h w) n -> b c h (w n)', h=output_shape[0])
      gt = rearrange(gt, 'b c (h w) n -> b c h (w n)', h=output_shape[0])

      # grad loss
      if args.gradloss > 0:
        ox = output - pt.roll(output, 1, 2)
        oy = output - pt.roll(output, 1, 3)
        gx = gt - pt.roll(gt, 1, 2)
        gy = gt - pt.roll(gt, 1, 3)
        loss_total = loss_total + args.gradloss * (pt.mean(pt.abs(ox - gx)) + pt.mean(pt.abs(oy - gy)))
      
      if args.perceptualloss > 0:
        loss_total = loss_total + args.perceptualloss * perceptual_loss(output, gt)
      
      if args.sparsityloss > 0:
        sparsity_loss_ref = get_sparsity_loss(pt.sigmoid(mpi_a))
        sparsity_loss_train = get_sparsity_loss(rearrange(mpi_a_sig, 'c n1 (h w) n2 -> n1 c h (w n2)', h=output_shape[0]))
        loss_total = loss_total + args.sparsityloss * (sparsity_loss_ref + sparsity_loss_train)

      epoch_loss_total += loss_total
      epoch_mse += mse

      loss_total.backward()
      optimizer.step()

      if args.use_learnable_pose:
        output, mpi_a, mpi_a_sig = model(dataset.sfm, feature, output_shape, sel)
        gt = rearrange(gt, 'b c h (w n) -> b c (h w) n', n=1)
        mask[mask == args.mask_weight] = 0
        loss_new = pt.mean((output - gt) ** 2 * mask) 
        loss_new.backward()
        optimizer_pose.step()

      step += 1
      if rank==0:
        toc_msg = ts.toc(step, loss_total.item())
        if step % args.tb_toc == 0:  print(toc_msg)
        ts.tic()

    if rank==0:
      writer.add_scalar('loss/total', epoch_loss_total/len(sampler_train), epoch)
      writer.add_scalar('loss/mse', epoch_mse/len(sampler_train), epoch)

    if epoch % args.tb_saveimage == 0 and args.tb_saveimage > 0 and rank == 0:
      with pt.no_grad():
        render = patch_render(model, dataset.sfm, feature, args.ray)
        # Spec = getMPI(model, dataset.sfm, m = args.sublayers, dataloader = None)

        writer.add_image('images/render', pt.cat([feature['image'].cuda(), render], 2)[0], epoch)
        pt.cuda.empty_cache()

    if epoch % args.tb_savempi == 0 and args.tb_savempi > 0 and epoch > 0 and rank == 0:
      generateAlpha(model, dataset, dataloader_val, writer, runpath, epoch)
      pt.cuda.empty_cache()


    if (epoch+1) % args.checkpoint == 0 or epoch == args.epochs-1:
      if np.isnan(loss_total.item()):
        exit()
      checkpoint(ckpt, model, optimizer, epoch+1, rank)

  print('Finished Training')
  generateAlpha(model, dataset, dataloader_val, None, runpath, dataloader_train = dataloader_train)
  if not args.no_video:
    render_video(model, dataset, args.ray, os.path.join(runpath, 'video_output', args.model_dir))
  if args.http:
    serve_files(args.model_dir, args.web_path)

def checkpoint(file, model, optimizer, epoch, rank):
  if rank==0:
    print("Checkpointing Model @ Epoch %d ..." % epoch)
    pt.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      }, file)

def loadFromCheckpoint(file, model, optimizer, rank):
  map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
  checkpoint = pt.load(file, map_location=map_location)
  # model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  model.load_state_dict(checkpoint['model_state_dict'])
  # print var name
  # for var_name in checkpoint['model_state_dict']:
  #   # print(var_name) 
  #   if not 'unet' in var_name:
  #     print(var_name)
  #     print(checkpoint['model_state_dict'][var_name])
  start_epoch = checkpoint['epoch']
  print("Loading %s Model @ Epoch %d" % (args.pretrained, start_epoch))
  return start_epoch

def backupConfigAndCode(runpath):
  if args.predict or args.clean:
    return
  model_path = os.path.join(runpath, args.model_dir)
  os.makedirs(model_path, exist_ok = True)
  now = datetime.now()
  t = now.strftime("_%Y_%m_%d_%H:%M:%S")
  with open(model_path + "/args.json", 'w') as out:
    json.dump(vars(args), out, indent=2, sort_keys=True)
  os.system("cp " + os.path.abspath(__file__) + " " + model_path + "/")
  # os.system("cp " + os.path.abspath(__file__) + " " + model_path + "/" + __file__.replace(".py", t + ".py"))
  os.system("cp " + model_path + "/args.json " + model_path + "/args" + t + ".json")


def loadDataset(dpath):
  # if dataset directory has only image, create LLFF poses
  colmapGenPoses(dpath)

  if args.scale == -1:
    args.scale = getDatasetScale(dpath, args.deepview_width, args.llff_width)

  if is_deepview(dpath) and args.ref_img == '':
    with open(dpath + "/ref_image.txt", "r") as fi:
      args.ref_img = str(fi.readline().strip())
  render_style = 'llff' if args.nice_llff else 'shiny' if args.nice_shiny else ''
  return OrbiterDatasetMulti(dpath, ref_img=args.ref_img, scale=args.scale,
                           dmin=args.dmin,
                           dmax=args.dmax,
                           invz=args.invz,
                           render_style=render_style,
                           offset=args.offset,
                           cv2resize=args.cv2resize,
                           mask_weight=args.mask_weight,
                           use_gaussian=args.use_gaussian,
                           render_keypoints_path=args.render_keypoints_path)


if __name__ == "__main__":
  sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
  # train()
  args.world_size = args.gpus
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '1242'
  colmapGenPoses(args.scene)
  if args.gpus > 1:
    mp.spawn(train, nprocs=args.gpus, args=(args,))
  else:
    train(0, args)
