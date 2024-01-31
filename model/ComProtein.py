# This file was written for the situation that only one available large-memory GPU (only Pytorch without HFAI)
import itertools
import math
import os
import random

import pyfastx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration, MinLengthLogitsProcessor
from transformers.generation.configuration_utils import *
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import GenerateOutput, GreedySearchOutput, GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from typing import Callable, List
from mmd_vae_v2 import *
from discriminator import *
import re
import numpy as np
import warnings
import inspect
import time
import math


class CustomizedDataset(Dataset):
    def __init__(self, data, seq_length):
        # Define the tokenizer user in dataset
        self.tokenizer = T5Tokenizer.from_pretrained('your_checkpoints_path_of_ProtT5-XL-UniRef50', do_lower_case=False)

        # Convert protein sequences into corresponding tensors
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        self.input_ids = []
        self.attention_mask = []
        self.seq_length = seq_length
        for (temp_input_ids, temp_attention_mask) in zip(ids['input_ids'], ids['attention_mask']):
            cur_input_ids = torch.tensor(temp_input_ids)
            cur_attention_mask = torch.tensor(temp_attention_mask)
            # print(cur_input_ids.shape[1], cur_attention_mask.shape[1])
            if cur_input_ids.shape[0] < self.seq_length:
                cur_input_ids = torch.cat([cur_input_ids, torch.zeros(1, self.seq_length - cur_input_ids.shape[0],
                                                                      dtype=torch.int64)], dim=1).to('cuda:0')
            elif cur_input_ids.shape[0] == self.seq_length:
                cur_input_ids = cur_input_ids.to('cuda:0')
            else:
                raise ValueError('INPUT_IDS LENGTH PROCESS ERROR!')
            self.input_ids.append(cur_input_ids)
            if cur_attention_mask.shape[0] < self.seq_length:
                cur_attention_mask = torch.cat([cur_attention_mask, torch.zeros(1, 500 - cur_attention_mask.shape[0],
                                                                                dtype=torch.int64)], dim=1).to('cuda:0')
            elif cur_attention_mask.shape[0] == self.seq_length:
                cur_attention_mask = cur_attention_mask.to('cuda:0')
            else:
                raise ValueError('ATTENTION_MASK LENGTH PROCESS ERROR!')
            self.attention_mask.append(cur_attention_mask)
        assert len(self.input_ids) == len(self.attention_mask)

        # Validation
        for item in self.input_ids:
            if item.shape[0] != self.seq_length:
                raise ValueError('DATA PROCESS ERROR!')
        for item in self.attention_mask:
            if item.shape[0] != self.seq_length:
                raise ValueError('DATA PROCESS ERROR!')
        self.len = len(self.input_ids)
        print(len(self.input_ids), len(self.attention_mask))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]


class CycleProtTransVAE(nn.Module):

    def __init__(self, latent_dim, batch_size, device, lambda_center=0.5, lambda_triplet=0.7, accumulation_step=10):
        super(CycleProtTransVAE, self).__init__()

        # Initialization of ProtT5 model
        self.tokenizer = T5Tokenizer.from_pretrained('your_checkpoints_path_of_ProtT5-XL-UniRef50', do_lower_case=False)
        self.t5 = T5ForConditionalGeneration.from_pretrained("your_checkpoints_path_of_ProtT5-XL-UniRef50")
        self.t5_encoder = self.t5.get_encoder()
        self.t5_decoder = self.t5.get_decoder()
        self.lm_head = self.t5.lm_head

        # Dropout and BN
        self.t5.eval()

        # Freeze the parameters in T5 model
        for p in self.t5.parameters():
            p.requires_grad = False

        # Other parameters which will be used in the next
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lambda_center = lambda_center
        self.lambda_triplet = lambda_triplet
        self.accumulation_step = accumulation_step

        # Initialize MMD-VAE for A to B and B to A
        self.mmd_vae_ab = MMD_VAE(zdims=self.latent_dim, NC=self.batch_size, NEF=self.batch_size * 2,
                                  NDF=self.batch_size * 2)
        self.mmd_vae_ba = MMD_VAE(zdims=self.latent_dim, NC=self.batch_size, NEF=self.batch_size * 2,
                                  NDF=self.batch_size * 2)

        # Initialize discriminator
        self.discriminator_A = Discriminator(seq_length=500, latent_dim=1024)
        self.discriminator_B = Discriminator(seq_length=500, latent_dim=1024)

        # Real and fake labels used in discriminator
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        # Define representation projection space
        # self.proj_layer1 = nn.Linear(in_features=1024, out_features=256)
        # self.proj_layer2 = nn.Linear(in_features=256, out_features=16)
        # self.relu = nn.ReLU()
        self.rep_proj_sequence = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=16)
        )

        # Initialize optimizers for generator-like components and discriminator
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.mmd_vae_ab.parameters(), self.mmd_vae_ba.parameters(),
                                                            self.rep_proj_sequence.parameters()),
                                            lr=0.0001)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.discriminator_A.parameters(),
                                                            self.discriminator_B.parameters()),
                                            lr=0.001)

        # Finally process the output logits
        self.logits_processor = MinLengthLogitsProcessor(min_length=200, eos_token_id=1)

        # Calculate loss inside this model, and then define the mse_loss and l1_loss in the process in additional
        self.gan_loss = GANLoss(gan_mode='lsgan')
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.triplet_loss = nn.TripletMarginLoss()

        # CUDA
        self.device = device

        print(f"Initialize model with lambda_center: {self.lambda_center}, lambda_triplet: {self.lambda_triplet}")

    def forward(self, input_A, input_B):
        # Take the assumption that 'input' is a set of non-MSA sequences
        # Get T5 model's output 't5_encoder_output_A' and 't5_encoder_output_B'
        input_ids_A = input_A[0]
        attention_mask_A = input_A[1]
        input_ids_B = input_B[0]
        attention_mask_B = input_B[1]
        # print(input_ids_A.shape, attention_mask_A.shape, input_ids_B.shape, attention_mask_B.shape)
        self.pro_A = self.t5_encoder(input_ids=input_ids_A, attention_mask=attention_mask_A)[0]
        self.pro_B = self.t5_encoder(input_ids=input_ids_B, attention_mask=attention_mask_B)[0]
        self.pro_A = torch.unsqueeze(self.pro_A, dim=0)
        self.pro_B = torch.unsqueeze(self.pro_B, dim=0)

        # After getting meaningful representation from ProtT5, use them through MMD-VAE
        # Generate from A to B and B to A for discriminator loss
        self.z_AtoB, self.pro_AtoB = self.mmd_vae_ab(self.pro_A)
        self.z_BtoA, self.pro_BtoA = self.mmd_vae_ba(self.pro_B)
        # print(pro_A.shape, pro_AtoB.shape) torch.Size([batch_size, constant_length, 1024])

        # Generate from A to B to A and B to A to B for reconstruction loss
        self.z_AtoBtoA, self.pro_AtoBtoA = self.mmd_vae_ba(self.pro_AtoB)
        self.z_BtoAtoB, self.pro_BtoAtoB = self.mmd_vae_ab(self.pro_BtoA)

        # Generate from B to B and A to A for identity loss
        self.z_BtoB, self.pro_BtoB = self.mmd_vae_ab(self.pro_B)
        self.z_AtoA, self.pro_AtoA = self.mmd_vae_ba(self.pro_A)

        # self.calculate_loss(pro_A, pro_B, pro_AtoB, pro_BtoA, pro_AtoBtoA, pro_BtoAtoB, pro_BtoB, pro_AtoA)
        # Return key components for calculating loss
        # return z_AtoB, z_BtoA, pro_A, pro_B, pro_AtoB, pro_BtoA, pro_AtoBtoA, pro_BtoAtoB, pro_BtoB, pro_AtoA

    def convert_to_display(self, samples):
        cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height, cnt, cnt, width])
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        samples = np.reshape(samples, [height * cnt, width * cnt])
        return samples

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def mmd_loss(self, z_AtoB, z_BtoA):
        # true_samples = torch.randn(BATCH_SIZE, ZDIMS, requires_grad=False).to(device)
        true_samples = torch.randn(z_AtoB.shape[0], z_AtoB.shape[1]).to(self.device)
        mmd_AtoB_gauss = self.compute_mmd(true_samples, z_AtoB)
        mmd_BtoA_gauss = self.compute_mmd(true_samples, z_BtoA)
        mmd_AtoB_BtoA = self.compute_mmd(z_AtoB, z_BtoA)
        loss = mmd_AtoB_gauss + mmd_BtoA_gauss + mmd_AtoB_BtoA
        return loss

    def center_loss(self, pro_A, pro_B, pro_generated):
        # Get the cluster center of real data and fake data
        center_pro_A = torch.sum(pro_A, dim=0)
        center_pro_A = torch.div(center_pro_A, self.batch_size)

        center_pro_B = torch.sum(pro_B, dim=0)
        center_pro_B = torch.div(center_pro_B, self.batch_size)

        center_pro_Generated = torch.sum(pro_generated, dim=0)
        center_pro_Generated = torch.div(center_pro_Generated, self.batch_size)

        center_loss = self.lambda_center * self.mse_loss(center_pro_A, center_pro_Generated) + \
                      (1 - self.lambda_center) * self.mse_loss(center_pro_B, center_pro_Generated)
        return center_loss

    def rep_proj_loss(self, pro_A, pro_B, pro_AtoB, pro_BtoA, cur_global_step):
        # Project protein sequences' representation into cluster center space
        proj_pro_A = self.rep_proj_sequence(pro_A)
        proj_pro_B = self.rep_proj_sequence(pro_B)
        proj_pro_AtoB = self.rep_proj_sequence(pro_AtoB)
        proj_pro_BtoA = self.rep_proj_sequence(pro_BtoA)

        # Calculate corresponding distances
        # distance_pro_AtoB_and_pro_A = self.mse_loss(proj_pro_AtoB, proj_pro_A)
        # distance_pro_AtoB_and_pro_B = self.mse_loss(proj_pro_AtoB, proj_pro_B)
        # distance_pro_BtoA_and_pro_A = self.mse_loss(proj_pro_BtoA, proj_pro_A)
        # distance_pro_BtoA_and_pro_B = self.mse_loss(proj_pro_BtoA, proj_pro_B)
        # distance_pro_A_and_pro_B = self.mse_loss(proj_pro_A, proj_pro_B)

        # Calculate corresponding loss
        # Generated protein sequences are located in the line between pro_A and pro_B
        # th_1 = distance_pro_AtoB_and_pro_A + distance_pro_AtoB_and_pro_B - distance_pro_A_and_pro_B
        # loss_1 = th_1 if th_1.item() > 0 else torch.zeros(th_1.shape).to(self.device)
        # th_2 = distance_pro_BtoA_and_pro_A + distance_pro_BtoA_and_pro_B - distance_pro_A_and_pro_B
        # loss_2 = th_2 if th_2.item() > 0 else torch.zeros(th_2.shape).to(self.device)

        # Use lambda_triplet to prevent generated protein sequences from being too close to target domain
        # th_3 = distance_pro_AtoB_and_pro_B - torch.tensor(self.lambda_triplet).to(self.device)
        # loss_3 = th_3 if th_3.item() > 0 else torch.zeros(th_3.shape).to(self.device)
        # th_4 = distance_pro_BtoA_and_pro_A - torch.tensor(self.lambda_triplet).to(self.device)
        # loss_4 = th_4 if th_4.item() > 0 else torch.zeros(th_4.shape).to(self.device)

        # loss_1 = self.triplet_loss(anchor=proj_pro_AtoB, positive=proj_pro_B, negative=proj_pro_A)
        # loss_2 = self.triplet_loss(anchor=proj_pro_BtoA, positive=proj_pro_A, negative=proj_pro_B)
        # loss_3 = self.triplet_loss(anchor=proj_pro_AtoB, positive=proj_pro_A, negative=proj_pro_B)
        # loss_4 = self.triplet_loss(anchor=proj_pro_BtoA, positive=proj_pro_B, negative=proj_pro_A)

        # Calculate corresponding mean centroid
        center_proj_pro_A = torch.sum(proj_pro_A, dim=0)
        center_proj_pro_A = torch.div(center_proj_pro_A, self.batch_size)
        center_proj_pro_B = torch.sum(proj_pro_B, dim=0)
        center_proj_pro_B = torch.div(center_proj_pro_B, self.batch_size)
        center_proj_pro_AtoB = torch.sum(proj_pro_AtoB, dim=0)
        center_proj_pro_AtoB = torch.div(center_proj_pro_AtoB, self.batch_size)
        center_proj_pro_BtoA = torch.sum(proj_pro_BtoA, dim=0)
        center_proj_pro_BtoA = torch.div(center_proj_pro_BtoA, self.batch_size)

        # Define distance constraint
        loss_1 = (1 - self.lambda_triplet) * self.mse_loss(center_proj_pro_A, center_proj_pro_AtoB) + \
                 self.lambda_triplet * self.mse_loss(center_proj_pro_B, center_proj_pro_AtoB)
        loss_2 = (1 - self.lambda_triplet) * self.mse_loss(center_proj_pro_B, center_proj_pro_BtoA) + \
                 self.lambda_triplet * self.mse_loss(center_proj_pro_A, center_proj_pro_BtoA)

        rep_proj_loss = 100000 * math.exp(1 - cur_global_step / 10.0) * (loss_1 + loss_2)

        return rep_proj_loss

    # def calculate_loss(self):
    #     # print(pro_A.shape, pro_AtoB.shape)
    #     discriminator_loss = 10 * (self.mse_loss(self.pro_A, self.pro_BtoA) + self.mse_loss(self.pro_B, self.pro_AtoB))
    #     reconstruction_loss = self.l1_loss(self.pro_A, self.pro_AtoBtoA) + self.l1_loss(self.pro_B, self.pro_BtoAtoB)
    #     identity_loss = self.l1_loss(self.pro_A, self.pro_AtoA) + self.l1_loss(self.pro_B, self.pro_BtoB)
    #     mmd_loss = 100 * self.mmd_loss(self.z_AtoB, self.z_BtoA)
    #     total_loss = discriminator_loss + reconstruction_loss + identity_loss + mmd_loss
    #     return total_loss, discriminator_loss, reconstruction_loss, identity_loss, mmd_loss

    def backward_G(self, cur_global_step):
        # GAN loss
        self.loss_G_A = self.gan_loss(self.discriminator_A(self.pro_AtoB), True)
        self.loss_G_B = self.gan_loss(self.discriminator_B(self.pro_BtoA), True)

        # Reconstruction (Cycle) loss
        self.loss_cycle_A = self.l1_loss(self.pro_A, self.pro_AtoBtoA)
        self.loss_cycle_B = self.l1_loss(self.pro_B, self.pro_BtoAtoB)

        # Identity loss
        self.loss_idt_A = self.l1_loss(self.pro_A, self.pro_AtoA)
        self.loss_idt_B = self.l1_loss(self.pro_B, self.pro_BtoB)

        # MMD loss
        self.loss_mmd = self.mmd_loss(self.z_AtoB, self.z_BtoA)

        # Center loss
        # self.loss_center = self.center_loss(self.pro_A, self.pro_B, self.pro_AtoB) + \
        #                    self.center_loss(self.pro_A, self.pro_B, self.pro_BtoA)

        # Triplet loss
        self.loss_triplet = self.rep_proj_loss(self.pro_A, self.pro_B, self.pro_AtoB, self.pro_BtoA, cur_global_step)

        # combined loss and calculate gradientssearch = SerpAPIWrapper
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + \
                      self.loss_idt_B + self.loss_mmd + self.loss_triplet
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.gan_loss(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = ((loss_D_real + loss_D_fake) * 0.5) / self.accumulation_step
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D_A = self.backward_D_basic(self.discriminator_A, self.pro_B, self.pro_AtoB)
        self.loss_D_B = self.backward_D_basic(self.discriminator_B, self.pro_A, self.pro_BtoA)
        self.loss_D = self.loss_D_A + self.loss_D_B

    def optimize_parameters(self, cur_global_step):
        # Generator-like components
        self.optimizer_G.zero_grad()
        self.backward_G(cur_global_step)
        self.optimizer_G.step()

        # Discriminator
        # TODO Update discriminators in the way of accumulation
        self.backward_D()
        if cur_global_step % self.accumulation_step == 0:
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()

        # Calculate total loss
        self.loss_total = self.loss_G + self.loss_D

        return self.loss_total, self.loss_G, self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B, \
               self.loss_idt_A, self.loss_idt_B, self.loss_mmd, self.loss_triplet, self.loss_D, self.loss_D_A, \
               self.loss_D_B



def main():
    # Define distributed training settings
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    if device is None:
        raise Exception("No available GPU! Please try again!")

    # Define training parameters
    epoch = 3000
    batch_size = 128
    data_A = []
    data_B = []
    latent_dim = 20
    save_path = 'output/'
    writer = SummaryWriter(log_dir="logs")
    global_step = 0
    lambda_center = 0.5
    lambda_triplet = 0.7

    # Utilize a unified constant sequences length for providing more controllable generation (500)
    seq_length = 500

    # Read in sequences data
    print("Loading sequences data...")
    fasta1 = pyfastx.Fastx('your_domain_A_path')
    fasta2 = pyfastx.Fastx('yout_domain_B_path')
    for seq in fasta1:
        # If length >= 500, cut it into [0: 498] which contains 499 tokens (for additional token </s>)
        if len(seq[1].strip()) >= seq_length:
            data_A.append(seq[1].strip()[0: seq_length - 1])
        else:
            data_A.append(seq[1].strip())
    for seq in fasta2:
        if len(seq[1].strip()) >= seq_length:
            data_B.append(seq[1].strip()[0: seq_length - 1])
        else:
            data_B.append(seq[1].strip())
    print("Loading sequences data in new method finished!")
    print(len(data_A), len(data_B))

    # To solving the matter that different datasets' length (the number of data), pro_A is always smaller or equal
    if len(data_A) <= len(data_B):
        pro_A_dataset = CustomizedDataset(data_A, seq_length)
        pro_B_dataset = CustomizedDataset(data_B, seq_length)
    else:
        pro_A_dataset = CustomizedDataset(data_B, seq_length)
        pro_B_dataset = CustomizedDataset(data_A, seq_length)

    pro_A_loader = DataLoader(dataset=pro_A_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    pro_B_loader = DataLoader(dataset=pro_B_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model and optimizer (filter out parameters whose 'requires_grad' is 'False')
    print("Initialize training model...")
    model = CycleProtTransVAE(latent_dim=latent_dim, batch_size=batch_size, lambda_center=lambda_center,
                              lambda_triplet=lambda_triplet, device=device).to(device)
    # print("=================================================")
    # print(model.tokenizer.get_vocab())
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    print("Initialize model finished!")

    # Verify all parameters about T5 model are frozen
    # flag_frozen = True
    # for p in model.t5.parameters():
    #     if p.requires_grad:
    #         flag_frozen = False
    # for p in model.t5_encoder.parameters():
    #     if p.requires_grad:
    #         flag_frozen = False
    # for p in model.t5_decoder.parameters():
    #     if p.requires_grad:
    #         flag_frozen = False
    # for p in model.lm_head.parameters():
    #     if p.requires_grad:
    #         flag_frozen = False
    # print(flag_frozen)
    # Variable 'flag_frozen' remains True

    # Starting training
    print("Start training...")
    for i in range(epoch):
        for j, data in enumerate(zip(pro_A_loader, pro_B_loader)):
            global_step += 1
            data_from_iter_A = data[0]
            data_from_iter_B = data[1]
            model(data_from_iter_A, data_from_iter_B)
            loss_total, loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, loss_mmd, \
            loss_triplet, loss_D, loss_D_A, loss_D_B = model.optimize_parameters(global_step)
            # return self.total_loss, self.loss_G, self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B, \
            #        self.loss_idt_A, self.loss_idt_B, self.mmd_loss, self.loss_D, self.loss_D_A, self.loss_D_B
            print("epoch: {:2d}, iteration: {:3d}, loss_total: {:6.4f}, loss_G: {:6.4f}, loss_G_A: {:6.4f}, "
                  "loss_G_B: {:6.4f}, loss_cycle_A: {:6.4f}, loss_cycle_B: {:6.4f}, loss_idt_A: {:6.4f}, "
                  "loss_idt_B: {:6.4f}, loss_mmd: {:6.4f}, loss_triplet: {:6.4f}, loss_D: {:6.4f}, loss_D_A: {:6.4f}, "
                  "loss_D_B: {:6.4f}"
                  .format(i + 1, j + 1, loss_total, loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A,
                          loss_idt_B, loss_mmd, loss_triplet, loss_D, loss_D_A, loss_D_B))
            # print("===================================================================================================")

            writer.add_scalar(tag='loss_total', scalar_value=loss_total, global_step=global_step)
            writer.add_scalar(tag='loss_G', scalar_value=loss_G, global_step=global_step)
            writer.add_scalar(tag='loss_G_A', scalar_value=loss_G_A, global_step=global_step)
            writer.add_scalar(tag='loss_G_B', scalar_value=loss_G_B, global_step=global_step)
            writer.add_scalar(tag='loss_cycle_A', scalar_value=loss_cycle_A, global_step=global_step)
            writer.add_scalar(tag='loss_cycle_B', scalar_value=loss_cycle_B, global_step=global_step)
            writer.add_scalar(tag='loss_idt_A', scalar_value=loss_idt_A, global_step=global_step)
            writer.add_scalar(tag='loss_idt_B', scalar_value=loss_idt_B, global_step=global_step)
            writer.add_scalar(tag='loss_mmd', scalar_value=loss_mmd, global_step=global_step)
            writer.add_scalar(tag='loss_triplet', scalar_value=loss_triplet, global_step=global_step)
            writer.add_scalar(tag='loss_D', scalar_value=loss_D, global_step=global_step)
            writer.add_scalar(tag='loss_D_A', scalar_value=loss_D_A, global_step=global_step)
            writer.add_scalar(tag='loss_D_B', scalar_value=loss_D_B, global_step=global_step)

    torch.save(model.mmd_vae_ab.state_dict(), save_path + "training_final_mmd_vae_ab.pt")
    torch.save(model.mmd_vae_ba.state_dict(), save_path + "training_final_mmd_vae_ba.pt")
    torch.save(model.discriminator_A.state_dict(), save_path + "training_final_discriminator_A.pt")
    torch.save(model.discriminator_B.state_dict(), save_path + "training_final_discriminator_B.pt")
    torch.save(model.rep_proj_sequence.state_dict(), save_path + "training_final_rep_proj_sequence.pt")


if __name__ == '__main__':
    # Start
    main()
