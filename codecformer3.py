import dac
print("descript audio codec v",dac.__version__)
from audiotools import AudioSignal
import torchaudio.transforms as T
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import collections
import warnings
import pyloudnorm
import random
import numpy as np
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
from speechbrain.lobes.models.transformer.Transformer import TransformerDecoder
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

class DACWrapper():
    '''
    Wrapper model for Descript Audio Codec
    '''
    def __init__(self, input_sample_rate=8000, DAC_model_path = None, DAC_sample_rate = 16000, Freeze=True):
        '''
        input_sample_rate: defaults to 8000 as it common in speech separation
        Model Path: Please provde the model path to the DAC model, otherwise the 16KHz model will be automatically downloaded.
        DAC_sample_rate: defaults to 16000. If using a DAC model other than the 16khz model, please specify this number.
        '''
        super(DACWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.DAC_sample_rate = DAC_sample_rate
        
        if DAC_model_path == None:
            model_path = dac.utils.download(model_type="16khz")
        
        self.model = dac.DAC.load(model_path)

        self.dac_sampler = T.Resample(input_sample_rate, DAC_sample_rate)
        self.org_sampler = T.Resample(DAC_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')

        else:
            print(f'Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        torchaudio resample function used here only requires last dimension to be time.
        condition: "dac" to set the sampling rate to the DAC sampling rate
                   "org" to set the sampling rate back to the original sampling rate

        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        '''
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1 #model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate
        if condition == "dac":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        
        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(),dim=2,keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be the torch tensor as per the data loader
        Expects x to be of dimensions [Batch, Channel, Time]
        '''
        #keep original lengths from pre resampling
        original_length = x.shape[-1]

        #make the input into the desired sampling rate
        x = self.resample_audio(x, "dac")

        #Now to perform the encoding
        x = self.model.preprocess(x, self.DAC_sample_rate)
        x_enc = self.model.encoder(x)

        return x_enc, original_length #the codec outputs a different length due to the masking
    
    def get_quantized_features(self, x):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        '''

        x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat = self.model.quantizer(x, None)
        return x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat

    def get_decoded_signal(self, x, original_length):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        original length is the original length of the input audio
        '''
        y_hat = self.model.decoder(x)

        # out = y_hat[...,:original_length]
        # out = self.resample_audio(out, "org")
        out = self.resample_audio(y_hat, "org")
        
        # T might have changed due to model. If so, fix it here
        if out.shape[-1] != original_length:
            T_origin = original_length
            T_est = out.shape[-1]
            
            if T_origin > T_est:
                out = F.pad(out, (0, T_origin - T_est))
            else:
                out = out[:, :, :T_origin]
        return out

class simpleSeparator2(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):
        
        x = self.ch_down(x)
        #[B,N,L]
        x = x.permute(0,2,1)
        #[B,L,N]
        x_b = self.block(x)
        #[B,L,N]
        x_b = x_b.permute(0,2,1)
        #[B,N,L]
        x = self.ch_up(x_b)

        B, N, L = x.shape
        masks = self.masker(x)
        
        #[B,N*num_spks,L]
        masks = masks.view(B*self.num_spks,-1,L)
        
        #[B*num_spks, N, L]
        x = self.output(masks) * self.output_gate(masks)
        x = self.activation(x)

        #[B*num_spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        
        # [B, spks, N, L]
        x = x.transpose(0,1)
        # [spks, B, N, L]

        return x