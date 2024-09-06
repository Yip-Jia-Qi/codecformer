import dac
print("descript audio codec v",dac.__version__)
import torchaudio.transforms as T
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
from wavtokenizer.decoder.pretrained import WavTokenizer

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
    def __init__(self, num_spks, channels, block, block_channels, activation=None):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        if not activation:
            self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        else:
            self.activation = activation
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

class WavTokenizerWrapper:
    '''
    Wrapper model for WavTokenizer
    '''
    def __init__(self, input_sample_rate=8000, model_config_path=None, model_ckpt_path=None, tokenizer_sample_rate=24000, Freeze=True):
        '''
        input_sample_rate: defaults to 8000 as expected file input
        model_config_path: Path to the config file for WavTokenizer
        model_ckpt_path: Path to the checkpoint file for WavTokenizer
        tokenizer_sample_rate: defaults to 24000. Specify if using a model with a different sample rate.
        '''
        super(WavTokenizerWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.tokenizer_sample_rate = tokenizer_sample_rate

        if model_config_path is None or model_ckpt_path is None:
            raise ValueError("Please provide both the model config and checkpoint paths.")

        self.model = WavTokenizer.from_pretrained0802(model_config_path, model_ckpt_path)

        self.dac_sampler = T.Resample(input_sample_rate, tokenizer_sample_rate)
        self.org_sampler = T.Resample(tokenizer_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')
        else:
            print(f'Model with {count_all_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        Resample the audio according to the condition.
        condition: "tokenizer" to set the sampling rate to the tokenizer's rate
                   "org" to set the sampling rate back to the original rate
        '''
        device = x.device

        assert len(x.shape) == 3, "Input tensor must have 3 dimensions [Batch, Channels, Time]"
        B, C, T = x.shape
        assert C == 1, "Input tensor must be mono-channel [Batch, 1, Time]"

        if condition == "tokenizer":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        else:
            raise ValueError("Unknown condition for resampling: {}".format(condition))
        
        x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be a torch tensor with dimensions [Batch, Channel, Time]
        '''
        original_length = x.shape[-1]

        # Resample the audio to the tokenizer's sample rate
        x = self.resample_audio(x, "tokenizer")
    
        # Remove channel dimensions for the audio data tensor
        x = x.squeeze(1)
        
        # Generate features and discrete codes
        bandwidth_id = torch.tensor([0]).to(x.device)
        features, _, _ = self.model.feature_extractor(x, bandwidth_id=bandwidth_id)
        return features, original_length
    
    def get_quantized_features(self, x, bandwidth_id=None):
        '''
        Expects input [B, D, T] where D is the encoded continuous representation of input.
        Returns quantized features, codes, latents, commitment loss, and codebook loss in the same format as DACWrapper.
        '''
        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0]).to(x.device)

        # Ensure the tensor has 3 dimensions [Batch, Channels, Time]
        if x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions [Batch, Channels, Time], but got {x.ndim} dimensions.")

        # Perform the quantization directly on the encoded features
        q_res = self.model.feature_extractor.encodec.quantizer(
            x, 
            frame_rate=self.model.feature_extractor.frame_rate, 
            bandwidth=self.model.feature_extractor.bandwidths[bandwidth_id]
        )

        # Extract necessary outputs
        quantized = q_res.quantized
        codes = q_res.codes
        latents = x  # The input x itself is the latent representation after encoding
        commit_loss = q_res.penalty

        # Placeholder for codebook_loss (not directly available, could be None)
        codebook_loss = None

        # Return the outputs in the expected format
        return quantized, codes, latents, commit_loss, codebook_loss

    def get_decoded_signal(self, features, original_length):
        '''
        Decodes the features back to the audio signal.
        '''
        # Decode the features to get the waveform
        bandwidth_id = torch.tensor([0]).to(features.device)

        x = self.model.backbone(features, bandwidth_id=bandwidth_id)
        y_hat = self.model.head(x)

        # Ensure the output has three dimensions [Batch, Channels, Time] before resampling
        if y_hat.ndim == 2:
            y_hat = y_hat.unsqueeze(1)  # Add a channel dimension if it's missing

        # Resample the decoded signal back to the original sampling rate
        y_hat_resampled = self.resample_audio(y_hat, "org")

        # Ensure the output shape matches the original length
        if y_hat_resampled.shape[-1] != original_length:
            T_origin = original_length
            T_est = y_hat_resampled.shape[-1]

            if T_origin > T_est:
                y_hat_resampled = F.pad(y_hat_resampled, (0, T_origin - T_est))
            else:
                y_hat_resampled = y_hat_resampled[:, :, :T_origin]

        return y_hat_resampled
