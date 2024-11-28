import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .whisper_files.model import AudioEncoder,LayerNorm,Linear,Conv1d,sinusoids,MultiHeadAttention,ResidualAttentionBlock
from .whisper_files.audio import pad_or_trim,log_mel_spectrogram
from .whisper_files.utils import exact_div
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d,AttentiveStatisticsPooling

def cut_pe(model_state_dict: dict, ctx: int):
    pe = model_state_dict['positional_embedding']
    pe = pe[0:ctx,:]
    model_state_dict['positional_embedding'] = pe
    return model_state_dict

class AudioEncoder_mask(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.n_head = n_head

    def forward(self, x: Tensor, len_ctx: Tensor = None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, N_FRAMES)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)      # (batch_size, n_ctx, n_mels)
        
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        
        if len_ctx is not None:
            B = x.size(0)
            max_len_ctx = int(max(len_ctx))
            x = x[:,:max_len_ctx,:]
            '''
            for index in range(B):
                x[index, int(len_ctx[index]):, :] = 0
            '''
            padding_mask = torch.ones([B, max_len_ctx], dtype=bool)
            for index in range(B):
                padding_mask[index, :int(len_ctx[index])] = 0
            key_padding_mask = padding_mask.view(B, 1, 1, max_len_ctx). \
                expand(-1, self.n_head, -1, -1).reshape(B, self.n_head, 1, max_len_ctx)
            new_padding_mask = torch.zeros_like(key_padding_mask, dtype=x.dtype)
            mask = new_padding_mask.masked_fill(key_padding_mask, float("-inf"))\
                .to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
            
        for block in self.blocks:
            #x = block(x, mask=mask)
            x = block(x)

        x = self.ln_post(x)
        return x

class WhisperEncoder(nn.Module):
    def __init__(
        self,
        loss: nn.Module,
        embedding_dim: int,
        sec: float,
        pooling_head: str,
        ckpt: str = '/data0/public/usr/zhangweijia/models/whisper/small_encoder.pt',
    ) -> None:
    
        super().__init__()
        
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        ckpt_dims = ckpt_dict['dims']
        self.n_mels = ckpt_dims['n_mels']
        # self.n_ctx = ckpt_dims['n_audio_ctx']
        self.n_state = ckpt_dims['n_audio_state']
        self.n_head = ckpt_dims['n_audio_head']
        self.n_layer = ckpt_dims['n_audio_layer']
        
        self.sec = sec
        self.n_frame = int( exact_div(sec*16000, 160) )
        self.n_ctx = int( (self.n_frame-1) // 2 + 1 )
        
        self.loss = loss
        self.embedding_dim = embedding_dim
        '''
        note that SAMPLE_RATE = 16000, N_SAMPLES = sec * SAMPLE_RATE, HOP_LENGTH = 160
        n_frame = exact_div(sec*16000, 160) <- better to be even
        After CONV: n_ctx = (n_frame-1) // 2 + 1 ~= sec * 50
        '''
        self.encoder = AudioEncoder_mask(self.n_mels, self.n_ctx, self.n_state, self.n_head, self.n_layer)        
        self.encoder.load_state_dict( cut_pe(ckpt_dict['model_state_dict'],self.n_ctx) )

        self.pooling_head = pooling_head
        if self.pooling_head == 'ASP':
            self.pooling = AttentiveStatisticsPooling(self.n_state,
                                                         attention_channels=embedding_dim//2)
            self.fc = nn.Linear(self.n_state * 2, embedding_dim)        
            self.ln = LayerNorm(self.embedding_dim)
        else:       
            self.conv = nn.Conv1d(self.n_state, self.embedding_dim, 8,4)           
            self.pooling = nn.MaxPool1d(4,4)
            self.n_ctx_pool = int( (self.n_ctx-8) // 4 + 1 )    # maxium
            self.n_ctx_pool = int( (self.n_ctx_pool-4) // 4 + 1 )    # maxium
            # would need pad
            self.fc = nn.Linear(self.n_ctx_pool,1)
            self.bn = nn.BatchNorm1d(self.embedding_dim)       
        
    def process_audio(
        self,
        waveform: torch.Tensor,
        len: torch.Tensor = None
    ) -> torch.Tensor:
        '''
        waveform: [B, T]
        Returns:  torch.Tensor, shape = (batch_size, n_mels, N_FRAMES)
        '''
        B = waveform.shape[0]
        spectrogram = torch.zeros((B,self.n_mels,self.n_frame)).to(waveform.device,waveform.dtype)       
                
        for i in range(B):
            audio = waveform[i]
            # audio = pad_or_trim(audio)
            spectrogram[i] = log_mel_spectrogram(audio,self.n_mels)
        
        if len is None:
            len_ctx = None
        else:
            '''
            len_ctx = torch.zeros((B,1)).to(waveform.device,int)
            for i in range(B):
                audio_len = len[i].item()
                mel_len = audio_len // 160
                len_ctx[i] = (mel_len-1) // 2 + 1
            '''
            len_ctx = ( (len//160 - 1) // 2 + 1 ).to(waveform.device,torch.int)

        return spectrogram, len_ctx

    def embedding(
        self,
        x: torch.Tensor,
        len: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x:  [B, T]
        
        for AudioEncoder's input:
            torch.Tensor, shape = (batch_size, n_mels, N_FRAMES)
            the mel spectrogram of the audio
        """
        x, len_ctx = self.process_audio(x, len) #(batch_size, n_mels, N_FRAMES)
        #x = x.permute(0,2,1)
               
        x = self.encoder(x, len_ctx)     # (batch_size, n_ctx?, n_state)

        if self.pooling_head == 'ASP':
            x = self.pooling(x.transpose(1, 2)).squeeze(2)  # (batch_size, n_state*2)
            x = self.ln(self.fc(x))
        else:
            x = self.pooling(self.conv(x.transpose(1, 2)))  # (batch_size, embedding_dim, n_ctx_pool?)
            pad_len = self.n_ctx_pool-x.shape[2]
            x = F.pad(x,[0,pad_len], "constant", 0)
            x = self.fc(x).squeeze(2)
            x = self.bn(x)           
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor = None,  
        out_emb: bool = False,
        len: torch.Tensor = None            
    ) -> Dict[str, Any]:
        
        x = self.embedding(x,len)        
        if out_emb:
            output_dict = {'embedding': x}
        else:
            output_dict = self.loss(x, label)
        return output_dict
        