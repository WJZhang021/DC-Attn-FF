import torch
import pickle
import os
from whisper_files.__init__ import download_model

def del_prefix(oldkey: str):
    newkey = oldkey[8:]
    return newkey

model_name = "large-v2" #"large-v3","medium","small"
model_dir =  '/data0/public/usr/zhangweijia/models/whisper'

whole_dir = model_dir+'/'+model_name+'.pt'
print(whole_dir)

# download pt file
#download_model(name=model_name, download_root=model_dir)
model_weights = torch.load(whole_dir)

# check what's in the file
#print(model_weights.keys())

# check the parameters of the model
#print(model_weights['dims'])

# transfer py file of the whole Whisper model to only its encoder
keys = model_weights['model_state_dict'].keys()
for key in list(keys):
    if key[0:7] != 'encoder':
        del model_weights['model_state_dict'][key]        
#print(model_weights['model_state_dict'].keys())

para_list = ['n_mels','n_audio_ctx','n_audio_state','n_audio_head','n_audio_layer']
keys = model_weights['dims'].keys()
for key in list(keys):
    if key not in para_list:
        del model_weights['dims'][key]
#print(model_weights['dims'])        

new_dict = {del_prefix(k):v  for k,v in model_weights['model_state_dict'].items()}
new_pt = {'dims': model_weights['dims'], 'model_state_dict': new_dict}
#print(new_pt['model_state_dict'].keys())
print(new_pt['dims']) 
#print(new_pt['model_state_dict']['positional_embedding'].shape)

new_dir = model_dir+'/'+model_name+'_encoder.pt'
print(new_dir)
#torch.save(new_pt, new_dir)

    
    


