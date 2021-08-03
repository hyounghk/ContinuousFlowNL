import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import *
from model.qg_flow import seqFlow  
from model.tvqa_abc import ABC
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions
from get_tokens import *

import numpy as np

from model.flow.flowv3 import Glow


def line_to_words(line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them
        words = [w for w in words if w != ","]
        words = ["<sos>"] + words + [eos_word] if eos else words
        return words
    
    
def validate(opt, dset, model, flow, temp_flow, mode="valid", cos=None):

    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=100, shuffle=False, collate_fn=pad_collate)
  
    beam = BeamSearcher()
            
#     preds = [open('generated/temp07.txt', "w")]
#     gts = open('generated/var_gts.txt', "w")
    for k, batch in enumerate(tqdm(valid_loader)):
        model_inputs, labels, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                       device=opt.device)

        model_inputs += [flow, temp_flow]
        model.get_conditional(*model_inputs)
        log_p, logdet, flow_z, inputs = model.encode(flow, temp_flow)     

        show_size = 3
        model.decode(flow, temp_flow, inputs, flow_z, True, show_size, cos)   

        sentenses = get_sentences(model, model.logits.view(-1, model.logits.size(-1)), dset, model.max_len, show_size, early_stop=False, show_all=True)

        print(sentenses)
        break
        


def interpolate(opt, dset, model, flow, temp_flow, mode="valid", cos=None):

    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=100, shuffle=False, collate_fn=pad_collate)
    with open('qg_results/gts.txt', 'r') as f:
        all_text = f.readlines()
        all_text = [temp.strip() for temp in all_text if 'where' in temp]
    
#     beam = BeamSearcher()           
#     sentences = ['where did sheldon and beverley go after they came up stairs ?',
#                  'where was rachel when joey said , no guys around , huh ?']

    
    word2idx = load_pickle('cache/word2idx.pickle')
    for i in range(1000):
        sentences = np.random.choice(all_text, 2, replace=False)
        sentence_indices = [[word2idx[w] if w in word2idx else word2idx["<unk>"]
                                for w in line_to_words(sentence, eos=True)] for sentence in sentences]
        maxsize = max(len(sentence_indices[0]), len(sentence_indices[1]))
        sentence_indices[0] += [0 for _ in range(maxsize-len(sentence_indices[0]))]
        sentence_indices[1] += [0 for _ in range(maxsize-len(sentence_indices[1]))]
        sentence_indices = torch.tensor(sentence_indices).cuda()
    #     print(sentence_indices)
        model_inputs = [flow, temp_flow] + [sentence_indices]
        model.get_conditional(*model_inputs)
        log_p, logdet, flow_z, inputs = model.encode(flow, temp_flow)     

        new_flow_z = []
        ratios = np.linspace(0.4, 0.6, 3)
        print(sentences[1])
        for ratio in ratios:
            for z in flow_z:
                new_flow_z += [ratio*z[0:1]+(1.0-ratio)*z[1:2]]
            show_size = 3
            model.decode(flow, temp_flow, inputs, new_flow_z, False, show_size, cos)   

            sentenses = get_sentences(model, model.logits.view(-1, model.logits.size(-1)), dset, model.max_len, show_size, early_stop=False, show_all=True)

            print(sentenses[0][6:])

        print(sentences[0])
        print()
#     input()
    # def extrapolate(opt, dset, model, flow, temp_flow, mode="valid", cos=None):

#     dset.set_mode(mode)
#     torch.set_grad_enabled(False)
#     model.eval()
#     valid_loader = DataLoader(dset, batch_size=100, shuffle=False, collate_fn=pad_collate)
  
#     beam = BeamSearcher()           
#     sentences = ['who does house say they should have listened to when foreman is talking about the cat ?', 'what holiday themed decoration is behind penny sitting on her dishes shelf when she is talking to leonard ?']
# #     sentences = ['what did house say to sam when she was walking out the door ?', 'what did rachel do before chandler said something was n\'t true ?']
# # what did house do to sam when she was walking out the door ?
#     word2idx = load_pickle('cache/word2idx.pickle')
#     sentence_indices = [[word2idx[w] if w in word2idx else word2idx["<unk>"]
#                             for w in line_to_words(sentence, eos=True)] for sentence in sentences]
#     maxsize = max(len(sentence_indices[0]), len(sentence_indices[1]))
#     sentence_indices[0] += [0 for _ in range(maxsize-len(sentence_indices[0]))]
#     sentence_indices[1] += [0 for _ in range(maxsize-len(sentence_indices[1]))]
#     sentence_indices = torch.tensor(sentence_indices).cuda()
# #     print(sentence_indices)
#     model_inputs = [flow, temp_flow] + [sentence_indices]
#     model.get_conditional(*model_inputs)
#     log_p, logdet, flow_z, inputs = model.encode(flow, temp_flow)     

#     new_flow_z = []
#     ratios = np.linspace(1.4, 1.7, 4)
#     for ratio in ratios:
#         for z in flow_z:
# #             new_flow_z += [z[0:1] + torch.randn(z[0:1].size()).cuda()*0.0002]
#             new_flow_z += [z[0:1] + (z[1:2]-z[0:1])*(1+ratio)]
#         show_size = 3
#         model.decode(flow, temp_flow, inputs, new_flow_z, False, show_size, cos)   

#         sentenses = get_sentences(model, model.logits.view(-1, model.logits.size(-1)), dset, model.max_len, show_size, early_stop=False, show_all=True)

#         print('ratio: ', ratio, ' ', sentenses)


if __name__ == "__main__":
#     os.environ['VISIBLE_CUDA_DEVICES'] = opt.gpus[0]
    torch.manual_seed(2019)
    
    opt = BaseOptions().parse()
#     torch.cuda.set_device(opt.gpus[0])
#     torch.cuda.set_device(opt.gpus[0])
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpus[0])
    
    writer = None
    opt.writer = writer

    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)

    model = seqFlow(opt)             
    model.load_embedding(dset.vocab_embedding)   
    model.cuda()
    
    flow = Glow(opt.flow_hidden, opt.flow_l, opt.flow_k, model.max_len, use_transformer=opt.use_transformer, use_recurrent=opt.use_recurrent, use_recurpling=opt.use_recurpling, squeeze_size=opt.squeeze_dim, embedding=model.embedding).cuda()
    cos = nn.CosineSimilarity(dim=2, eps=1e-12)
    dset.return_tokens = True
    if len(opt.gpus) > 1:
        flow = torch.nn.DataParallel(flow)
        temp_flow = flow.module
    else:
        temp_flow = flow
#     try:
    if opt.restore_name:
#         model.load_state_dict(torch.load(opt.results_dir_base+opt.restore_name+'/best_valid.pth', map_location='cuda:0'))
        flow.load_state_dict(torch.load(opt.results_dir_base+opt.restore_name+'/best_valid_flow.pth', map_location='cuda:0'))

    cudnn.benchmark = True

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
    print('The number of parameters of model is', num_params, "M")
    interpolate(opt, dset, model, flow, temp_flow, mode="valid", cos=cos)
    


