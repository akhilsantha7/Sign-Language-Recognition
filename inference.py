import argparse
import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor) #Cos of this error:Expected tensor to have CPU Backend, but got tensor with CUDA Backend (while checking arguments for batch_norm_cpu)
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model_custom import Model, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    score =0
    # Image preprocessing, normalization for the pretrained resnet
    references = list()
    hypotheses = list()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader, vocab_size = get_loader(args.openpose_dir, args.caption_path, args.vocab_path,
                                         transform, args.batch_size,
                                         shuffle=True, num_workers=args.num_workers)


    encoder = Model(2, edge_importance_weighting=True, graph_args={'layout':'new_openpose', 'strategy': 'spatial'}).eval()
    # encoder = Model(2, edge_importance_weighting=True, graph_args={'layout':'new_openpose', 'strategy': 'spatial'}).to(device)
    # decoder = DecoderRNN(1024, args.hidden_size, vocab_size, args.num_layers).to(device)
    decoder = DecoderRNN(1024, args.hidden_size, vocab_size, args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    enc_chkpt = torch.load(args.encoder_path)
    encoder.load_state_dict(enc_chkpt['encoder_state_dict'])
    # encoder_ckpt = torch.load(args.model_path)
    epoch_ckpt = enc_chkpt['epoch']
    dec_chkpt = torch.load(args.decoder_path)
    decoder.load_state_dict(dec_chkpt['decoder_state_dict'])

    data_iterator = iter(data_loader)

    vocab_dictionary = {}

    infile = open(args.vocab_path.strip(), 'r')
    line_num = 1
    for line in infile:
        vocab_dictionary[line_num] = line.strip()
        line_num += 1
    # print(vocab_dictionary)
    infile.close()
    # pdb.set_trace()
    for i, (images, captions, file_name, lengths) in enumerate(data_loader):
        # pdb.set_trace()
        # Set mini-batch dataset
        images = images.to(device)
        images[images == -999] = 0.0
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        targets = targets.tolist()
        gt_caption = ""
        for gt in targets:
            # pdb.set_trace()
            if gt != 0:
                gt_caption = gt_caption + vocab_dictionary[gt] + " "
        # pdb.set_trace()

        # Forward, backward and optimizel
        features = encoder(images)

        outputs = decoder.sample(features)
        out_list = outputs.tolist()[0]
        out_caption = ""
        for out in out_list:
            # pdb.set_trace()
            if out != 0:
                out_caption = out_caption + vocab_dictionary[out] + " "
        # pdb.set_trace()
        print(" GT caption: ", gt_caption)
        print(" OUT caption: ", out_caption)

        gt_1 = gt_caption.replace('<s>', '')
        gt_2 = gt_1.replace('</s>', '')
        out_1 = out_caption.replace('<s>','')
        out_2 = out_1.replace('</s>','')
        gt_list = gt_2.split()
        out_list = out_2.split()
        print(gt_list,out_list)
        


        score = score +sentence_bleu([gt_list], out_list)
        print(score)
    print(score/100)
############## bleu score #################
        # img_caps = gt_caption.tolist()
    #     img_captions = list(
    #         map(lambda c: [w for w in c if w not in {word_map['<s>'], word_map['</s>']}],
    #             gt_caption))
    #     references.append(img_captions)
    #     hypotheses.append([w for w in out_caption if w not in {word_map['<s>'], word_map['</s>']}])
    #     assert len(references) == len(hypotheses)
    # bleu4 = corpus_bleu(references, hypotheses)


    
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpose_dir', type=str,
                        default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/openpose',
                        help='directory for resized images')
    # parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_st-gcn/out_dir2/encoder-681-12.ckpt',
                        help='path for trained encoder')
    # /home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/image_captioning/out_dir/encoder-10-5.ckpt, /shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_captioning/out_dir_org/encoder-100-50.ckpt
    parser.add_argument('--decoder_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_st-gcn/out_dir2/decoder-681-12.ckpt',
                        help='path for trained decoder')
    # /shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_captioning/out_dir_org/decoder-100-50.ckpt
    parser.add_argument('--vocab_path', type=str,
                        default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.vocab.de',
                        help='path for vocabulary wrapper')
    parser.add_argument('--caption_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.train_re_100.de', help='path for train annotation json file')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=2)  # was 128
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--inference', type=str, default=True)
    args = parser.parse_args()
    main(args)