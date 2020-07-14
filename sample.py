import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import pdb
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from model_custom_test import Model, DecoderRNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def load_image(image_path, transform=None):
#     image = Image.open(image_path)
#     image = image.resize([224, 224], Image.LANCZOS)
    
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
    
#     return image

def load_image(image_path, transform=None):
    image = np.load(image_path)
    image = np.reshape(image, (300, 25, 2))
    image = torch.Tensor(image)

    image = image.permute(2, 0, 1).contiguous()
            # image = image.cpu()
    image = np.asarray(image)
    image = np.reshape(image, (2, 300, 25, 1))
    image = np.reshape(image, (1,2, 300, 25, 1))

    return image

def main(args):
    # Image preprocessing
    # transform = transforms.Compose([
    #     transforms.ToTensor(), 
    #     transforms.Normalize((0.485, 0.456, 0.406), 
    #                          (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    # with open('./vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)

    # print(vocab)
    # pdb.set_trace()
    vocab  = args.vocab_path
    all_vocab = [line.rstrip('\n') for line in open(vocab)]
    # Build models
    # encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    encoder = Model(2, edge_importance_weighting=True, graph_args={'layout':'new_openpose', 'strategy': 'spatial'}).eval()

    decoder = DecoderRNN(1024, args.hidden_size, len(all_vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image_tensor = load_image(args.image, None)
    image_tensor = torch.Tensor(image_tensor)
    image_tensor = image_tensor.to(device)

    
    # 
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    pdb.set_trace()
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    pdb.set_trace()
    # Convert word_ids to words
    sampled_caption = []
    # all_vocab = [line.rstrip('\n') for line in open(vocab)]
    for word_id in sampled_ids:
        pdb.set_trace()
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break


###########################
    # vocab_dictionary = {}
    
    # infile = open(vocab, 'r')
    # line_num = 1
    # for line in infile:
    #     vocab_dictionary[line_num] = line.strip()
    #     line_num += 1
    # # print(vocab_dictionary)
    # infile.close()
    # vocab_values = list(vocab_dictionary.values())
    # print(vocab_values)
    # for word_id in sampled_ids:
    #     word = vocab_values.idx2word[word_id]
    #     sampled_caption.append(word)
    #     if word == '<end>':
    #         break
############################
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    # image = Image.open(args.image)
    # plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_captioning/out_dir_org/encoder-100-50.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_captioning/out_dir_org/decoder-100-50.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.vocab.de', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)