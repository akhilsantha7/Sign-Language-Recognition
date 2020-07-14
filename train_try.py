import argparse
import torch
# torch.set_default_tensor_type(torch.cuda.FloatTensor) #Cos of this error:Expected tensor to have CPU Backend, but got tensor with CUDA Backend (while checking arguments for batch_norm_cpu)
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_custom import get_loader
from build_vocab import Vocabulary
from model_custom import Model, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(args):
    # Create model directory
    # pdb.set_trace()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    # with open(args.vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)
    #     pdb.set_trace()
    
    # Build data loader
    # data_loader, vocab_size = get_loader(args.image_dir, args.caption_path, args.vocab_path,
    #                          transform, args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers)
    data_loader, vocab_size = get_loader(args.openpose_dir, args.caption_path, args.vocab_path,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    # encoder = Model(args.embed_size, ).to(device)
    encoder = Model(2, edge_importance_weighting=True, graph_args={'layout':'points_27', 'strategy': 'spatial'}).to(device)
    decoder = DecoderRNN(1024, args.hidden_size, vocab_size, args.num_layers).to(device)
    # pdb.set_trace()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) +  list(encoder.data_bn.parameters()) #list(encoder.linear.parameters()) +
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        data_iterator = iter(data_loader)
        for i, (images, captions, lengths) in enumerate(data_loader):
            # pdb.set_trace()
            # Set mini-batch dataset
            images = images.to(device)
            # print('iter is ', i)
            # print(images)
            # pdb.set_trace()
            # pdb.set_trace()
            captions = captions.to(device)
            # pdb.set_trace()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # pdb.set_trace()
            
            # Forward, backward and optimize
            features = encoder(images)
            # pdb.set_trace()
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                # pdb.set_trace()
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/image_captioning/out_dir/' , help='path for saving trained models')
    parser.add_argument('--model_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_captioning/out_dir2' , help='path for saving trained models')

    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')

    parser.add_argument('--vocab_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.vocab.de', help='path for vocabulary wrapper')
    
    # parser.add_argument('--image_dir', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_10.sign', help='directory for resized images')
    parser.add_argument('--image_dir', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.train_re_10.sign', help='directory for resized images')

    parser.add_argument('--openpose_dir', type=str,
                        default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/openpose-10/',
                        help='directory for resized images')
    
    # parser.add_argument('--caption_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_10.de', help='path for train annotation json file')
    parser.add_argument('--caption_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.train_re_10.de', help='path for train annotation json file')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models') # was 1000

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2) # was 128
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)