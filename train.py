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
    encoder = Model(2, edge_importance_weighting=True, graph_args={'layout':'new_openpose', 'strategy': 'spatial'}).to(device)
    decoder = DecoderRNN(1024, args.hidden_size, vocab_size, args.num_layers).to(device)

    if args.continue_from_ckpt == True:
        # pdb.set_trace()
        enc_chkpt = torch.load(args.encoder_path)
        encoder.load_state_dict(enc_chkpt['encoder_state_dict'])
        # encoder_ckpt = torch.load(args.model_path)
        epoch_ckpt = enc_chkpt['epoch']
        dec_chkpt = torch.load(args.decoder_path)
        decoder.load_state_dict(dec_chkpt['decoder_state_dict'])


    # pdb.set_trace()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) +  list(encoder.data_bn.parameters()) #list(encoder.linear.parameters()) +
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)

    vocab_dictionary = {}

    infile = open(args.vocab_path.strip(), 'r')
    line_num = 1
    for line in infile:
        vocab_dictionary[line_num] = line.strip()
        line_num += 1
    # print(vocab_dictionary)
    infile.close()

    if args.continue_from_ckpt:
        epoch_global = epoch_ckpt
    else:
        epoch_global = 0

    for epoch in range(epoch_global, args.num_epochs):
        # pdb.set_trace()
        data_iterator = iter(data_loader)
        for i, (images, captions, file_name, lengths) in enumerate(data_loader):
            # pdb.set_trace()
            # Set mini-batch dataset
            images = images.to(device)
            images[images == -999] = 0.0
            # pdb.set_trace()
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
           
            targets_list = targets.tolist()
            gt_caption = ""
            for gt in targets_list:
                # pdb.set_trace()
                if gt != 0:
                    gt_caption = gt_caption + vocab_dictionary[gt] + " "

            # pdb.set_trace()
            # Forward, backward and optimize
            features = encoder(images)
            # pdb.set_trace()
            outputs = decoder(features, captions, lengths)
            # out_list = outputs.tolist()[0]
            # out_caption = ""
            # for out in out_list:
            #     # pdb.set_trace()
            #     if out != 0:
            #         out_caption = out_caption + vocab_dictionary[out] + " "
            # pdb.set_trace()
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            j = i+ 1
            if np.mod(epoch, 20) == 0:

                # Save the model checkpoints
                if (j) % args.save_step == 0:
                    # pdb.set_trace()
                    # torch.save(decoder.state_dict(), os.path.join(
                    #     args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                    # torch.save(encoder.state_dict(), os.path.join(
                    #     args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                    torch.save({
                        'epoch': epoch + 1,
                        'decoder_state_dict':decoder.state_dict(),
                        'loss': loss
                        }, os.path.join(
                        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, j)))


                    torch.save({
                            'epoch': epoch + 1,
                            'encoder_state_dict':encoder.state_dict(),
                            'loss': loss
                        }, os.path.join(
                            args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, j)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/image_st-gcn/out_dir2/' , help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')

    # parser.add_argument('--vocab_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.vocab.de', help='path for vocabulary wrapper')
    # parser.add_argument('--image_dir', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_100.sign', help='directory for resized images')
    # parser.add_argument('--openpose_dir', type=str,
    #                     default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/openpose',
    #                     help='directory for resized images') # /shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/openpose, /home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/openpose
    # parser.add_argument('--caption_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_100.de', help='path for train annotation json file')

    # parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    # parser.add_argument('--save_step', type=int , default=50, help='step size for saving trained models') # was 1000

    # # Model parameters
    # parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    # parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    # parser.add_argument('--num_epochs', type=int, default=2001)
    # parser.add_argument('--batch_size', type=int, default=1) # was 128
    # parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--learning_rate', type=float, default=0.0001) #was 0.001
    # parser.add_argument('--inference', type=str, default=False)
    parser.add_argument('--continue_from_ckpt', type=str, default=False)
    parser.add_argument('--encoder_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/image_captioning/out_dir_100/encoder-1001-50.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/image_captioning/out_dir_100/decoder-1001-50.ckpt',
                        help='path for trained decoder')
    # args = parser.parse_args()
    # print(args)
    # main(args)
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')

    parser.add_argument('--vocab_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.vocab.de', help='path for vocabulary wrapper')
    
    # parser.add_argument('--image_dir', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_10.sign', help='directory for resized images')
    parser.add_argument('--image_dir', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.train_re_100.sign', help='directory for resized images')

    parser.add_argument('--openpose_dir', type=str,
                        default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/openpose',
                        help='directory for resized images')
    
    # parser.add_argument('--caption_path', type=str, default='/home/ta2184/sign_language_review_paper/image_captioning_example_pytorch/Data/phoenix2014T.train_re_10.de', help='path for train annotation json file')
    parser.add_argument('--caption_path', type=str, default='/shared/kgcoe-research/mil/phonenix_dataset/Akhil/stgcn_LSTM/st_gcn/Data/phoenix2014T.train_re_100.de', help='path for train annotation json file')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1, help='step size for saving trained models') # was 1000

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2) # was 128
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)