import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GermanDataset(data.Dataset):
    def __init__(self, root, captions_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        # pdb.set_trace()
        self.root = root
        self.captions_path = captions_path
        # self.ids = list(self.coco.anns.keys())
        all_captions = [line.rstrip('\n') for line in open(captions_path)]
        self.ids = len(all_captions)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # pdb.set_trace()
        captions_path = self.captions_path
        root = self.root
        vocab = self.vocab

        # ann_id = self.ids[index]
        # caption = coco.anns[ann_id]['caption']
        # img_id = coco.anns[ann_id]['image_id']
        # path = coco.loadImgs(img_id)[0]['file_name']

        all_captions = [line.rstrip('\n') for line in open(captions_path)]

        all_openpose_files = os.listdir(root)




        # all_images = [line.rstrip('\n') for line in open(root)]
        # self.ids = len(all_images)
        vocab_dictionary = {}

        infile = open(vocab, 'r')
        line_num = 1
        for line in infile:
            vocab_dictionary[line_num] = line.strip()
            line_num += 1
        # print(vocab_dictionary)
        infile.close()
        # pdb.set_trace()

        # pdb.set_trace()


        # vocab = [line.rstrip('\n') for line in open(vocab)]
        caption = all_captions[index]
        # image_id = all_images[index]



        # pdb.set_trace()

        # print("image folder is ", image_id)
        # print("caption is ", caption)

        # list_images = os.listdir(image_id)
        # print("number of frames: ", len(list_images))

        image_array = []

        for openpose in all_openpose_files:
           
            image = np.load(os.path.join(root, openpose))
            image = np.reshape(image, (300, 27, 2))
            image = torch.Tensor(image)
            image = image.permute(2, 0, 1).contiguous()
            # image = image.cpu()
            image = np.asarray(image)
            image = np.reshape(image, (2, 300, 27, 1))

            # pdb.set_trace()
            # image = Image.open(os.path.join(image_id, image)).convert('RGB')

            # if self.transform is not None:
            #     image = self.transform(image)
            #     image = np.asarray(image)
            #     image = image.astype('float32')
            #     image_array.append(image)

        # image_array = np.asarray(image_array)
        # print("new array shape: ", image_array.shape)
        # print(image_array)

        caption = caption.strip()
        caption_tokens = caption.split(" ")
        # print("caption tokens are :", caption_tokens )
        caption_ids = []
        vocab_keys = list(vocab_dictionary.keys())
        vocab_values = list(vocab_dictionary.values())
        # pdb.set_trace()
        caption_ids.append(vocab_keys[vocab_values.index('<s>')])
        # vocab_dictionary.values().index(['<s>'])



        # print("After appending start token ", caption_ids, "\n")
        #
        # caption_ids.extend([vocab_keys[vocab_values.index(token)] for token in caption_tokens])
        # caption_ids.extend([vocab_keys[vocab_values.index(token)] for token in caption_tokens if token in vocab_values else [vocab_keys[vocab_values.index('UNK')]])

        for token in caption_tokens:
            if token in vocab_values:
                caption_ids.extend([vocab_keys[vocab_values.index(token)]])
            else:
                caption_ids.extend([vocab_keys[vocab_values.index('UNK')]])


        #
        # print("After mapping it to tokens ", caption_ids, "\n")
        #
        caption_ids.append(vocab_keys[vocab_values.index('</s>')])
        #
        # print("After appending end token", caption_ids, "\n")
        #
        # target = torch.Tensor(caption_ids)
        #
        # print("complete target", target, "\n")

        # pdb.set_trace()
        # current_caption_matrix = pad_sequences(caption_ids, padding='post', maxlen=30)
        # pdb.set_trace()

        max_captioon_len = 30

        zero_ids = [0] * max_captioon_len


        caption_len = len(caption_ids)

        if caption_len < max_captioon_len:
            caption_ids = caption_ids[:len(caption_ids)] + zero_ids[:(max_captioon_len - caption_len)]
        elif caption_len > max_captioon_len:
            caption_ids = caption_ids[:max_captioon_len]

        image = torch.Tensor(image)
        # image = image.to(device)
        target = torch.Tensor(caption_ids)
        # print("target is ", target)

        return image, target

    def __len__(self):
        return (self.ids)


# class CocoDataset(data.Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#     def __init__(self, root, json, vocab, transform=None):
#         """Set the path for images, captions and vocabulary wrapper.
#
#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             vocab: vocabulary wrapper.
#             transform: image transformer.
#         """
#         pdb.set_trace()
#         self.root = root
#         self.coco = COCO(json)
#         self.ids = list(self.coco.anns.keys())
#         self.vocab = vocab
#         self.transform = transform
#
#     def __getitem__(self, index):
#         """Returns one data pair (image and caption)."""
#         coco = self.coco
#         vocab = self.vocab
#         ann_id = self.ids[index]
#         caption = coco.anns[ann_id]['caption']
#         img_id = coco.anns[ann_id]['image_id']
#         path = coco.loadImgs(img_id)[0]['file_name']
#
#         # pdb.set_trace()
#
#         print("image is ", path)
#         print("caption is ", caption)
#
#         image = Image.open(os.path.join(self.root, path)).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#
#         # pdb.set_trace()
#
#         # Convert caption (string) to word ids.
#         tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#         caption = []
#         caption.append(vocab('<start>'))
#
#         print("After appending start token ", caption, "\n")
#
#         caption.extend([vocab(token) for token in tokens])
#
#         print("After mapping it to tokens ", caption, "\n")
#
#         caption.append(vocab('<end>'))
#
#         print("After appending end token", caption, "\n")
#
#         target = torch.Tensor(caption)
#
#         print("complete target", target, "\n")
#
#         # pdb.set_trace()
#
#         return image, target
#
#     def __len__(self):
#         return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # pdb.set_trace()
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # pdb.set_trace()
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # pdb.set_trace()
    # images = torch.from_numpy(images)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    # pdb.set_trace()
    # print("inside colate_fn images", images, "\n")
    # print("inside colate_fn targets", targets, "\n")
    # print("inside colate_fn lengths", lengths, "\n")
    # lengths = lengths.cpu()
    return images, targets, lengths

def get_loader(root, captions_path, vocab, transform, batch_size, shuffle, num_workers):



    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    # pdb.set_trace()
    germandata = GermanDataset(root=root,
                         captions_path=captions_path,
                         vocab=vocab,
                         transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=germandata,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    # for data_check in data_loader:
    #     pdb.set_trace()
    #     print(data_check.drop_last)
    all_vocab = [line.rstrip('\n') for line in open(vocab)]
    # pdb.set_trace()
    return data_loader, len(all_vocab)