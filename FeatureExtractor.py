#encoding:utf-8
import sys
import os
import shutil
import exceptions
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageFile
from config import *
import random
random.seed(2018)

'''
The Recurrent Neural Network module.
'''
class TermEncoder(nn.Module):
    def __init__(self, voca_size, input_size, hidden_size, RNN_choice, n_layers=1, bidirection=False, embed=True):
        super(TermEncoder, self).__init__()
        self.n_layers = n_layers
        self.num_directions = 2 if bidirection else 1
        self.hidden_size = hidden_size
        self.embed = embed
        if self.embed:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(load_voca()))
        if RNN_choice=='LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layers, bidirectional=bidirection)
        elif RNN_choice=='GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=n_layers, bidirectional=bidirection)
    
    def forward_LSTM(self, input, hidden, cell, embed=True):
        if self.embed:
            input = self.embedding(input).unsqueeze(1)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        return output, hidden, cell
    
    def forward_GRU(self, input, hidden, embed=True):
        if self.embed:
            input = self.embedding(input).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers*self.num_directions, 1, self.hidden_size))
        return result.cuda() if use_cuda else result
    
    def initCell(self):
        result = Variable(torch.zeros(self.n_layers*self.num_directions, 1, self.hidden_size))
        return result.cuda() if use_cuda else result


'''
The implementation of the feature representaion module.
'''
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.dropout = 0.0
        self.Weights = nn.Parameter(torch.FloatTensor(len(tag_dict), fearture_dim, fearture_dim).uniform_()*0.02-0.01, requires_grad=True)
        self.Bias = nn.Parameter(torch.FloatTensor(len(tag_dict), 1, fearture_dim).uniform_()*0.02-0.01, requires_grad=True)
        self.RNN_choice = 'GRU'
        self.BiRNN = TermEncoder(voca_size, embedding_dim, hidden_size_2, self.RNN_choice, bidirection=False)
        self.CNN = models.alexnet(num_classes = fearture_dim, pretrained=True)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(fearture_dim, fearture_dim),
            torch.nn.Dropout(self.dropout),
            torch.nn.PReLU(),
            torch.nn.Linear(fearture_dim, 512),
            torch.nn.Dropout(self.dropout),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
        self.intent = torch.nn.Linear(fearture_dim, fearture_dim)
        self.act = nn.PReLU()
    
    def read_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.transpose(np.array(img,dtype='float32'), (2, 0, 1))
        img = torch.from_numpy(img)
        return img
    
    def RNN(self, terms):
        if self.RNN_choice == 'GRU':
            encoder_hidden = self.BiRNN.initHidden()
            encoder_output, encoder_hidden = self.BiRNN.forward_GRU(terms, encoder_hidden)
        elif self.RNN_choice == 'LSTM':
            encoder_hidden = self.BiRNN.initHidden()
            encoder_cell = self.BiRNN.initCell()
            encoder_output, encoder_hidden, encoder_cell = self.BiRNN.forward_LSTM(terms, encoder_hidden, encoder_cell)
        return encoder_output 
    
    def iter_tree(self, node, query_feature):
        if node.tag == 'txt':
            text = [int(item) for item in node.attrib['cut_id'].split(' ')]
            text_color = node.attrib['color']
            text_input = Variable(torch.LongTensor(text))
            text_input = text_input.cuda() if use_cuda else text_input
            text_feature = self.RNN(text_input)[-1]
            text_feature = F.normalize(text_feature, p=2, dim=1, eps=1e-12)
            return text_feature, node.tag
        elif node.tag == 'img':
            img_path = img_dir + node.attrib['path']
            try:
                img = self.read_img(img_path)
                img_input = Variable(torch.unsqueeze(img, 0))
                img_input = img_input.cuda() if use_cuda else img_input
                img_feature = self.CNN(img_input)
                img_feature = F.normalize(img_feature, p=2, dim=1, eps=1e-12)
            except:
                img_feature = Variable(torch.zeros(1,fearture_dim))
                img_feature = img_feature.cuda() if use_cuda else img_feature
            return img_feature, node.tag
        else:
            features, attentions = [], []
            for child in node:
                feature, nodeTag = self.iter_tree(child, query_feature)
                attention = F.cosine_similarity(feature, query_feature, dim=1)      
                w = self.Weights[tag_dict[nodeTag]]
                b = self.Bias[tag_dict[nodeTag]]            
                new_feature = self.act(torch.addmm(b, feature, w))
                features.append(new_feature)
                attentions.append(attention)
            features = torch.transpose(torch.cat(features, dim=0), 0, 1)
            attentions = F.softmax(torch.cat(attentions, dim=0))
            fused_feature = torch.mv(features, attentions).view(1,-1)
            return fused_feature, node.tag
    
    def forward(self, root, query_feature):
        result_feature, nodeTag = self.iter_tree(root, query_feature)
        return result_feature


