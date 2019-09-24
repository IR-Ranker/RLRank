#encoding:utf-8
import sys
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import random
random.seed(2018)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirection=False, choice='GRU'):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.num_directions = 2 if bidirection else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=n_layers, bidirectional=bidirection) 

    def forward_GRU(self, state, hidden):
        if self.num_directions==2:
            hidden = torch.cat([hidden, hidden], dim=0)
        output, hidden = self.rnn(state, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers*self.num_directions, batch_size, self.hidden_size), requires_grad=True)
        return result.cuda() if use_cuda else result


class CCS(nn.Module):
    def __init__(self):
        super(CCS, self).__init__()   
        self.BG = RNN(fearture_dim, fearture_dim, bidirection=True)
        self.SequenceModel = RNN(rnn_level2_input_dim*3, rnn_level2_hidden_dim, bidirection=False)
        self.Proj_sess = torch.nn.Linear(fearture_dim*2, rnn_level2_input_dim)
        self.Proj_result = torch.nn.Linear(fearture_dim, rnn_level2_input_dim)
        self.click_decoder = torch.nn.Linear(rnn_level2_hidden_dim, 1)
        self.click_encoder = nn.Embedding(3, rnn_level2_input_dim)

    def sample(self, click_probility):
        notclick_probility = 1.0-click_probility
        weights = torch.cat([notclick_probility, click_probility], 1)
        clicks = torch.multinomial(weights, 1, replacement=True)
        return clicks

    def step_by_step(self, sess_features, result_features, batch_size):
        encoder_hidden = self.SequenceModel.initHidden(batch_size)
        outputs_prob, outputs_click = [], []
        previous_clicks = Variable(torch.full((batch_size, result_num+1), 2), requires_grad=True).long().cuda()
        for i in range(result_num):
            click_feature = self.click_encoder(previous_clicks[:,i])
            feature_input = torch.cat([sess_features[i], result_features[i], click_feature], -1)
            encoder_output, encoder_hidden = self.SequenceModel.forward_GRU(
                torch.unsqueeze(feature_input, 0), encoder_hidden)
            click_now_prob = torch.sigmoid(torch.squeeze(self.click_decoder(encoder_hidden), 0))
            click_now = self.sample(click_now_prob)
            previous_clicks[:,i+1] = torch.squeeze(click_now.long(), 1)
            outputs_prob.append(click_now_prob)
            outputs_click.append(click_now)
        outputs_prob = torch.cat(outputs_prob, 1)
        outputs_click = torch.cat(outputs_click, 1) 
        return outputs_prob, outputs_click.float()

    def ClickPredictionWithLabel(self, result_features, query_feature,  clicks, batch_size):
        #First Level RNN : Session Feature
        output, hidden = self.BG.forward_GRU(result_features, query_feature)    
        sess_feature = torch.squeeze(output[-1], 0)
        sess_feature_proj = self.Proj_sess(sess_feature).expand(result_num, -1, -1)
        
        #Result Feature
        result_feature_proj = self.Proj_result(result_features)
        
        #Click Feature
        previous_clicks = clicks[:, :-1] 
        padding = Variable(torch.ones(clicks.size(0), 1)*2, requires_grad=True).long()
        padding = padding.cuda() if use_cuda else padding
        previous_clicks = torch.cat([padding, previous_clicks], 1)
        click_feature_proj = torch.transpose(self.click_encoder(previous_clicks), 0, 1)
        
        #Concat Input Features
        feature_input = torch.cat([sess_feature_proj, result_feature_proj, click_feature_proj], -1)
        
        #Second Level RNN : Click Prediction
        hidden = self.SequenceModel.initHidden(batch_size)
        click_output, click_hidden = self.SequenceModel.forward_GRU(feature_input, hidden)
        click_prediction = self.click_decoder(click_output)
        click_prediction = torch.sigmoid(torch.transpose(torch.squeeze(click_prediction), 0, 1))
        return click_prediction

    def ClickPrediction(self, result_features, query_feature, batch_size):
        #First Level RNN : Session Feature
        output, hidden = self.BG.forward_GRU(result_features, query_feature) 
        sess_feature = output[-1] 
        sess_feature_proj = self.Proj_sess(sess_feature).expand(result_num, -1, -1) 
        
        #Result Feature
        result_feature_proj = self.Proj_result(result_features) 
                
        #Second Level RNN : Click Prediction
        outputs_prob, outputs_click = self.step_by_step(sess_feature_proj, result_feature_proj, batch_size)
        return outputs_prob, outputs_click
