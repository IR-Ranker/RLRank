#encoding:utf-8
import sys
import os
import math
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from tensorboardX import SummaryWriter
import random
random.seed(2018)

from CCS import *


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



class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dropout = 0.0
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(fearture_dim*2, fearture_dim),
            torch.nn.Dropout(self.dropout),
            torch.nn.PReLU(),
            torch.nn.Linear(fearture_dim, 512),
            torch.nn.Dropout(self.dropout),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 1),
        )
    
    def policy(self, step, result_features, state, actions, batch_size, mode):
        mask = torch.ge(actions, 0)
        actions_not_choosed = torch.masked_select(actions, mask).reshape(batch_size, -1)

        actions_not_choosed_expand = torch.transpose(torch.unsqueeze(actions_not_choosed, 2).expand(-1, -1, fearture_dim), 0, 1)
        choosed_results = torch.gather(result_features, 0, actions_not_choosed_expand.long())

        actions_num = choosed_results.size(0)
        state_expand = state.expand(actions_num, -1, -1)

        feature_fusion = torch.cat([state_expand, choosed_results], 2)
        action_scores = torch.transpose(torch.squeeze(self.dense(feature_fusion), -1), 0, 1)
        action_scores = F.softmax(action_scores, 1)
        
        if mode == 'random':
            sample = torch.multinomial(action_scores, 1, replacement=False)
        elif mode == 'max':
            sample = torch.max(action_scores, dim=1)[1].unsqueeze(1)

        choosed_result = torch.gather(actions_not_choosed, 1, sample)
        choosed_score = torch.gather(action_scores, 1, sample)

        for batch_ind in range(batch_size):
            actions[batch_ind][choosed_result[batch_ind].long()] = -step

        return choosed_result, choosed_score, actions



class RLRanker(nn.Module):
    def __init__(self):
        super(RLRanker, self).__init__()   
        self.CCS = CCS()
        self.StateRNN = RNN(fearture_dim, fearture_dim)
        self.PolicyNet = PolicyNetwork()

    def RL_Steps_sample_multitimes(self, result_features, query_feature, batch_size_expand, mode):
        actions = torch.zeros(batch_size_expand, result_num, requires_grad=True).cuda()
        for i in range(result_num):
            actions[:,i] = i
        pis = []
        state_order = []
        state = query_feature
        for step in range(1, step_num+1):
            choosed_result, choosed_score, actions = self.PolicyNet.policy(step, result_features, state, actions, batch_size_expand, mode)

            choosed_result_expand = torch.unsqueeze(choosed_result.expand(-1, fearture_dim), 0)
            choosed_result_features = torch.gather(result_features, 0, choosed_result_expand.long())

            output, hidden = self.StateRNN.forward_GRU(choosed_result_features, state)     
            state = hidden

            state_order.append(choosed_result)
            pis.append(choosed_score)
        state_order = torch.cat(state_order, 1)
        pis = torch.cat(pis, 1)
        return state_order, pis

    def ClickSimulator(self, result_features, query_feature, batch_size, T_Simulator):
        batch_size = result_features.size(1)
        clicks_all = []
        for i in range(T_Simulator):
            prob, click = self.NCM.ClickPrediction(result_features, query_feature, batch_size)
            clicks_all.append(click.unsqueeze(0))
        clicks_all = torch.cat(clicks_all, 0).mean(0)
        return clicks_all
    
    def reward_func(self, clicks, choice):
        if choice=='CTR-AC':
            rewards = []
            for i in range(step_num):
                rewards.append(torch.mean(clicks[:,:i+1], dim=1).unsqueeze(1))
            rewards = torch.cat(rewards, dim=1)
        elif choice=='MRR-AC':
            batch_size = clicks.size(0)
            weight = torch.FloatTensor([1./1,1./2,1./3,1./4,1./5,1./6,1./7,1./8,1./9,1./10]).expand(batch_size,-1).cuda()
            clicks = clicks*weight
            rewards = []
            for i in range(step_num):
                rewards.append((clicks[:,:i+1].sum(1)/weight[:,:i+1].sum(1)).unsqueeze(1))
            rewards = torch.cat(rewards, dim=1)
        elif choice=='RBP-AC':
            batch_size = clicks.size(0)
            p = 0.8
            weight = torch.FloatTensor([pow(p,0),pow(p,1),pow(p,2),pow(p,3),pow(p,4),pow(p,5),pow(p,6),pow(p,7),pow(p,8),pow(p,9)]).expand(batch_size,-1).cuda()
            clicks = clicks*weight
            rewards = []
            for i in range(step_num):
                rewards.append((clicks[:,:i+1].sum(1)/weight[:,:i+1].sum(1)).unsqueeze(1))
            rewards = torch.cat(rewards, dim=1)
        elif choice=='DCG-AC':
            batch_size = clicks.size(0)
            weight = torch.FloatTensor([1./math.log(2,2),1./math.log(3,2),1./math.log(4,2),1./math.log(5,2),1./math.log(6,2),1./math.log(7,2),1./math.log(8,2),1./math.log(9,2),1./math.log(10,2),1./math.log(11,2)]).expand(batch_size,-1).cuda()
            clicks = clicks*weight
            rewards = []
            for i in range(step_num):
                rewards.append((clicks[:,:i+1].sum(1)/weight[:,:i+1].sum(1)).unsqueeze(1))
            rewards = torch.cat(rewards, dim=1)
        return rewards   

    def loss_func(self, rewards, pis, gamma, beta):
        batch_size_temp = rewards.size(0)
        log_pis = pis.log()
        R = torch.zeros(batch_size_temp).cuda()
        loss = torch.zeros(batch_size_temp).cuda()
        for i in reversed(range(step_num)):
            R = gamma * R + rewards[:, i]
            loss = loss - log_pis[:, i]*R
            loss = torch.mean(loss / step_num)
        return loss
