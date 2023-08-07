from abc import ABC
from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class QValueNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(QValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.q_value = nn.Linear(in_features=512, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, states):
        x = states / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)

        # Init layer weights
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, states):
        x = states / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return Categorical(probs), probs + z


############################################################### Discrete DIAYN policy ############################
class Discriminator(nn.Module, ABC): # From state -> n_skills
    def __init__(self, state_shape, n_skills):
        super(Discriminator, self).__init__()
        self.state_shape = state_shape
        self.n_skills = n_skills

        # state_shape must be channels first
        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)

        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)

        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.q = nn.Linear(in_features=512, out_features=self.n_skills)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q.weight)
        self.q.bias.data.zero_()

    def forward(self, state):
        # Restore original state shape
        state = state.view(state.size(0), *self.state_shape) # (batch_size, channels * w * h)

        x = state / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        logits = self.q(x)
        return logits


class PolicyNetwork_DIAYN(nn.Module, ABC):
    def __init__(self, state_shape, n_actions, n_skills):
        super(PolicyNetwork_DIAYN, self).__init__()

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_skills = n_skills

        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size+n_skills, out_features=512)
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)

        # init weights:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()



    def forward(self, augmented_state):

        state = augmented_state[:, :-self.n_skills]
        skill = augmented_state[:, -self.n_skills:]

        # Restore original state shape
        state = state.view(state.size(0), *self.state_shape)

        x = state / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, skill], dim=1)

        x = F.relu(self.fc(x))

        action_logits = self.logits(x)
        dist = Categorical(logits=action_logits)
        return dist


    def sample_or_likelihood(self, augmented_state):
        dist = self(augmented_state)

        u = dist.sample()  # Sample an action
        log_prob = dist.log_prob(value=u)
        return u, log_prob



class ValueNetwork_DIAYN(nn.Module, ABC):  # NN: from state -> CNN_layers -> flatten + n_skills -> 1
    def __init__(self, state_shape, n_skills):
        super(ValueNetwork_DIAYN, self).__init__()
        self.state_shape = state_shape
        self.n_skills = n_skills

        # self.n_hidden_filters = n_hidden_filters
        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size + n_skills, out_features=512)
        self.value = nn.Linear(in_features=512, out_features=1)

        # init weights:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()


        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        # init_weight(self.value, initializer="xavier uniform")
        # self.value.bias.data.zero_()

    def forward(self,  augmented_state):

        state = augmented_state[:, :-self.n_skills]
        skill = augmented_state[:, -self.n_skills:]

        # Restore original state shape
        state = state.view(state.size(0), *self.state_shape)

        x = state / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, skill], dim=1)

        x = F.relu(self.fc(x))

        return self.value(x)


class QNetwork_DIAYN(nn.Module, ABC):  # NN: from state -> CNN_layers -> flatten + n_skills -> n_actions
    def __init__(self, state_shape, n_actions, n_skills):
        super(QNetwork_DIAYN, self).__init__()
        self.state_shape = state_shape
        self.n_skills = n_skills
        self.n_actions = n_actions
        # self.n_hidden_filters = n_hidden_filters
        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size + n_skills, out_features=512)
        self.q = nn.Linear(in_features=512, out_features=self.n_actions)

        # init weights:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q.weight)
        self.q.bias.data.zero_()


        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        # init_weight(self.value, initializer="xavier uniform")
        # self.value.bias.data.zero_()

    def forward(self,  augmented_state):

        state = augmented_state[:, :-self.n_skills]
        skill = augmented_state[:, -self.n_skills:]

        # Restore original state shape
        state = state.view(state.size(0), *self.state_shape)

        x = state / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, skill], dim=1)

        x = F.relu(self.fc(x))

        return self.q(x)
