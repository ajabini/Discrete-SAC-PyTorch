import numpy as np
from Brain.model import PolicyNetwork_DIAYN, QNetwork_DIAYN, ValueNetwork_DIAYN, Discriminator
import torch
from Memory.replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn import functional as F
from torch.nn.functional import log_softmax

def inverse_concat_state_latent(combined, original_shape, n_skills):
    # Split tensors
    state = combined[:, :-n_skills]
    latent = combined[:, -n_skills:]

    # Reshape state if needed
    if len(original_shape) == 3:
        state = state.view(state.size(0), *original_shape)

        # Get index of maximum skill
    latent_idx = latent.argmax(dim=1)

    return state, latent_idx


class SAC: #Discrete
    def __init__(self, **config):
        self.config = config
        self.state_shape = self.config["state_shape"]
        self.n_actions = self.config["n_actions"]
        self.lr = self.config["lr"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.memory = Memory(memory_size=self.config["mem_size"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Diff from continuous case of SAC: Must have n_skills
        self.policy_network = PolicyNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)

        self.q_value_network1 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.q_value_target_network1 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).to(self.device)
        self.q_value_target_network2 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).to(self.device)

        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

        self.entropy_target = 0.98 * (-np.log(1 / self.n_actions))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)
        self.alpha_opt = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

    def store(self, state, action, reward, next_state, done):
        state = from_numpy(state).byte().to("cpu")
        reward = torch.CharTensor([reward])
        action = torch.ByteTensor([action]).to('cpu')
        next_state = from_numpy(next_state).byte().to('cpu')
        done = torch.BoolTensor([done])
        self.memory.add(state, reward, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, *self.state_shape)
        actions = torch.cat(batch.action).view((-1, 1)).long().to(self.device)
        rewards = torch.cat(batch.reward).view((-1, 1)).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, *self.state_shape)
        dones = torch.cat(batch.done).view((-1, 1)).to(self.device)

        return states, rewards, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the Q-Value target
            with torch.no_grad():
                _, next_probs = self.policy_network(next_states)
                next_log_probs = torch.log(next_probs)
                next_q1 = self.q_value_target_network1(next_states)
                next_q2 = self.q_value_target_network2(next_states)
                next_q = torch.min(next_q1, next_q2)
                next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
                target_q = rewards + self.gamma * (~dones) * next_v

            q1 = self.q_value_network1(states).gather(1, actions)
            q2 = self.q_value_network2(states).gather(1, actions)
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)

            # Calculating the Policy target
            _, probs = self.policy_network(states)
            log_probs = torch.log(probs)
            with torch.no_grad():
                q1 = self.q_value_network1(states)
                q2 = self.q_value_network2(states)
                q = torch.min(q1, q2)

            policy_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            log_probs = (probs * log_probs).sum(-1)
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.update_counter += 1

            self.alpha = self.log_alpha.exp()

            if self.update_counter % self.config["fixed_network_update_freq"] == 0:
                self.hard_update_target_network()

            return alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def choose_action(self, states, do_greedy=False):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).byte().to(self.device)
        with torch.no_grad():
            dist, p = self.policy_network(states)
            if do_greedy:
                action = p.argmax(-1)
            else:
                action = dist.sample()
        return action.detach().cpu().numpy()[0]

    def hard_update_target_network(self):
        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()
        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

    def set_to_eval_mode(self):
        self.policy_network.eval()


class Disc_DIAYN_Agent: # Based on SAC agent in DIAYN code
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.state_shape = self.config["state_shape"]
        self.state_vec_size = np.prod(self.state_shape)

        self.n_skills = self.config["n_skills"]
        self.n_actions = self.config["n_actions"]
        self.lr = self.config["lr"]

        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)

        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() and config['cuda'] else "cpu"
        torch.manual_seed(self.config["seed"])
        #
        # policy network: from  (n_states + n_skills) -> n_actions
        self.policy_network = PolicyNetwork_DIAYN(state_shape=self.state_shape, n_actions=self.n_actions,n_skills=self.n_skills).to(self.device)

        self.q_value_network1 = QNetwork_DIAYN(state_shape=self.state_shape, n_actions=self.n_actions, n_skills=self.n_skills).to(self.device)

        self.q_value_network2 = QNetwork_DIAYN(state_shape=self.state_shape, n_actions=self.n_actions, n_skills=self.n_skills).to(self.device)


        self.value_network = ValueNetwork_DIAYN(state_shape = self.state_shape, n_skills=self.n_skills ).to(self.device)

        self.value_target_network = ValueNetwork_DIAYN(state_shape = self.state_shape, n_skills=self.n_skills).to(self.device)

        self.hard_update_target_network()
        self.discriminator = Discriminator(state_shape = self.state_shape, n_skills=self.n_skills).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).byte().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]


    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))
        states = torch.cat(batch.state).view(self.batch_size, *batch.state[0].shape).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view((-1, 1)).to(self.device)
        actions = torch.cat(batch.action).view((-1, 1)).long().to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, *batch.state[0].shape).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            policy_action, log_probs= self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states).gather(dim=1, index=policy_action.unsqueeze(1))
            q2 = self.q_value_network2(states).gather(dim=1, index=policy_action.unsqueeze(1))

            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach().unsqueeze(1)

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)
            next_state_vec = torch.split(next_states, [self.state_vec_size, self.n_skills], dim=-1)[0]
            logits = self.discriminator(next_state_vec)
            p_z = p_z.gather(-1, zs)

            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states).gather(dim=1, index=actions)
            q2 = self.q_value_network2(states).gather(dim=1, index=actions)

            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            # Calculating the policy loss
            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.state_vec_size, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            # Update networks
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()




    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)


    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)

class Disc_DIAYN_NN_Agent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.state_shape = self.config["state_shape"]
        self.n_hidden_filters = self.config["n_hiddens"]
        self.n_skills = self.config["n_skills"]
        self.n_actions = self.config["n_actions"]
        self.lr = self.config["lr"]

        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)

        self.memory = Memory(self.config["mem_size"], self.config["seed"])

        self.device = "cuda" if torch.cuda.is_available() and config['cuda'] else "cpu"
        torch.manual_seed(self.config["seed"])

        # policy network: (n_states+n_skills) -> n_actions
        self.policy_network = PolicyNetwork_NN(n_states=self.n_states,
                                            n_actions=self.config["n_actions"],
                                            n_skills = self.n_skills,
                                            n_hidden_filters=self.n_hidden_filters).to(self.device)

        self.q_value_network1 = QvalueNetwork_NN(n_states=self.n_states,
                                              n_actions=self.config["n_actions"],
                                              n_skills=self.n_skills,
                                              n_hidden_filters=self.n_hidden_filters).to(self.device)

        self.q_value_network2 = QvalueNetwork_NN(n_states=self.n_states,
                                              n_actions=self.config["n_actions"],
                                              n_skills=self.n_skills,
                                              n_hidden_filters=self.n_hidden_filters).to(self.device)

        self.value_network = ValueNetwork_NN(n_states=self.n_states, n_skills=self.n_skills,
                                          n_hidden_filters=self.n_hidden_filters).to(self.device)

        self.value_target_network = ValueNetwork_NN(n_states=self.n_states, n_skills=self.n_skills,
                                          n_hidden_filters=self.n_hidden_filters).to(self.device)

        self.hard_update_target_network()

        self.discriminator = Discriminator_NN(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=n_hidden_filters).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))
        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(
            self.device)
        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)

