import json

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from .nets.mlps import MLPCritic
from .nets.rnns import RobRNNActor
from .nets.rnns import RobRNNTildeActor


def pathlength(path):
    return len(path["reward"])


class PGAgent:
    def __init__(
            self,
            computation_graph_args,
            sample_trajectory_args,
            estimate_return_args,
            tilde=False,
            init_trunc_norm=False,
    ):
        super(PGAgent, self).__init__()
        self._init_dicts = {
            'computation_graph_args': computation_graph_args,
            'sample_trajectory_args': sample_trajectory_args,
            'estimate_return_args': estimate_return_args,
        }

        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.states_size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.rnn_bias = computation_graph_args['rnn_bias']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args[
            'min_timesteps_per_batch']
        self.step_num = sample_trajectory_args['step_num']  # RNN steps
        self.rnn_test_nostates = sample_trajectory_args['rnn_test_nostates']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args[
            'normalize_advantages']

        if self.discrete:
            # should be a logit over discrete predictions
            raise NotImplementedError
        else:
            if tilde:
                self.rnn = RobRNNTildeActor(
                    self.ob_dim,
                    self.size,
                    self.states_size,
                    self.ac_dim,
                    bias=self.rnn_bias,
                    dtype=torch.float32,
                    use_trunc_normal=init_trunc_norm,
                )
            else:
                self.rnn = RobRNNActor(
                    self.ob_dim,
                    self.size,
                    self.states_size,
                    self.ac_dim,
                    bias=self.rnn_bias,
                    dtype=torch.float32,
                    use_trunc_normal=init_trunc_norm,
                )

        self.opt = torch.optim.Adam(
            self.rnn.parameters(),
            lr=self.learning_rate,
        )

        if self.nn_baseline:
            self.mlp = MLPCritic(
                self.ob_dim,
                self.size,
                self.n_layers,
                dtype=torch.float32,
                use_trunc_normal=init_trunc_norm,
            )
            self.baseline_opt = torch.optim.Adam(
                self.mlp.parameters(),
                lr=self.learning_rate
            )

    def save(self, savedir):
        json.dump(self._init_dicts, open(savedir + 'args.json', 'w+'))
        torch.save(self.rnn, savedir + 'RobustRNNActor.pt')
        if self.nn_baseline:
            torch.save(self.mlp, savedir + 'MLPCritic.pt')

    @staticmethod
    def load(savedir):
        init_dict = json.load(open(savedir + 'args.json'))
        agent = PGAgent(
            init_dict['computation_graph_args'],
            init_dict['sample_trajectory_args'],
            init_dict['estimate_return_args'],
        )
        agent.rnn = torch.load(savedir + 'RobustRNNActor.pt')
        agent.rnn.eval()
        if agent.nn_baseline:
            agent.mlp = torch.load(savedir + 'MLPCritic.pt')
            agent.mlp.eval()

        return agent

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        with tqdm(total=self.min_timesteps_per_batch) as tbar:
            while True:
                animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
                path = self.sample_trajectory(env, animate_this_episode)
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                if timesteps_this_batch > self.min_timesteps_per_batch:
                    break

                tbar.update(n=pathlength(path))
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode, init_state=None, horizon=None, sample=True):
        # setup storage
        obs, acs, rewards, log_ps, dones, sts = [], [], [], [], [], []

        # re-initialize
        if init_state is None:
            ob = env.reset()
        else:
            _ = env.reset()
            env._state = np.array(init_state, dtype=np.float32)
            ob = env.get_obs()

        xi0 = torch.zeros(self.states_size, dtype=torch.float32)

        xi = xi0.clone()
        steps = 0
        while True:
            ob = torch.FloatTensor(ob)
            st = torch.FloatTensor(env.get_state())

            # record observation
            sts.append(st)
            obs.append(ob)

            # forward pass
            ac_mean, xi, _, ac, log_p = self.rnn.forward(ob, xi=xi, sample=sample)

            if self.rnn_test_nostates:
                xi = xi0.clone()

            # record control
            acs.append(ac)
            log_ps.append(log_p)

            # advance environment
            ob, rew, done, _ = env.step(ac)

            # record signals
            rewards.append(rew)
            dones.append(done)

            steps += 1

            if horizon is not None:
                if steps > horizon:
                    dones[-1] = True
                    break
            else:
                if done or steps > self.max_path_length:
                    dones[-1] = True
                    break

        path = {
            "state": torch.stack(sts),
            "observation": torch.stack(obs),
            "reward": torch.stack(rewards),
            "action": torch.stack(acs),
            "log_probs": torch.stack(log_ps),
            "termination": torch.FloatTensor(
                np.array(dones, dtype=np.float32)),
        }

        return path

    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path
            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
            ----------------------------------------------------------------------------------

            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in
            Agent.define_placeholders).

            Recall that the expression for the policy gradient PG is

                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

            where

                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t.

            You will write code for two cases, controlled by the flag 'reward_to_go':

              Case 1: trajectory-based PG

                  (reward_to_go = False)

                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
                  entire trajectory (regardless of which time step the Q-value should be for).

                  For this case, the policy gradient estimator is

                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

                  where

                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

                  Thus, you should compute

                      Q_t = Ret(tau)

              Case 2: reward-to-go PG

                  (reward_to_go = True)

                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute

                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}


            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above.
        """
        q_n = np.zeros(0)
        if self.reward_to_go:
            for _ in range(len(re_n)):
                r = re_n[_]
                T = r.shape[0]
                q_path = np.zeros(T)
                temp = 0
                for t in range(T - 1, -1, -1):
                    q_path[t] = r[t] + self.gamma * temp
                    temp = q_path[t]
                q_n = np.append(q_n, q_path)
        else:
            for _ in range(len(re_n)):
                r = re_n[_]
                T = r.shape[0]
                w = np.power(self.gamma, range(T))
                q_path = np.sum(w * r)
                q_n = np.append(q_n, q_path * np.ones(T))
        return q_n

    def compute_advantage(self, ob_no, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        if self.nn_baseline:
            # make baseline prediction
            with torch.no_grad():
                b_n = self.mlp(ob_no)

                # re-scale to match current q value stats
                b_n = b_n * q_n.std() + q_n.mean()

                # reward - baseline_prediction
                adv_n = torch.tensor(q_n).unsqueeze(dim=-1) - b_n
        else:
            adv_n = q_n.copy()

        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path
            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        return q_n, adv_n.squeeze()

    def update_parameters(self, ob_no, q_n, adv_n, lp_n, grad_clip=0.5):
        """
            Update the parameters of the policy and (possibly) the neural network baseline,
            which is trained to approximate the value function.
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
            returns:
                nothing
        """
        if self.nn_baseline:
            # re-scale targets to have mean=0, std=1
            target_n = torch.FloatTensor((q_n - q_n.mean()) / (q_n.std() + 1e-8))

            # import pdb
            # pdb.set_trace()

            c_loss = 0.0
            for _ in range(20):
                bn = self.mlp(ob_no).squeeze()

                # pdb.set_trace()
                loss_fn = nn.MSELoss()
                base_loss = loss_fn(bn, target_n)
                # pdb.set_trace()
                c_loss += (1/20) * base_loss.item()
                self.baseline_opt.zero_grad(set_to_none=True)
                base_loss.backward()
                self.baseline_opt.step()

        self.opt.zero_grad(set_to_none=True)

        # Now we have to slice the samples into chunks for rnn
        t = self.step_num
        n = len(ob_no) // t  # number of chunks (new n, aka batch size)

        # Slicing into compatible sizes
        adv_nt = adv_n[:n*t].reshape(n, t, -1)
        logprob_nt = lp_n[:n*t].reshape(n, t, -1)

        policy_loss = (-adv_nt * logprob_nt).mean()

        policy_loss.backward()

        # nn.utils.clip_grad_value_(self.rnn.parameters(), 10.0)
        nn.utils.clip_grad_value_(self.rnn.parameters(), grad_clip)
        # nn.utils.clip_grad_value_(self.rnn.parameters(), 0.3)
        # nn.utils.clip_grad_value_(self.rnn.parameters(), 0.1)

        # nn.utils.clip_grad_norm_(self.rnn.parameters(), 10.0)
        # nn.utils.clip_grad_norm_(self.rnn.parameters(), 2.0)
        # print(nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=0.01, norm_type='inf'))

        self.opt.step()

        if self.nn_baseline:
            return policy_loss.item(), c_loss

        return policy_loss.item()
