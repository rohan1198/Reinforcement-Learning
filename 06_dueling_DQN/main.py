import numpy as np
import time
import argparse
import collections

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from lib import agent, common, experience, network, wrappers



if __name__ == "__main__":
    params = common.HYPERPARAMS["pong"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default = False, action = "store_true", help = "Enable CUDA")
    parser.add_argument("--env", default = params["env_name"], help = "Name of the environment")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    if device == torch.device("cuda"):
        print(torch.cuda.get_device_properties(device))
    else:
        print("Running on CPU!")
    print("\n")

    env = wrappers.make_env(args.env)

    net = network.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = network.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment = "-" + params["run_name"] + "-dueling")
    print(net)
    print("\n")

    buffer = experience.ReplayBuffer(capacity = params["replay_size"])
    agent = agent.Agent(env, buffer)
    
    epsilon = params["epsilon_start"]
    optimizer = optim.Adam(net.parameters(), lr = params["learning_rate"])
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_frames"])

        reward = agent.play_step(net, epsilon, device = device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])

            print(f"{frame_idx} frames | {len(total_rewards)} games | {reward:.3f} reward | {mean_reward:.3f} mean reward | {epsilon:.2f} epsilon | {speed:.2f} frames/sec")

            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")

                if best_mean_reward is not None:
                    print(f"Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f} model saved!")
                
                best_mean_reward = mean_reward
            
            if mean_reward > params["stop_reward"]:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < params["replay_initial"]:
            continue

        if frame_idx % params["target_net_sync"]:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(params["batch_size"])
        
        loss_t = common.calc_loss(batch, net, tgt_net, params["gamma"], device = device)
        loss_t.backward()
        optimizer.step()

    writer.close()