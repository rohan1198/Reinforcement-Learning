import torch



def projection_distribution(target_model, next_observation, reward, done, batch_size, vmin, vmax, atoms, gamma, n_step, device):
    delta_z = float(vmax - vmin) / (atoms - 1)
    support = torch.linspace(vmin, vmax, atoms).to(device)

    next_dist = target_model.forward(next_observation).detach().mul(support)
    next_action = next_dist.sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, atoms)
    next_dist = next_dist.gather(1, next_action).squeeze(1) 

    reward = reward.unsqueeze(1).expand_as(next_dist)
    done = done.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = reward + (1 - done) * (gamma ** n_step) * support
    Tz = Tz.clamp(min = vmin, max = vmax)
    b = (Tz - vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * atoms, batch_size).long().unsqueeze(1).expand_as(next_dist).to(device)

    proj_dist = torch.zeros_like(next_dist, dtype=torch.float32)
    proj_dist.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist




def calc_loss(eval_model, target_model, buffer, vmin, vmax, atoms, gamma, batch_size, n_steps, device):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_observation = torch.FloatTensor(next_observation).to(device)
    done = torch.FloatTensor(done).to(device)

    proj_dist = projection_distribution(target_model, next_observation, reward, done, batch_size, vmin, vmax, atoms, gamma, n_steps, device)

    dist = eval_model.forward(observation)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.detach().clamp_(0.01, 0.99)

    loss = - (proj_dist * dist.log()).sum(1).mean()

    return loss
