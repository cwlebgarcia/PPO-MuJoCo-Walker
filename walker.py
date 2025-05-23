from dm_control import suite,viewer
import numpy as np
import torch
from torch.optim import Adam
from backend import ActorCritic, uth_t,ValueFunction
import torch.nn.functional as F
from torch.distributions.normal import Normal


def discount_cumsum(x, gamma):
    """
    Compute discounted cumulative sums of vectors.
    """
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(len(x)-1)):
        out[t] = x[t] + gamma * out[t+1]
    return out


# Calculate advantage.
def calc_adv(rews,vals,gamma,lam):
    deltas = rews + gamma * vals[1:] - vals[:-1]
    advantages = []
    advantage = 0

    for delta in reversed(deltas):
        advantage = delta + gamma * lam * advantage
        advantages.append(advantage)

    advantages = list(reversed(advantages))

    return np.array(advantages)

# Surrogate loss function for continuous action space.
def compute_loss_pi(data, q_net, clip_ratio=0.2):
    obs, acts, logp_old, adv = data["obs"], data["acts"], data["logp"], data["adv"]
    
    mu, std = q_net(obs)
    pi = Normal(mu, std)
    logp = pi.log_prob(acts).sum(-1)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -torch.min(ratio * adv, clip_adv).mean()

    approx_kl = (logp_old-logp).mean().item()

    return loss_pi,approx_kl

def compute_loss_v(data, v_net):
    obs, ret = data["obs"], data["ret"]
    v_pred = v_net(obs)
    return F.mse_loss(v_pred, ret)

def update(q_net,v_net, data, pi_optimizer, v_optimizer,target_kl,clip_ratio, train_pi_iters=80, train_v_iters=80):
    # Normalize the advantage
    adv= data["adv"]
    adv = (adv-adv.mean())/(adv.std() + 1e-9)
    data["adv"] =adv
    for _ in range(train_pi_iters):
        loss_pi,approx_kl = compute_loss_pi(data, q_net)
        if approx_kl > target_kl:
            break
        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()
    for _ in range(train_v_iters):
        loss_v = compute_loss_v(data, v_net)
        v_optimizer.zero_grad()
        loss_v.backward()
        v_optimizer.step()


# Rollour is the process of using the current policy to interact with the environment, training is updating the policy based on the data from the rollout.
def rollout(e,ac,T=1000):
    """
    e: environment
    uth: controller
    T: time-steps
    """

    traj=[]
    t=e.reset()
    x=t.observation
    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
    for _ in range(T):
        with torch.no_grad():
            u,v,log_p = ac.step(torch.from_numpy(x).float().unsqueeze(0).to(device))
        r = e.step(u)
        x= r.observation
        xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
        
        t=dict(xp=xp,
               u=u,
               r=r.reward,
               d=r.last(),
               v = v.item(),
               logp = log_p.item())
        traj.append(t)
        x=xp
        if r.last():
            break
    return traj

def ppo(e,q_net,v_net,epochs,traj_steps,pi_lr,v_lr,clip_ratio,target_kl,
        gamma,lam,train_pi_iters,train_v_iters,device):

    # Establish optimizers
    pi_optimizer = Adam(q_net.parameters(),lr = pi_lr)
    v_optimizer = Adam(v_net.parameters(),lr= v_lr)

    # Initialize buffers
    rew_hist = []
    disc_ret_hist = []
    timestep_hist = []
    total_timesteps = 0
    for epoch in range(epochs):
        state_buffer, action_buffer = [], []
        rew_buffer, value_buffer = [], []
        logp_buffer = []
        ep_rew = 0

        t = e.reset()
        x=t.observation
        x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
        
        for t in range(traj_steps): # Run for the maximum number of steps from hyperparameters. Since there is no failure condition it can train indefinitely.
            total_timesteps +=1
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
            with torch.no_grad():
                # Get action and value from the state
                mu,std = q_net(x_tensor)
                pi = Normal(mu,std)
                u = pi.sample()
                logp = pi.log_prob(u).sum(-1)
                v = v_net(x_tensor)

            u_np = u.squeeze(0).cpu().numpy()
            r = e.step(u_np)
            xp = r.observation
            xp = np.array(xp['orientations'].tolist()+[xp['height']]+xp['velocity'].tolist())
            
            state_buffer.append(x)
            action_buffer.append(u_np)
            rew_buffer.append(r.reward)
            value_buffer.append(v.item())
            logp_buffer.append(logp.item())

            x= xp
            ep_rew += r.reward
            if r.last() or t == traj_steps - 1:
                final_val = 0
                if not r.last():
                    with torch.no_grad():
                        final_val = v_net(torch.from_numpy(x).float().unsqueeze(0)).item()

                values_plus_final = np.append(value_buffer, final_val)
                
                adv_buffer = calc_adv(rew_buffer,values_plus_final,gamma,lam)
                returns = adv_buffer + np.array(value_buffer)[:len(adv_buffer)] if len(value_buffer) > 0 else adv_buffer

                data = dict(obs = torch.tensor(np.array(state_buffer),dtype = torch.float32).to(device),
                            acts = torch.tensor(np.array(action_buffer),dtype = torch.float32).to(device),
                            logp = torch.tensor(np.array(logp_buffer),dtype = torch.float32).to(device),
                            adv = torch.tensor(adv_buffer,dtype = torch.float32).to(device),
                            ret = torch.tensor(returns,dtype = torch.float32).to(device)
                            )
                update(q_net,v_net,data,pi_optimizer,v_optimizer,target_kl,clip_ratio,train_pi_iters,train_v_iters)

                disc_return = discount_cumsum(np.array(rew_buffer),gamma)[0]
                disc_ret_hist.append(disc_return)
                timestep_hist.append(total_timesteps)
                rew_hist.append(ep_rew)

                # Clear buffers for next trajectory
                state_buffer.clear()
                action_buffer.clear()
                rew_buffer.clear()
                value_buffer.clear()
                logp_buffer.clear()


                if r.last():
                    r = e.reset()
                    x = r.observation
                    x = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
                    ep_rew = 0
                if t == traj_steps - 1:
                    break

        avg_return = np.mean(rew_hist) if rew_hist else 0.0
        print(f"Epoch {epoch:3d}: Avg Reward = {avg_return:.2f}")

    return np.array(timestep_hist), np.array(rew_hist), np.array(disc_ret_hist)
                
                

if __name__ == '__main__':
    r0 = np.random.RandomState(2332) # Initialize seed?
    e = suite.load('walker', 'walk',
                    task_kwargs={'random': r0}) # Load the environment. Doesn't use gym so it's a bit different
    U=e.action_spec();udim=U.shape[0]; # Get the shape of the action space
    X=e.observation_spec();xdim=14+1+9; # Get the shape of the observation space.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    print("Using device:", device)

    epochs = 3000 # Number of training epochs
    gamma, lam = 0.99, 0.95 # Discount factor. It doesn't look like they ever use lam anywhere else
    pi_lr, v_lr = 3e-4, 1e-3 # Learning rates for the value and policy networks
    train_pi_iters, train_v_iters = 80, 80 # Number of times to perform policy updates I believe
    gamma, lam = 0.99, 0.95 # Lowkey do it again
    max_ep_len = 2000 # Maximum number of steps per episode
    clip_ratio =0.2 # Offset for clamping surrogate loss
    target_kl = 1.5*0.01
    #ac = ActorCritic(xdim,udim).to(device)
    q_net = uth_t(xdim,udim).to(device) # Supposedly action network but the notation makes it seems like it's the reward
    v_net = ValueFunction(xdim).to(device) # Value Network

    # Pass arguments to the PPO function Lowkey bro might have been high since he just defined all of these
    timesteps, rewards, disc_returns = ppo( 
        e, q_net, v_net,
        epochs=3000, traj_steps=2000,
        pi_lr=3e-4, v_lr=1e-3,
        clip_ratio=0.2, target_kl=1.5e-2,
        gamma=0.99, lam=0.95,
        train_pi_iters=80, train_v_iters=80,
        device=device
    )


        
    torch.save(q_net.state_dict(), "ppo_walk.pt") # Saves the model parameters