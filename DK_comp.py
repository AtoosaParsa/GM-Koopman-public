""" Deep Koopman framework for modeling and analysis of granular materials
The training data is generated from a two-particle system with random initial displacements.
A dataset is generated from the particle displacements (not the velocities) in time and then resampled with a fixed timestep.

This program uses the NeuroMANCER framework <https://github.com/pnnl/neuromancer> which comes with a BSD license.

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '0.0.24'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"



import sys
import os
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import pickle
import dill
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from sklearn.preprocessing import MinMaxScaler

from neuromancer.psl import plot
from neuromancer import psl
from neuromancer.system import Node, System
from neuromancer.slim import slim
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer. modules import blocks

from switch_binary_hertz import switch_hertz
from callbacks import Callback


def solve_granular(u_0, v_0, compression, sample_rate=10):
    """ Numerical simulation of the granular chain
    
        Parameters:
        u_0: initial displacements
        v_0: initial velocities
        compression: initial compression of the chain
        sample_rate: sampling rate for the trajectories

        Returns:
        X_sampled: displacements in time
        V_sampled: velocities in time
    
    """
    cont, Ek, Ep, p, X_numerical, V_numerical= switch_hertz.evaluate_2(stiffness, damping, compression, timeSteps, dt, u_0, v_0)
    X_sampled = X_numerical[::sample_rate]
    V_sampled = V_numerical[::sample_rate]
    
    return [X_sampled, V_sampled]



font = {'family' : 'sans-serif','weight' : 'semibold'}  
plt.rc('font', **font) 
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (15.0, 10.0)
plt.rcParams['savefig.bbox'] = 'tight'

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    dev = torch.device('cuda')
    torch.set_default_device('cuda')
else:
    dev = torch.device('cpu')
print(f"Device is: {dev}")



###### setup parameters

# set the stiffness values of the two particles
config = int(sys.argv[1])-1
print(f'particle configuration: {config}')

# output directory for saving the results
savedir = f'dk_hertz_{config}'

try:
    os.remove(savedir)
except OSError:
    pass

os.makedirs(savedir, exist_ok=True)

trajs = 10000 + 100
sample_rate = 2 #5
compression_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
stiffnesses = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
stiffness = stiffnesses[config]
comp_id = 0
damping = 0
dt = 5e-3
timeSteps = 2000
compression = compression_list[comp_id]
At = 1e-3


###### generate trajectories

x0_train = np.mgrid[-10*At:10.1*At:((10*At+10*At)/(100-1)), -10*At:10.1*At:((10*At+10*At)/(100-1))].reshape(2,-1).T
x0_train = np.random.permutation(np.round(x0_train, decimals=5))
x0_test = np.round(np.random.uniform(low=-10*At, high=10.1*At, size=[100, 2]), decimals=5)
x0 = np.vstack((x0_train, x0_test))
v0 = np.zeros((trajs, 2))

num_cores = multiprocessing.cpu_count()

print("start numerical simulations... ", flush=True)

outputs = Parallel(n_jobs=num_cores)(delayed(solve_granular)(x0[i, :], v0[i, :], compression, sample_rate)for i in np.arange(0, trajs))

print("trajectory generation ended", flush=True)

# adjust the simulation time and timestep according to the sampling rate
timeSteps = int(timeSteps/sample_rate)
dt = dt * sample_rate
data_raw = []
for i in outputs:
    out = np.hstack((np.reshape(i[0][:, 0], (timeSteps, 1)), \
                     np.reshape(i[0][:, 1], (timeSteps, 1))))
    data_raw.append(out)

print("resampled: simulation time is: "+str(timeSteps))
print("resampled: dt is: "+str(dt))
data_raw = np.array(data_raw)
print(f"data_raw shape: {data_raw.shape}", flush=True)

savepath = os.path.join(savedir, 'data.dat')
f = open(savepath, 'wb')
pickle.dump(data_raw, f)
f.close()

###### data normalization
# no normalization
data_scaled = data_raw

###### creating train, test, and dev datasets
train_sim = data_scaled[0:5000, :, :] # [trajs, timeSteps, inputSize]
dev_sim = data_scaled[5000:10000, :, :]
test_sim = data_scaled[10000:10100, :, :]
ny = len(stiffness) # dimension of the NN's input, positions and velocities of the particles
nsteps = int(timeSteps/2) # prediction step for training
nbatch = int(train_sim.shape[0] * (timeSteps / nsteps))
nsim = int(timeSteps) # actual simulation time, can be longer than nsteps, this is used for testing

trainY = train_sim.reshape(nbatch, nsteps, ny)
trainY = torch.tensor(trainY, dtype=torch.float32, device=dev)
print(f"training set: {trainY.size()}")

train_data = DictDataset({'Y': trainY, 'Y0': trainY[:, 0:1, :]}, name='train') # Y0 is the state (x, v) in the first timestep
train_loader = DataLoader(train_data, batch_size=500,
                          collate_fn=train_data.collate_fn, shuffle=True, generator=torch.Generator(device=dev),)

nbatch = int(dev_sim.shape[0] * (timeSteps / nsteps))
devY = dev_sim.reshape(nbatch, nsteps, ny)
devY = torch.tensor(devY, dtype=torch.float32, device=dev)
dev_data = DictDataset({'Y': devY, 'Y0': devY[:, 0:1, :]}, name='dev')
dev_loader = DataLoader(dev_data, batch_size=500,
                        collate_fn=dev_data.collate_fn, shuffle=True, generator=torch.Generator(device=dev),)

testY = test_sim.reshape(test_sim.shape[0], nsim, ny)
testY = torch.tensor(testY, dtype=torch.float32, device=dev)
test_data = {'Y': testY, 'Y0': testY[:, 0:1, :], 'name': 'test'}

###### NN model parameters
nx_koopman = 50
n_hidden = 200
n_layers = 5
learningRate = 0.0001
weights = [10., 1., 1., 1.]
epochs = 50000

# encoder NN
encode = blocks.MLP(ny, nx_koopman, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU, hsizes=n_layers*[n_hidden])
encode_Y0 = Node(encode, ['Y0'], ['X'], name='encoder_Y0')
encode_Y = Node(encode, ['Y'], ['X_traj'], name='encoder_Y')

# decoder NN
decode = blocks.MLP(nx_koopman, ny, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU, hsizes=n_layers*[n_hidden])
decode_y0 = Node(decode, ['X'], ['Yhat0'], name='decoder_Y0')
decode_y = Node(decode, ['X'], ['Yhat'], name='decoder_Y')

# Koopman NN
K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)
Koopman = Node(K, ['X'], ['X'], name='K')

dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)
nodes = [encode_Y0, decode_y0, encode_Y, dynamics_model, decode_y]

# loss functions
Y = variable("Y")
Y0 = variable('Y0')
Yhat = variable('Yhat')
Yhat0 = variable('Yhat0')
X_traj = variable('X_traj')
X = variable('X')

#### new loss functions and changed the weights --> update the paper
# trajectory prediction loss
y_loss = weights[0]*(Yhat[:, 1:-1, :] == Y[:, 1:, :])^2
y_loss.name = "y_loss"

# one-step prediction loss
onestep_loss = weights[1]*(Yhat[:, 1, :] == Y[:, 1, :])^2
onestep_loss.name = "onestep_loss"

# latent trajectory prediction loss
x_loss = weights[2]*(X[:, 1:-1, :] == X_traj[:, 1:, :])^2
x_loss.name = "x_loss"

# reconstruction loss
reconstruct_loss = weights[3]*(Y0 == Yhat0)^2
reconstruct_loss.name = "reconstruct_loss"

objectives = [y_loss, x_loss, onestep_loss, reconstruct_loss]
loss = PenaltyLoss(objectives, constraints=[])

problem = Problem(nodes, loss)

optimizer = torch.optim.Adam(problem.parameters(), lr=learningRate)
logger = BasicLogger(args=None, savedir=savedir, verbosity=1, stdout=["dev_loss", "train_loss"])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=epochs/2,
    warmup=epochs/2,
    epochs=epochs,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
    callback=Callback(),
    device = dev,
)

##### training
best_model = trainer.train()

savepath = os.path.join(savedir, 'sys.pth')
torch.save(dynamics_model, savepath, pickle_module=dill)

##### testing
problem.load_state_dict(best_model)
problem.nodes[3].nsteps = test_data['Y'].shape[1]

print(f"test data: {test_data['Y'].shape}")
for sample in range(1, 5):
    testY = test_sim[sample, :, :].reshape(1, nsim, ny)
    testY = torch.tensor(testY, dtype=torch.float32, device=dev)
    test_data = {'Y': testY, 'Y0': testY[:, 0:1, :], 'name': 'test'}

    test_outputs = problem.step(test_data)

    pred_traj = test_outputs['Yhat'][:, 1:-1, :].detach().cpu().numpy().reshape(-1, ny).T
    true_traj = test_data['Y'][:, 1:, :].detach().cpu().numpy().reshape(-1, ny).T
    #print(pred_traj.shape)

    # plot trajectories
    figsize = 10
    fig, ax = plt.subplots(ny, figsize=(figsize, figsize))
    labels = [f'$X_{k}$' for k in range(ny)]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        if ny > 1:
            axe = ax[row]
        else:
            axe = ax
        axe.set_ylabel(label, labelpad=20, fontsize=14, fontweight='bold')
        axe.set_axisbelow(True)
        axe.grid(which='minor', color='gray', linestyle=':', linewidth=0.2)
        axe.grid(which='major', color='gray', linestyle='-', linewidth=0.4)
        axe.plot(t1, color='tomato', linestyle='solid', linewidth=3.0, label='Desired')
        axe.plot(t2, color='cornflowerblue', linestyle='dashed', linewidth=3.0, label='Predicted')
        axe.tick_params(labelbottom=False, labelsize=14)
        axe.legend(loc='upper right', fontsize=14)
    
    axe.tick_params(labelbottom=True, labelsize=14)    
    axe.set_xlabel('$Time$', fontsize=14)

    plt.tight_layout()
    savename = 'predictions_s'+str(sample)+'.png'
    savepath = os.path.join(savedir, savename)
    plt.savefig(savepath)
    plt.show()
    plt.close(fig)

# compute eigenvalues and eigenvectors
eig, eig_vec = torch.linalg.eig(K.weight)

eReal = eig.real.detach().cpu().numpy()
eImag = eig.imag.detach().cpu().numpy()

print("Koopman eigenvalues: ")
print(eig)
print("eigenvectors: ")
print(eig_vec)

# plot Koopman eigenvalues
t = np.linspace(0.0, 2 * np.pi, 1000)
x_circ = np.cos(t)
y_circ = np.sin(t)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x_circ, y_circ, color='cornflowerblue', linewidth=4, zorder=1)
ax.scatter(eReal, eImag, color='red', marker='o', zorder=2)
ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.2)
ax.grid(which='major', color='gray', linestyle='-', linewidth=0.4)
ax.set_aspect('equal', 'box')
ax.set_xlabel("$Re(\lambda)$", fontsize=14)
ax.set_ylabel("$Im(\lambda)$", fontsize=14)
fig.suptitle('Koopman Operator Eigenvalues')

plt.tight_layout()
savename = 'eigenvalues.png'
savepath = os.path.join(savedir, savename)
plt.savefig(savepath)
plt.show()
plt.close(fig)
