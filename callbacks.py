""" Callback classes for versatile behavior in the Trainer object at specified checkpoints.
This program uses the NeuroMANCER framework <https://github.com/pnnl/neuromancer> which comes with BSD license.
It's modified to save data every 100 epochs.

"""


import os
from copy import deepcopy
import pickle
import torch
import dill

class Callback:
    """
    Callback base class which allows for bare functionality of Trainer
    """
    def __init__(self):
        pass

    def begin_train(self, trainer):
        pass

    def begin_epoch(self, trainer, output):
        pass

    def begin_eval(self, trainer, output):
        pass

    def end_batch(self, trainer, output):
        pass

    def end_eval(self, trainer, output):
        pass

    def end_epoch(self, trainer, output):
        print("epoch: "+str(trainer.current_epoch), flush=True)
        savedir = trainer.logger.savedir
        savepath = os.path.join(savedir, 'loss.dat')
        loss = ('y_loss', 'onestep_loss', 'x_loss', 'reconstruct_loss', 'train_loss', 'dev_loss')
        f = open(savepath, 'ab')
        entries = []
        for k, v in output.items():
            try:
                if k in loss:
                    entries.append(v.item())
            except (ValueError, AttributeError) as e:
                pass
        pickle.dump(entries , f)
        f.close()
        if (trainer.current_epoch % 100 == 0):
            savepath = os.path.join(savedir, 'best_model_epoch.pth')
            torch.save(trainer.best_model, savepath, pickle_module=dill)
            savepath = os.path.join(savedir, 'model_epoch.pth')
            torch.save(trainer.model, savepath, pickle_module=dill)
        return True

    def end_train(self, trainer, output):
        print("end of training")        
        return True

    def begin_test(self, trainer):
        pass

    def end_test(self, trainer, output):
        pass
