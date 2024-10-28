import torch
import torch.nn as nn
import numpy as np
import os
import logging
from torch.utils.data import Dataset, DataLoader

from model.joint import Joint_model
from model.gaitgraph.gaitgraph2 import GaitGraph2
from model.tdgcn import Model as TD
from model.ctrgcn import CTRGCN

model_list = {
    "joint": Joint_model,
    "graph": GaitGraph2,
    "TD": TD,
    "CTR": CTRGCN,
}

# Load test data
test_data_joint = np.load('./data/test_joint.npy').copy()
test_data_bone = np.load('./data/test_bone.npy').copy()
test_data_joint_motion = np.load('./data/test_joint_motion.npy').copy()
test_data_bone_motion = np.load('./data/test_bone_motion.npy').copy()
test_label = np.load('./data/test_label.npy').copy()
label = torch.from_numpy(test_label).cuda()

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(name + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class TestLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.label = torch.from_numpy(test_label).cuda()
        self.joint = torch.from_numpy(test_data_joint).cuda()
        self.bone = torch.from_numpy(test_data_bone).cuda()
        self.joint_motion = torch.from_numpy(test_data_joint_motion).cuda()
        self.bone_motion = torch.from_numpy(test_data_bone_motion).cuda()

    def __getitem__(self, index):
        return {
            "joint": self.joint[index],
            "bone": self.bone[index],
            "joint_motion": self.joint_motion[index],
            "bone_motion": self.bone_motion[index],
        }, self.label[index]

    def __len__(self):
        return len(self.label)

def evaluate_model(model, testloader, mod, logger):
    model.eval()
    model.cuda().half()
    pred = torch.zeros((2000, 155)).cuda()  # Assuming 2000 samples, 155 classes
    with torch.no_grad():
        for batch, (train_data, _) in enumerate(testloader):
            pred[batch * 20:(batch + 1) * 20] = model(train_data[mod].half())
    idx = torch.argmax(pred, dim=1)
    acc = torch.sum(idx == label) / 2000
    logger.info(f"{model.__class__.__name__} accuracy: {acc.item():.4f}")
    return pred


def get_result(model_names, mods, models):
    logger = get_logger("test")
    testloader = DataLoader(TestLoader(), batch_size=20, num_workers=4)
    pred_list = []
    label = torch.from_numpy(np.load('./data/test_label.npy')).cuda()  # 假设标签已加载到GPU

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f"./ckpt/{model_names[i]}.pth", map_location='cuda'))
        pred = evaluate_model(model, testloader, mods[i], logger)
        pred_list.append(pred)

    # Aggregate predictions
    p = torch.sum(torch.stack(pred_list), dim=0)
    p = torch.argmax(p, dim=1)
    acc = torch.sum(p == label) / 2000
    logger.info(f"Ensemble prediction accuracy: {acc.item():.4f}")

    # Save results
    os.makedirs("./res", exist_ok=True)
    name = model_names[0] if len(model_names) == 1 else "vm"
    np.save(f"./res/{name}.npy", np.array(pred_list))


# Define model names and instances
names = ["bone_CTR_60", "joint_CTR_60", "joint_TD_60"]
models = [CTRGCN(), CTRGCN(), TD()]
mods = ["bone", "joint", "joint"]

get_result(names, mods, models)