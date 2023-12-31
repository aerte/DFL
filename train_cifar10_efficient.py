#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/03/05 16:14:14
@Author  :   Bo (+ adapted by Felix)
'''
import mnist_utils as mnist_utils
import numpy as np
import torch
import os
import torch.nn as nn
import time
import configs
import pickle
import get_subset_cifar10 as gsc
from numpy import savetxt

device = torch.device("cuda")


class Train(object):
    def __init__(self, conf, data_group, num_local_epochs, sigma, exist_model):
        self.conf = conf
        self.data_group = data_group
        self.num_local_epochs = num_local_epochs
        self.model_use = mnist_utils.create_model(conf).to(device)

        if conf.round > 0:
            self.model_use.load_state_dict(exist_model)

        self.optimizer = mnist_utils.define_optimizer(self.model_use, lr=conf.lr)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.tr_data_loader, self.tt_data_loader = data_group

        print("Training data size", len(self.tr_data_loader))
        print("Testing data size", len(self.tt_data_loader))

        # Asserting that the classifier actually trains

        parameter_list = [p for p in self.model_use.parameters() if p.requires_grad == False]
        assert len(parameter_list) == 0

    def get_grad(self):
        grad_group = []
        for name, p in self.model_use.named_parameters():
            if p.requires_grad and "bias" not in name:
                grad_group.append(np.reshape(p.grad.data.detach().cpu().numpy(), [-1]))
        return grad_group[0]

    def _update_batch_tr(self, _image, _label, global_step):
        _image, _label = _image.to(device), _label.to(device)
        #        print(_image.shape, _label.shape)
        self.optimizer.zero_grad()
        _pred = self.model_use(_image)
        #        print("the shape of the prediction", _pred.shape)
        _loss = self.loss_fn(_pred, _label) / len(_image)
        #        print("the loss", _loss)
        _loss.backward()
        self.optimizer.step()
        accu = (_pred.argmax(axis=-1) == _label).sum().div(len(_image))
        #print("Training loss: {:.4f} and Training accuracy {:.2f}".format(_loss.item(), accu.item()))
        return self.get_grad()

    def _eval(self, global_step, data_use, str_use):
        self.model_use.eval()
        val_loss, val_accu = 0.0, 0.0
        for i, (_image, _label) in enumerate(data_use):
            _image, _label = _image.to(device), _label.to(device)

            _pred = self.model_use(_image)
            _loss = self.loss_fn(_pred, _label)
            _accu = (_pred.argmax(axis=-1) == _label).sum()
            val_loss += _loss.detach().cpu().numpy()
            val_accu += _accu.detach().cpu().numpy()
        print("{} loss: {:.4f} and {} accuracy {:.2f}".format(str_use, val_loss / len(data_use) / len(_image),
                                                              str_use, val_accu / len(data_use) / len(_image)))
        return val_loss, val_accu / len(data_use) / len(_image)

    def run(self):
        global_step = 0
        for j in range(self.num_local_epochs):
            grad_group = []
            for i, (_im, _la) in enumerate(self.tr_data_loader):
                _grad_group = self._update_batch_tr(_im, _la, global_step)
                grad_group.append(_grad_group)
                global_step += 1
        _val_loss, _val_accu = self._eval(global_step, self.tt_data_loader, "test")
        return self.model_use.state_dict()


def run_train(conf, tr_loader, tt_loader, exist_model):
    """
    model_dir: ../exp_data/../communication_round_%02d/
    """
    print("===========================================================")
    print("                    Local ID %02d " % conf.use_local_id)
    print("===========================================================")

    print("The used batch size %d for client %d at round %d" % (conf.batch_size, conf.use_local_id, conf.round))

    train_obj = Train(conf, [tr_loader, tt_loader], conf.num_local_epochs, conf.sigma, exist_model)
    client_model = train_obj.run()
    print("Done Local ID %02d" % conf.use_local_id)
    torch.save(client_model,
               conf.model_dir + "/client_id_%02d.pt" % conf.use_local_id)
    return client_model


def run_server(conf):
    time_init = time.time()
    model_group = {}
    dir2load = conf.model_dir
    print("starting to calculate the aggregated model")
    for i in range(conf.n_clients):
        try:
            _model_param = torch.load(dir2load + "/client_id_%02d.pt" % i, map_location=device)
        except EOFError:
            print("Having problems loading model parameters")
            return 0.0
        if i == 0:
            for k in _model_param.keys():
                model_group[k] = _model_param[k] * (1 / conf.n_clients)
        else:
            for k in _model_param.keys():
                model_group[k] += _model_param[k] * (1 / conf.n_clients)
    torch.save(model_group, dir2load + "/aggregated_model.pt")
    # Validation on Test Set
    tt_loss, tt_accu, preds, taf = check_test_accuracy(model_group, conf)
    ####### Checking the overconfidence on the Training Set #######
    tr_loss, tr_accu, tr_preds, tr_taf = test_server_on_training(model_group, conf)

    print("time on the server", time.time() - time_init)

    # Testing the server on the training data to test for overconfidence

    return tt_loss, tt_accu, preds, taf, tr_loss, tr_accu, tr_preds, tr_taf

def test_server_on_training(model_checkpoints, conf):
    print('########## Test Server on Training Data ############')

    tt_loader = gsc.get_cifar10_dataset_server(conf)

    model_use = mnist_utils.create_model(conf).to(device)
    model_use.load_state_dict(model_checkpoints)
    loss, accu = 0.0, 0.0
    num_class = 10

    batch_size = conf.batch_size

    preds = np.zeros([len(tt_loader) * batch_size, num_class])
    taf = np.zeros([len(tt_loader) * batch_size, 2])

    print("---------Prediction shape: ", preds.shape)

    for i, (_im, _la) in enumerate(tt_loader):
        _im, _la = _im.to(device), _la.to(device)
        _pred = model_use(_im)
        _loss = nn.CrossEntropyLoss(reduction='sum')(_pred, _la) / len(_im)
        _accu = (_pred.argmax(axis=-1) == _la).sum()
        loss += _loss.detach().cpu().numpy()
        accu += _accu.detach().cpu().numpy()

        ####
        preds[i * batch_size:(i + 1) * batch_size] = _pred.detach().cpu().numpy()
        taf[i * batch_size:(i + 1) * batch_size, 0] = _la.detach().cpu().numpy()
        taf[i * batch_size:(i + 1) * batch_size, 1] = _pred.argmax(axis=-1).detach().cpu().numpy()
        ####

    #print("The shape of the prediction", np.shape(preds))
    loss = loss / len(tt_loader)
    accu = accu / len(tt_loader) / batch_size
    print("Server model TRAINING SET loss: %.4f and accuracy: %.4f" % (loss, accu))
    print('####################################################')
    return loss, accu, preds, taf




def check_test_accuracy(model_checkpoints, conf):

    tt_loader = gsc.get_cifar10_test_dataset(1000)

    model_use = mnist_utils.create_model(conf).to(device)
    model_use.load_state_dict(model_checkpoints)
    loss, accu = 0.0, 0.0
    num_class = 10

    ####
    preds = np.zeros([len(tt_loader) * 1000, num_class])
    taf = np.zeros([len(tt_loader) * 1000, 2])
    ####

    for i, (_im, _la) in enumerate(tt_loader):
        _im, _la = _im.to(device), _la.to(device)
        _pred = model_use(_im)
        _loss = nn.CrossEntropyLoss(reduction='sum')(_pred, _la) / len(_im)
        _accu = (_pred.argmax(axis=-1) == _la).sum()
        loss += _loss.detach().cpu().numpy()
        accu += _accu.detach().cpu().numpy()

        ####
        preds[i * 1000:(i + 1) * 1000] = _pred.detach().cpu().numpy()
        taf[i * 1000:(i + 1) * 1000, 0] = _la.detach().cpu().numpy()
        taf[i * 1000:(i + 1) * 1000, 1] = _pred.argmax(axis=-1).detach().cpu().numpy()
        ####

    print("The shape of the prediction", np.shape(preds))
    loss = loss / len(tt_loader)
    accu = accu / len(tt_loader) / 1000
    print("Server model loss: %.4f and accuracy: %.4f" % (loss, accu))
    return loss, accu, preds, taf




def train_with_conf(conf):
    model_mom = "../exp_data/"

    conf.folder_name = "cifar10"
    conf.dir_name = "version_1"

    model_dir = model_mom + "%s/%s/" % (conf.folder_name, conf.dir_name)

    stat_use = model_dir + "/stat.obj"
    if os.path.exists(stat_use):
        if conf.use_local_id == 0:
            content = pickle.load(open(stat_use, "rb"))
    else:
        content = {}
        content["server_loss"] = []
        content["server_accu"] = []

        #content["server_train_loss"] = []
        #content["server_train_accu"] = []

    tr_loader = gsc.get_cifar10_dataset(conf, transform_apply=True)
    tt_loader = gsc.get_cifar10_test_dataset(conf.batch_size)

    print("GPU availability", torch.cuda.is_available())

    seed_use = np.random.randint(0, 100000, 1)[0]
    conf.random_state = np.random.RandomState(seed_use)

    print("The used learning rate", conf.lr)
    print("The seed", seed_use)

    if conf.round == 0:
        conf.random_state = np.random.RandomState(seed_use)
        mnist_utils.seed_everything(seed_use)

    model_path = model_dir + "/communication_round_%03d/" % conf.round
    data_mom = "../data/version%02d/communication_round_%03d/" % (conf.version, conf.round)

    if not os.path.exists(data_mom):
        os.makedirs(data_mom, exist_ok=True)

    if conf.use_local_id == 0:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    conf.model_dir = model_path

    if conf.round == 0:
        exist_model = mnist_utils.initial_model(conf)
    else:
        try:
            exist_model = torch.load(
                model_dir + "/communication_round_%03d/" % (conf.round - 1) + "/aggregated_model.pt",
                map_location=device)
        except FileNotFoundError:
            print("Failed at round ", conf.round - 1)
            return []
    time_init = time.time()
    _model = run_train(conf, tr_loader, tt_loader,
                       exist_model)

    print("finish training model time", time.time() - time_init)

    while True:
        if np.sum([os.path.isfile(conf.model_dir + "/client_id_%02d.pt" % j) for j in
                   range(conf.n_clients)]) == conf.n_clients:

            if conf.use_local_id == 0:
                time.sleep(10)

                #### We validate the server on the test set ####
                #tt_loss, tt_accu, preds, taf = run_server(conf)
                tt_loss, tt_accu, preds, taf, tr_loss, tr_accu, tr_preds, tr_taf = run_server(conf)
                content["server_loss"].append(tt_loss)
                content["server_accu"].append(tt_accu)

                savetxt(data_mom+"loss_accu.csv", np.array([tt_loss, tt_accu]),delimiter=',')
                savetxt(data_mom+"taf.csv", taf,delimiter=',')
                savetxt(data_mom+"server_pred.csv", preds, delimiter=',')
                ####

                #### We validate the server on the training set ####
                #content["server_train_loss"].append(tr_loss)
                #content["server_train_accu"].append(tr_accu)

                savetxt(data_mom + "loss_accu_train.csv", np.array([tr_loss, tr_accu]), delimiter=',')
                savetxt(data_mom + "taf_train.csv", tr_taf, delimiter=',')
                savetxt(data_mom + "server_pred_train.csv", tr_preds, delimiter=',')
                time.sleep(5)

                print('SUCCESSFUL ROUND')

                with open(stat_use, "wb") as f:
                    pickle.dump(content, f)
                print("Finish getting the server model at round", conf.round)
                break
            else:
                break


   # _, _, preds, _ = check_test_accuracy(_model, conf)
   # savetxt(data_mom + "client%02d.csv" % conf.use_local_id, preds, delimiter=',')


    del exist_model
    del _model
    if conf.round >= 4 and conf.use_local_id == 0:
        path2remove(model_dir + "/communication_round_%03d/" % (conf.round - 4))


def path2remove(model_dir):
    if "communication_round_" in model_dir:
        if os.path.exists(model_dir):
            sub_dir = [v for v in os.listdir(model_dir) if ".pt" in v]
            for v in sub_dir:
                if os.path.isfile(model_dir + v):
                    os.remove(model_dir + v)
            os.removedirs(model_dir)


if __name__ == "__main__":
    conf = configs.give_args()
    conf.lr = float(conf.lr)
    if conf.round == 0:
        for arg in vars(conf):
            print(arg, getattr(conf, arg))
    train_with_conf(conf)













