import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    checkpoint_dir = "checkpoints/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 公共路径
    common_path = "{}/{}/{}/{}/{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        "{}_{}_{}_{}".format(
            args["target_domain"],
            args["prefix"],
            args["seed"],
            args["convnet_type"]
        )
    )
    # 日志文件名
    logfilename = "logs/{}".format(common_path)

    # 检查点文件名
    checkpoint_file = "checkpoints/{}.pth".format(common_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["source_domain"],
        args["target_domain"]
    )
    topk = args["topk"]
    topk_key = f"top{topk}"

    model = factory.get_model(args["model_name"], args)

    model.checkpoint_file = checkpoint_file
    if args["resume"] and os.path.exists(checkpoint_file):
        model.load_checkpoint(checkpoint_file)
        cur_task = model._cur_task + 1
        logging.info(f"Resumed training from checkpoint: {checkpoint_file}")
    else:
        logging.info("Training from scratch")
        cur_task = 0

    source_domains = sorted(args["source_domain"])
    target_domain = args["target_domain"]
    domains = sorted(source_domains + [target_domain]) if target_domain not in source_domains else sorted(
        source_domains)
    cnn_curve, nme_curve = {}, {}
    cnn_matrix, nme_matrix = {}, {}
    for domain in domains:
        cnn_curve.setdefault(domain, {"top1": [], topk_key: []})
        nme_curve.setdefault(domain, {"top1": [], topk_key: []})
        cnn_matrix.setdefault(domain, [])
        nme_matrix.setdefault(domain, [])

    for task in range(cur_task, data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        model.save_checkpoint(checkpoint_file)

        if nme_accy is not None:
            for domain in domains:  # 遍历每个 domain
                logging.info("CNN [{}]: {}".format(domain, cnn_accy[domain]["grouped"]))
                logging.info("NME [{}]: {}".format(domain, nme_accy[domain]["grouped"]))
                cnn_keys = [key for key in cnn_accy[domain]["grouped"].keys() if '-' in key]
                cnn_keys_sorted = sorted(cnn_keys)
                cnn_values = [cnn_accy[domain]["grouped"][key] for key in cnn_keys_sorted]
                cnn_matrix[domain].append(cnn_values)

                nme_keys = [key for key in nme_accy[domain]["grouped"].keys() if '-' in key]
                nme_keys_sorted = sorted(nme_keys)
                nme_values = [nme_accy[domain]["grouped"][key] for key in nme_keys_sorted]
                nme_matrix[domain].append(nme_values)

                cnn_curve[domain]["top1"].append(cnn_accy[domain]["top1"])
                cnn_curve[domain][topk_key].append(cnn_accy[domain][topk_key])

                nme_curve[domain]["top1"].append(nme_accy[domain]["top1"])
                nme_curve[domain][topk_key].append(nme_accy[domain][topk_key])

                logging.info("CNN [{}] top1 curve: {}".format(domain, cnn_curve[domain]["top1"]))
                logging.info("CNN [{}] {} curve: {}".format(domain, topk_key, cnn_curve[domain][topk_key]))
                logging.info("NME [{}] top1 curve: {}".format(domain, nme_curve[domain]["top1"]))
                logging.info("NME [{}] {} curve: {}".format(domain, topk_key, nme_curve[domain][topk_key]))

                print('Average Accuracy [{}] (CNN):'.format(domain),
                      sum(cnn_curve[domain]["top1"]) / len(cnn_curve[domain]["top1"]))
                print('Average Accuracy [{}] (NME):'.format(domain),
                      sum(nme_curve[domain]["top1"]) / len(nme_curve[domain]["top1"]))

                logging.info("Average Accuracy [{}] (CNN): {}".format(domain, sum(cnn_curve[domain]["top1"]) / len(
                    cnn_curve[domain]["top1"])))
                logging.info("Average Accuracy [{}] (NME): {}".format(domain, sum(nme_curve[domain]["top1"]) / len(
                    nme_curve[domain]["top1"])))

            target_nme_acc = nme_accy[target_domain]["top1"]
            source_nme_acc = sum([nme_accy[domain]["top1"] for domain in source_domains]) / len(source_domains)
            harmonic_mean = 2 * source_nme_acc * target_nme_acc / (source_nme_acc + target_nme_acc)
            logging.info("NME Harmonic Mean : {:.4f}".format(harmonic_mean))
            macro_nme_acc = sum([nme_accy[domain]["top1"] for domain in domains]) / len(domains)
            logging.info("NME Macro Mean : {:.4f}".format(macro_nme_acc))

            target_cnn_acc = cnn_accy[target_domain]["top1"]
            source_cnn_acc = sum([cnn_accy[domain]["top1"] for domain in source_domains]) / len(source_domains)
            harmonic_mean = 2 * source_cnn_acc * target_cnn_acc / (source_cnn_acc + target_cnn_acc)
            logging.info("CNN Harmonic Mean : {:.4f}".format(harmonic_mean))

            macro_cnn_acc = sum([cnn_accy[domain]["top1"] for domain in domains]) / len(domains)
            logging.info("CNN Macro Mean : {:.4f}".format(macro_cnn_acc))
        else:
            for domain in domains:  # 遍历每个 domain
                logging.info("CNN [{}]: {}".format(domain, cnn_accy[domain]["grouped"]))

                cnn_keys = [key for key in cnn_accy[domain]["grouped"].keys() if '-' in key]
                cnn_keys_sorted = sorted(cnn_keys)
                cnn_values = [cnn_accy[domain]["grouped"][key] for key in cnn_keys_sorted]
                cnn_matrix[domain].append(cnn_values)

                cnn_curve[domain]["top1"].append(cnn_accy[domain]["top1"])
                cnn_curve[domain][topk_key].append(cnn_accy[domain][topk_key])

                logging.info("CNN [{}] top1 curve: {}".format(domain, cnn_curve[domain]["top1"]))
                logging.info("CNN [{}] {} curve: {}".format(domain, topk_key, cnn_curve[domain][topk_key]))

                logging.info("Average Accuracy [{}] (CNN): {}".format(domain, sum(cnn_curve[domain]["top1"]) / len(
                    cnn_curve[domain]["top1"])))

            target_cnn_acc = cnn_accy[target_domain]["top1"]
            source_cnn_acc = sum([cnn_accy[domain]["top1"] for domain in source_domains]) / len(source_domains)
            harmonic_mean = 2 * source_cnn_acc * target_cnn_acc / (source_cnn_acc + target_cnn_acc)
            logging.info("CNN Harmonic Mean : {:.4f}".format(harmonic_mean))

            macro_cnn_acc = sum([cnn_accy[domain]["top1"] for domain in domains]) / len(domains)
            logging.info("CNN Macro Mean : {:.4f}".format(macro_cnn_acc))

    if len(cnn_matrix) > 0:
        for domain in domains:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix[domain]):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])

            print('Accuracy Matrix (CNN) [{}]:'.format(domain))
            print(np_acctable)
            print('Forgetting (CNN) [{}]:'.format(domain), forgetting)
            logging.info('Forgetting (CNN) [{}]: {}'.format(domain, forgetting))

    if nme_accy is not None:
        for domain in domains:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix[domain]):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])

            logging.info('Forgetting (NME) [{}]: {}'.format(domain, forgetting))

    logging.getLogger().handlers[0].setFormatter(logging.Formatter('%(message)s'))
    for domain in domains:
        logging.info("'{}_{}_{}_top1':{},".format(args["increment"], args["target_domain"].lower(), domain.lower(),
                                                  cnn_curve[domain]["top1"]))
        logging.info("'{}_{}_{}_{}':{},".format(
            args["increment"],
            args["target_domain"].lower(),
            domain.lower(),
            topk_key,
            cnn_curve[domain][topk_key]
        ))
        model.save_checkpoint(checkpoint_file)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
