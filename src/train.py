from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
# import torch
# import torch.utils.data
# from torchvision.transforms import transforms as T
import paddle
from paddle.io import DataLoader
import paddle.vision.transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
# from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

def main(opt):
    # torch.manual_seed(opt.seed)
    # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    paddle.seed(opt.seed)
    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    opt.device = paddle.get_device()

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    
    start_epoch = 0

    # Get dataloader

    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        use_shared_memory=False,
        drop_last=True
    )
    print('Starting training...')
    Trainer = train_factory[opt.task]
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # optimizer = paddle.optimizer.Adam(learning_rate=opt.lr, parameters=model.parameters()) # 这句代码的作用纯粹是为了传个参数，

    # trainer = Trainer(opt, model, optimizer)
    trainer = Trainer(opt, model)
    optimizer = trainer.optimizer  # 见base_trainer.py
    id_classifier = trainer.loss.classifier # 见base_trainer.py
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    if 'fairmot_hrnet_w18' in opt.load_model:
        model = load_model(model, opt.load_model)
    elif opt.load_model != '':
        model, optimizer, start_epoch, id_classifier = load_model(
            model, opt.load_model, trainer.optimizer, trainer.loss.classifier, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pdparams'.format(mark)),
                       epoch, model, optimizer, id_classifier)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pdparams'),
                       epoch, model, optimizer, id_classifier)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pdparams'.format(epoch)),
                       epoch, model, optimizer, id_classifier)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            optimizer.set_lr(lr)
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pdparams'.format(epoch)),
                       epoch, model, optimizer, id_classifier)
    logger.close()


if __name__ == '__main__':
    # torch.cuda.set_device(2)
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, opt.gpus))
    main(opt)
