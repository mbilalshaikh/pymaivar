import warnings
import numpy as np
import os
import json
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn.functional as F
# import torchnet as tnt
# from torchnet.engine import Engine
from torch.backends import cudnn
from resnet import resnet
# from torchnet.engine import Engine
from utils import cast, data_parallel, print_tensor_dict
from torch.utils.data import DataLoader
from dataset_utils import MyDataset
from resnet import resnet
import torchvision.models as models

from absl import flags
from absl import logging

from utils2 import featureNormalize, llgc_meta, linear_test_accuracy
# from triplet import triplet_loss_adapted_from_tf
cudnn.benchmark = True
FLAGS = flags.FLAGS
warnings.filterwarnings("ignore")


def get_log_name(prefix=''):
    if 'face' in FLAGS.lt:
        path = FLAGS.lt + prefix + '_logs/' + FLAGS.d + '/' + FLAGS.n + '/'
    elif 'contr' in FLAGS.lt:
        path = prefix + 'logs_contarst/' + FLAGS.d + '/' + FLAGS.n + '/'
    # elif 'xnt' in FLAGS.lt:
    #     path = prefix + 'logs_xnt/' + FLAGS.d + '/' + FLAGS.n + '/'
    else:
        path = prefix + 'logs/' + FLAGS.d + '/' + FLAGS.n + '/'
    so = 'so-' if FLAGS.so else ''
    # log_name = FLAGS.lt + '-' if 'tfa' in FLAGS.lt else ''
    log_name = so + str(FLAGS.nl)  # + '-' # + FLAGS.m # + '-'  # + str(FLAGS.es)
    # if FLAGS.m == 'llgc':
    #     log_name = log_name + '-' + FLAGS.m + '-' + str(FLAGS.s)
    if FLAGS.m is not None:
        log_name = log_name + '-' + FLAGS.m
    # if FLAGS.bs != 100:
    log_name += '-' + str(FLAGS.bs)
    # ag = '-ag' if FLAGS.ag else ''
    # if FLAGS.mixup:
    #     path += 'mixup/'
    # elif FLAGS.autoaugment:
    #     path += 'autoaugment/'
    # elif 'tfa' in FLAGS.lt:
    #     path = path + FLAGS.lt + '/'
    weights =   '-w' if FLAGS.w else '-nw'
    # if FLAGS.wres:
    #     weights += FLAGS.wres
    # l2 = '-l2' if FLAGS.l2 else ''
    log_name += '-' + FLAGS.lbl + '-' + FLAGS.opt.lower() + weights # + l2  # + os.uname()[1]

    return path, log_name


def get_linear_clf():
    if FLAGS.lbl == 'KNN':
        return KNeighborsClassifier(metric='euclidean', n_neighbors=9)
    if FLAGS.lbl == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()


def select_top_k(imgs, lbls, scores, orig_lbls=None, k=4000):
    sampled_images = []
    sampled_labels = []
    orig_lbls_selected = []
    number_classes = np.unique(lbls)
    per_class = k // len(number_classes)
    args = np.argsort(scores)
    indices = []
    for key in number_classes:
        selected = 0
        for index in args:
            if lbls[index] == key:
                sampled_images.append(imgs[index])
                sampled_labels.append(lbls[index])
                indices.append(index)
                if orig_lbls is not None:
                    orig_lbls_selected.append(orig_lbls[index])
                selected += 1
                if per_class == selected:
                    break
    if orig_lbls is not None:
        orig_lbls_selected = np.array(orig_lbls_selected)
        sampled_labels = np.array(sampled_labels)
        print('selection accuracy  {:.2f}%'.format(accuracy_score(orig_lbls_selected, sampled_labels) * 100.))
    return np.array(sampled_images), sampled_labels, indices


def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch >= 60:
        lr = 0.0002
    if epoch >= 120:
        lr = 0.0004
    if epoch >= 160:
        lr = 0.00008

    return lr


def  lr_exp_decay(epoch, lr):
    k = 0.1
    return FLAGS.lr * np.exp(-k*epoch)


def get_network(arch=FLAGS.n, input_shape=(32, 32, 3), weight=False, ii=FLAGS.ii):
    params = None
    if 'dense' in arch:
        initial_iter = 300 if ii == 0 else ii
        conv_base = models.densenet161(pretrained=weight)
    elif 'res18' in arch:
        initial_iter = 400 if ii == 0 else ii
        conv_base = conv_base = models.resnet18(pretrained=weight)
    elif 'inception' in arch:
        initial_iter = 400 if ii == 0 else ii
        conv_base = models.inception_v3(pretrained=weight)
    elif 'mobilenet' in arch:
        initial_iter = 400 if ii == 0 else ii
        conv_base =  models.mobilenet_v2(pretrained=weight)
    elif 'vgg16' in arch:
        initial_iter = 400 if ii == 0 else ii
        conv_base = models.vgg16(pretrained=weight)
    elif arch in 'wrn50_2':
        conv_base = models.wide_resnet50_2(pretrained=weight)
    else:
        print()
        initial_iter = 400 if ii == 0 else ii
        conv_base, params = resnet(FLAGS.depth, FLAGS.width, FLAGS.nc)

    # initial_iter = 400 if ii == 0 else ii
    FLAGS.ii = initial_iter
    # print_tensor_dict(params)
    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)
    return conv_base, params


def get_optimizer(opt, lr, params,  wd=FLAGS.wd):
    from torch.optim import SGD, Adam, RMSprop
    print('creating optimizer with lr = ', lr)
    if opt.lower() == 'adam':
        return Adam([v for v in params.values() if v.requires_grad], lr)
    elif opt.lower() == 'adamw':
        return Adam([v for v in params.values() if v.requires_grad], lr, weight_decay=wd)
    elif opt.lower() in ['sgd', 'sgdn']:
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9)
    else:
        # print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=wd)


def get_model(size, channels, arch=FLAGS.n #net name#
            , weights=False, path='', data=None):
    f, params  = get_network(arch=arch, input_shape=(size, size, channels), weight=weights)
    # resnet(28, 2.0, 10)

    # optimizer = get_optimizer(FLAGS.opt, FLAGS.lr)
    if data:
        test_loader = create_dataloaders(data.test.images, data.test.labels, train=False)
    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    def h(sample):
        inputs = cast(sample[0], FLAGS.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(FLAGS.ngpu))).float()
        return F.cross_entropy(y, targets), y

    def log(t, state):
        # torch.save(dict(params=params, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
        #            os.path.join(path, 'model.pt7'))
        z = {**t}
        with open(os.path.join(path + '-' + 'log.txt'), 'a') as flog:
            flog.write('json_stats: '+ json.dumps(z) + '\n')  # + json.dumps(z)
        print(t)

    epoch = 0
    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(loss)
        # if state['train']:
        #     state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        # state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if str(epoch) in FLAGS.epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = get_optimizer(FLAGS.opt, lr * FLAGS.lr_decay_ratio, params)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        test_acc = 0.
        if data:
            with torch.no_grad():
                engine.test(h, test_loader)

            test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            # "epoch": state['epoch'],
            # "num_classes": num_classes,
            # "n_parameters": n_parameters,
            "train_time": train_time,
            # "test_time": timer_test.value(),
        }, state))
        print('==> epochs:  (%d/%d), ' %
              (state['epoch'], FLAGS.ii))  # test_acc

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    # engine.train(h, train_loader, opt.epochs, optimizer)
    return engine, h, f, params


def get_network_output(model, input_images, normalize=True, v=0, bs=FLAGS.bs):
    feat = model.predict(input_images, verbose=v, batch_size=bs)
    if normalize:
        feat, _, _ = featureNormalize(feat)
    return feat


def compute_supervised_accuracy(model, imgs, lbls, ret_labels=False, v=0, bs=FLAGS.bs):
    accuracy = model.evaluate(imgs, lbls, verbose=v, batch_size=bs)
    accuracy = np.round(accuracy[1] * 100., 2)
    if ret_labels:
        labels = model.predict(imgs, verbose=v, batch_size=bs)
        return accuracy, labels
    return accuracy


def compute_linear_clf_accuracy(model, l_imgs, l_lbls, t_imgs, t_lbls, ret_labels=False, norm=True):
    if l_lbls.ndim > 1:
        l_lbls = np.argmax(l_lbls, 1)
        t_lbls = np.argmax(t_lbls, 1)
    labels, accuracy = linear_test_accuracy(get_network_output(model, l_imgs, normalize=norm), l_lbls,
                                            get_network_output(model, t_imgs, normalize=norm), t_lbls, FLAGS.lbl,
                                            verbose=False)
    if ret_labels:
        return accuracy, labels
    return accuracy


def calculate_accuracy(model, l_imgs, l_lbls, t_imgs, t_lbls, norm=True, ret_labels=False):
    if FLAGS.so:
        return compute_supervised_accuracy(model, t_imgs, t_lbls, ret_labels)
    else:
        return compute_linear_clf_accuracy(model, l_imgs, l_lbls, t_imgs, t_lbls, ret_labels, norm)


def calculate_torch_acc(x, y, net):
    correct = 0
    total = 0
    test_loader = create_dataloaders(x, y, train=False)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    return 100 * correct / total

def log_accuracy(model, h, f, dso, when='', norm=True, unlab=True, batch_size=FLAGS.bs, supervised_only=True):
    acc_type = FLAGS.lbl
    linear_acc = -1
    if FLAGS.semi and unlab:
        labeled_imgs = dso.train.labeled_ds.images
        test_imgs = dso.train.unlabeled_ds.images
        if not supervised_only:
            linear_acc = compute_linear_clf_accuracy(model, labeled_imgs, dso.train.labeled_ds.labels,
                                                     test_imgs, dso.train.unlabeled_ds.labels,
                                                     norm=norm)
        if FLAGS.so:
            test_loader = create_dataloaders(test_imgs, dso.test.labels, train=False)
            with torch.no_grad():
                model.test(h, test_loader)

            ac = calculate_torch_acc(test_imgs, dso.test.labels, f)
            # ac = compute_supervised_accuracy(model, test_imgs, dso.train.unlabeled_ds.labels)
            logging.info('={} Unlab accuracy={} {:.2f}% supervised {:.2f} %'.format(when, acc_type, linear_acc, ac))
        else:
            logging.info('======{} Unlab  accuracy========= {} {:.2f} %'.format(when, acc_type, linear_acc))

    labeled_imgs = dso.train.labeled_ds.images
    test_imgs = dso.test.images
    if not supervised_only:
        linear_acc = compute_linear_clf_accuracy(model, labeled_imgs, dso.train.labeled_ds.labels,
                                                 test_imgs, dso.test.labels, norm=norm)
    if FLAGS.so:
        # test_loader = create_dataloaders(list(test_imgs), list(dso.test.labels), train=False)
        print('!!! test shapes', test_imgs.shape, dso.test.labels.shape)
        print('!!! test dtypes',  type(test_imgs[0,0,0,1]), type(dso.test.labels[0]), test_imgs[0,0,0,1])
        tds = tnt.dataset.TensorDataset([test_imgs, dso.test.labels])
        test_loader =  tds.parallel(batch_size=FLAGS.bs, num_workers=4, shuffle=True)
        with torch.no_grad():
            model.test(h, test_loader)
        ac = classacc.value()[0]
        # ac = compute_supervised_accuracy(model, test_imgs, dso.test.labels)
        ret_ac = ac
        logging.info('{}  Test accuracy: {}   {:.2f} % supervised  {:.2f} %'.format(when, acc_type, linear_acc, ac))
    else:
        logging.info('======={} {} Test accuracy=========  {:.2f} %'.format(when, acc_type, linear_acc))
        ret_ac = linear_acc

    return ret_ac


def get_callbacks(model, images, labels, test_images, test_labels, train_iter, print_test_error, verb, print_lr=False):
    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    evaluate = KnnEvaluator(model, (images, labels), (test_images, test_labels), test_every=FLAGS.vf,
                            name=FLAGS.lbl)
    nan = callbacks.TerminateOnNaN()
    calls = [nan]  # callbacks=[csa])
    if FLAGS.lrd == 'simple':
        calls.append(callbacks.LearningRateScheduler(adjust_learning_rate))
    elif FLAGS.lrd == 'lc':
        decay_steps = 10000
        lr_decayed = tf.keras.experimental.LinearCosineDecay(FLAGS.lr, decay_steps)
        lr = callbacks.LearningRateScheduler(lr_decayed)
        calls.append(lr)
    elif FLAGS.lrd == 'cosine':
        decay_steps = 1000
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            FLAGS.lr, decay_steps)
        lr = callbacks.LearningRateScheduler(lr_decayed_fn)
        calls.append(lr)
    elif FLAGS.lrd == 'ca':
        from cosine_annealing import CosineAnnealingScheduler
        csa = CosineAnnealingScheduler(T_max=train_iter, eta_max=0.05, eta_min=4e-4)
        calls.append(csa)
    elif FLAGS.lrd == 'exp':
        calls.append(callbacks.LearningRateScheduler(lr_exp_decay))

    if FLAGS.early_stop:
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        calls.append(early_stop)
        print_test_error = True  # temporary for validation data and val_loss
    if FLAGS.csv:
        log_dir, log_name = get_log_name()
        csv = tf.keras.callbacks.CSVLogger(log_dir + FLAGS.pre + '/' + log_name + '.csv')
        calls.append(csv)
    if verb >= 2:
        tf_version = int(tf.__version__.split('.')[0])
        if tf_version > 1:
            import tensorflow_addons as tfa
            tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
            calls.append(tqdm_callback)
            verb = 0
        else:
            verb = 1
    if print_lr:
        from tensorflow.keras.callbacks import Callback

        class PrintLr(Callback):
            def on_epoch_end(self, epoch):
                lr = self.model.optimizer.lr
                decay = self.model.optimizer.decay
                iterations = self.model.optimizer.iterations
                lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
                print(K.eval(lr_with_decay))
        calls.append(PrintLr)

    if print_test_error:
        if not FLAGS.so:
            calls.append(evaluate)
    return calls, print_test_error, verb


def create_dataloaders(images, labels, batch_size= FLAGS.bs, train=True):

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    val = not train
    datasets = MyDataset(images, labels, test_data=val)
    data_loader = DataLoader(datasets, batch_size, shuffle=train,
               num_workers=FLAGS.nthread, pin_memory=torch.cuda.is_available())
    return  data_loader


def do_training(model, h, params, images, labels,   test_images, test_labels, train_iter=FLAGS.ii, print_test_error=True,
                verb=FLAGS.verbose, batch_size=FLAGS.bs, ):
    train_loader =  create_dataloaders(images, labels, batch_size)
    optimizer = get_optimizer(FLAGS.opt, FLAGS.lr, params)
    model.train(h, train_loader, train_iter, optimizer)


def start_training(model, h, params, dso, initial_iter=FLAGS.ii, verb=FLAGS.verbose):
    do_training(model, h, params, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images, dso.test.labels,
                initial_iter, print_test_error=True, verb=verb)


def get_confidence(model, mdso, unlabeled_imgs, unlabeled_lbls,alpha=0.9, sigma=1.2, lbls=None, labeled=0):
    # te_acc_nn = 0
    # true_test_labels = unlabeled_lbls
    train_labels = mdso.train.labeled_ds.labels
    if unlabeled_lbls.ndim > 1:
        train_labels = np.argmax(train_labels, 1)
        unlabeled_lbls = np.argmax(unlabeled_lbls, 1)
    # LDA or other linear models
    if FLAGS.lbl != 'knn':
        clf = get_linear_clf()
        test_image_feat = get_network_output(model, unlabeled_imgs)
        current_labeled_train_feat = get_network_output(model, mdso.train.labeled_ds.images)
        clf.fit(current_labeled_train_feat, train_labels)
        pred_lbls = clf.predict(test_image_feat)
        if FLAGS.blb == 'lda':
            calc_score = clf.decision_function(test_image_feat)
        else:
            calc_score = clf.predict_proba(test_image_feat)

        calc_score = np.max(calc_score, 1)
        calc_score = calc_score * -1.  # negate probs for same notion as distance
    elif FLAGS.so:
        test_image_feat = model.predict(unlabeled_imgs)
        pred_lbls = np.argmax(test_image_feat, 1)
        calc_score = np.max(test_image_feat, 1)
        calc_score = calc_score * -1.  # negate probs for same notion as distance
    # elif FLAGS.m == 'llgc':
    #     unlabeled_feats = get_network_output(model, unlabeled_imgs)
    #     pred_lbls, knn_err = linear_test_accuracy(get_network_output(model, mdso.train.labeled_ds.images),
    #                                        train_labels, unlabeled_feats, unlabeled_lbls)
    #
    #     y_input = np.concatenate([lbls, pred_lbls], 0)  # lbls llgc_meta  label_spreading llgc_meta_numba [:, 0]
    #     lgc_lbls, llgc_prob, llgc_err = llgc_meta(dm1=unlabeled_feats,
    #                                               dm0=get_network_output(model, mdso.train.labeled_ds.images),
    #                                               alpha=alpha, sigma=sigma, Y_input=y_input, lbls_all=True,
    #                                               Y_original=np.concatenate((predicted_original_lbls, unlabeled_lbls)),
    #                                               verbose=False)
    #     pred_lbls = lgc_lbls[labeled:]
    #     calc_score = llgc_prob[labeled:]
    #     calc_score = calc_score * -1.  # negate probs for same notion as distance
    else:  # default to KNN with k=1 distance as confidence score
        pred_lbls = []
        calc_score = []
        k = 1
        test_image_feat = get_network_output(model, unlabeled_imgs)
        current_labeled_train_feat = get_network_output(model, mdso.train.labeled_ds.images)
        for j in range(len(test_image_feat)):
            search_feat = np.expand_dims(test_image_feat[j], 0)
            # calculate the sqeuclidean similarity and sort
            dist = cdist(current_labeled_train_feat, search_feat, 'sqeuclidean')
            rank = np.argsort(dist.ravel())
            pred_lbls.append(train_labels[rank[:k]])
            calc_score.append(dist[rank[0]])

    pred_lbls = np.array(pred_lbls)
    labels = pred_lbls.squeeze()
    print('predicted accuracy {:.2f} %'.format(accuracy_score(unlabeled_lbls, pred_lbls)*100.))
    calc_score = np.array(calc_score)
    score = calc_score.squeeze()
    return labels, score


def self_learning_modular(model, mdso, labeled=FLAGS.nl, num_iterations=FLAGS.i, percentile=FLAGS.sp, network_updates=FLAGS.mti):
    # predicted_labeled_imgs = mdso.train.labeled_ds.images
    predicted_labeled_lbls = mdso.train.labeled_ds.labels
    predicted_original_lbls = mdso.train.labeled_ds.labels
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # predicted_labeled_imgs_inds = []
    n_label = labeled
    template = 'overall accuracy of predicted total {} is {:.2f}%'
    template2 = 'total selected based on percentile {} having accuracy {:.2f}%'
    template3 = "Labeled= {},LLGC= {} selection={}% iterations= {}"
    template4 = 'Length of predicted {}, labeled  {},  unlabeled {}'
    pred_str = 'supervised ' if FLAGS.so else FLAGS.lbl
    percentile = FLAGS.sp if percentile == 0. else percentile
    logging.info(template3.format(labeled, False, 100 * percentile, num_iterations))
    for i in range(num_iterations):
        print('=============== iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        adaptive_network_updates = 2000 * FLAGS.bs / len(lbls)
        do_training(model, imgs, lbls, mdso.test.images, mdso.test.labels, network_updates, verb=FLAGS.verbose)

        pred_lbls, calc_score = get_confidence(model, mdso, unlabeled_imgs, unlabeled_lbls)
        unlab_acc = accuracy_score(unlabeled_lbls, pred_lbls) * 100.
        print('==>', pred_str, ' predicting for  lbls ', unlabeled_lbls.shape, '=', np.round(unlab_acc, 2))
        to_select = int(len(unlabeled_lbls) * percentile)
        selected_imgs, selected_lbls, ind = select_top_k(unlabeled_imgs, pred_lbls, calc_score, unlabeled_lbls, to_select)
        # ind = np.argsort(calc_score)
        total_selected = len(ind)  # int(len(unlabeled_lbls) * percentile)
        # ind = ind[:total_selected]
        # ind = ind.squeeze()
        # predicted_labeled_imgs_inds.append(ind)

        original_lbls_selected = unlabeled_lbls[ind]
        select_acc = accuracy_score(original_lbls_selected, selected_lbls) * 100.  # pred_lbls[ind]
        print(template2.format(total_selected, select_acc))
        # selection based on top k func
        new_labeled_imgs = unlabeled_imgs[ind]  # selected_imgs
        # new_labeled_imgs = new_labeled_imgs.squeeze()
        new_labeled_lbls = pred_lbls[ind]  # [ind,0]
        # new_labeled_lbls = new_labeled_lbls.squeeze()
        # merging new labeled for next loop iteration
        imgs = np.concatenate([imgs, new_labeled_imgs], axis=0)
        lbls = np.concatenate([lbls, new_labeled_lbls], axis=0)
        labeled = len(imgs)
        predicted_labeled_lbls = np.concatenate((predicted_labeled_lbls, new_labeled_lbls))
        predicted_original_lbls = np.concatenate((predicted_original_lbls, original_lbls_selected))
        # overall accuracy of predicted
        acc = accuracy_score(predicted_original_lbls[n_label:], predicted_labeled_lbls[n_label:]) * 100.
        print(template.format(len(predicted_labeled_lbls[n_label:]), acc))
        test_acc = calculate_accuracy(model, mdso.train.labeled_ds.images, mdso.train.labeled_ds.labels,
                                      mdso.test.images, mdso.test.labels)
        # remove selected
        unlabeled_imgs = np.delete(unlabeled_imgs, ind, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, ind, 0)
        print(template4.format(labeled - n_label, labeled, len(unlabeled_lbls)))
        logging.info("{},{:.2f},{:.2f}".format(i+1, unlab_acc, test_acc))  # ith,unlab,test accuracy
        #
        # start_training(model, mdso, 200)
        # logging.info('fine tuning on n-label after each meta iteration')
        # calculate_accuracy(model, mdso, 'after each meta iteration fine tuning')
    return imgs, lbls, predicted_original_lbls


def meta_learning(model, dso, meta_iterations=FLAGS.i, mti=FLAGS.mti, selection_percentile=FLAGS.sp):
    if 'fix' in FLAGS.m:
        print('fixed modular self learning......')
        self_learning_fixed_k(model, dso, labeled=FLAGS.nl, num_iterations=meta_iterations,
                              percentile=selection_percentile, network_updates=mti)
    else:
        if 'm' in FLAGS.m:
            print('modular self learning......')
            self_learning_modular(model, dso, labeled=FLAGS.nl, num_iterations=meta_iterations,
                                  percentile=selection_percentile, network_updates=mti)
        else:
            self_learning(model, dso, labeled=FLAGS.nl, num_iterations=meta_iterations,
                          percentile=selection_percentile, network_updates=mti)


def save_embedding(model, name, dso):
    from Visualizations import t_sne_vis, load_label_names
    if 'cifar10' in FLAGS.d:
        names = load_label_names()
    else:
        names = range(FLAGS.nc)
    eye_lbl = dso.test.labels
    test_lbls = [names[i] for i in eye_lbl]
    if FLAGS.so:
        t_embed, t_ax = t_sne_vis(get_network_embeddings(model, dso.test.images), dso.test.labels, 2, test_lbls)
    else:
        t_embed, t_ax = t_sne_vis(get_network_output(model, dso.test.images), dso.test.labels, 2, test_lbls)
    np.savez_compressed('./embed2/'+name+'.npz', embed=t_embed, labels=dso.test.labels)


def save_weights(model, dso=None, include_top=True):
    post = ''
    if dso and FLAGS.so:
        ac = np.round(model.evaluate(dso.test.images, dso.test.labels, verbose=0)[1], 4) * 100.
    else:
        ac = -1
    if FLAGS.so:
        post = "-so"
    else:
        post = "-apn"
    if include_top:
        model.save_weights("./weights/{}-{}-ep-{}-acc-{:.2f}-weights"+post+".h5".format(FLAGS.n, FLAGS.d, FLAGS.ii, ac))
    else:
        extracted_model = tf.keras.Model(model.input, model.layers[-1].input)
        extracted_model.save_weights("./weights/{}-{}-ep-{}-acc-{:.2f}-weights_notop"+post+".h5".format(FLAGS.n, FLAGS.d,
                                                                                                FLAGS.ii, ac))


def flatten_list(t):
    flat_list = [item for sublist in t for item in sublist]
    flat_list = np.array(flat_list)
    return flat_list