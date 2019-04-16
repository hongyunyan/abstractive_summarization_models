""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from utils import reorder_sequence, special_word_num
import time 



def coll_fn(data):
    source_lists, target_lists = unzip(data)

    sources = [[source for source in article_source] for article_source in list(source_lists)]
    targets = [[target for target in article_target] for article_target in list(target_lists)]

    assert all(sources) and all(targets)
    #现在是两层list，每个文章一个list，list里面包含 n个句子，  
    return sources, targets

#截取输入texts中的每篇文章在最大长度限制内的单词的小写组成的list的list
@curry
def tokenize(max_len, texts):
    return [[t.lower().split()[:max_len] for t in article ] for article in texts] 

#返回一个list,其中的每个list包含的是words_list中每个字在词典的标号,没有的词用unk表示。
def conver2id(unk, word2id, article_lists):
    word2id = defaultdict(lambda: unk, word2id)
    return [[[word2id[w] for w in sent] for sent in sent_lists ] for sent_lists in article_lists]

#输出一一对应的atricle和abstract的list
@curry
def prepro_fn(max_len, batch):
    sources, targets = batch
 
    sources = tokenize(max_len, sources)
    targets = tokenize(max_len, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch(unk, word2id, batch):
    #给没出现的字替换为unknown,然后对于原来输入的article和abs,生成读应的新版
    sources, targets = map(list, unzip(batch))
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    #还是一个sources的list的list
    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor


@curry
def batchify_fn(pad, start, end,  eoa, data, cuda=True):
    #我希望的这边的sources是一个大的list，里面包含了每个artical，然后每个article是一个list，包含了

    data.sort(key=lambda data:len(data[0]), reverse=True)
    sources, targets = list(map(list, unzip(data)))

    #删除targets中abs长度大于20的文本
    del_num = []
    for i in range(len(sources)):
        if (len(targets[i]) > 20):
            del_num.append(i)

    for num in reversed(del_num):
        del sources[num]
        del targets[num]

    source_article_len = [len(source) for source in sources]
    target_article_len = [len(target) for target in targets]

    source_sent = [] 
    for source in sources:
        for src in source:
            source_sent.append(src)

    src_sent_len = [len(src) for src in source_sent]

    tar_ins = []
    tar_outs = []
    for target in targets:
        sent_num = 0
        for tar in target:
            tar_ins.append([special_word_num + sent_num] + tar)
            tar_outs.append(tar + [end])
            sent_num += 1
        tar_outs[-1][-1] = eoa

    source = pad_batch_tensorize(source_sent, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(tar_outs, pad, cuda)

    fw_args = (source, source_article_len, src_sent_len, tar_in, target_article_len)
    loss_args = (target, )
    return fw_args, loss_args


#把每个batch的内容整理成source,target对,塞进queue中,并且每轮push完push一个epoch数字进queue
def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver') #这tm是个啥
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            #每个hyper_batch的元素都是source和target对
            indexes = list(range(0, len(hyper_batch), batch_size)) #根据batch_size来选择为每一轮batch的内容
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])

                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
