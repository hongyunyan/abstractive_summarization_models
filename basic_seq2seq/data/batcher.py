""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp


# Batching functions
def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, source_lists))
    targets = list(filter(bool, target_lists))

    sources = [" ".join(source) for source in sources]
    targets = [" ".join(target) for target in targets]

    assert all(sources) and all(targets)
    return sources, targets

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

#截取输入texts中的每篇文章在最大长度限制内的单词的小写组成的list的list
@curry
def tokenize(max_len, texts):
    return [t.lower().split()[:max_len] for t in texts]

#返回一个list,其中的每个list包含的是words_list中每个字在词典的标号,没有的词用unk表示。
def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

#输出一一对应的atricle和abstract的list
@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def prepro_fn_extract(max_src_len, max_src_num, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    return batch

@curry
def convert_batch(unk, word2id, batch):
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy(unk, word2id, batch):
    #给没出现的字替换为unknown,然后对于原来输入的article和abs,生成读应的新版
    sources, targets = map(list, unzip(batch))

    larger_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in larger_word2id:
                larger_word2id[word] = len(larger_word2id)  #把没有出现过的词加到新的词典里里面去，等于这个词典也是不会有unk的
    unk_sources = conver2id(unk, larger_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    unk_targets = conver2id(unk, larger_word2id, targets)
    targets = conver2id(unk, word2id, targets) 

    batch = list(zip(sources, unk_sources, targets, unk_targets))

    return batch

@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ff(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
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
def batchify_fn(pad, start, end, data, cuda=True):
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    tar_ins = [[start] + tgt for tgt in targets]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, unk_sources, targets, unk_targets = tuple(map(list, unzip(data)))
    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    unk_sources = [unk_src for unk_src in unk_sources]

    targets = [[start] + tgt for tgt in targets]  #为啥要加start和end啊
    unk_targets = [tgt + [end] for tgt in unk_targets]

    #根据最长的句子进行对其他句子的填充
    source = pad_batch_tensorize(sources, pad, cuda)   #(34,81)
    target = pad_batch_tensorize(targets, pad, cuda)   #(34,31)
    unk_target = pad_batch_tensorize(unk_targets, pad, cuda)
    unk_source = pad_batch_tensorize(unk_sources, pad, cuda)

    unk_source_size = unk_source.max().item() + 1 #30022

    fw_args = (source, src_lens, target, unk_source, unk_source_size) #这tm返回的是啥啊

    loss_args = (unk_target, ) #回一个有所有单词的target的list

    #我测试的10个train json 文件中一共有34个句子，因此这边的source第一维为34，第二维就看这句子有多长了
    return fw_args, loss_args


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
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
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
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
            hyper_batch.sort(key=self._sort_key) #为啥要排序啊？？？
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
