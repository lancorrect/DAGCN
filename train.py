import sys
import time
import os
import argparse
import logging
import numpy as np
import random
from sklearn import metrics
from time import strftime, localtime
import torch
from torch.utils.data import DataLoader
from transformers import BertModel
from model import GCNClassifier
from model_bert import GCNBertClassifier
from data_utils import build_tokenizer, build_embedding_matrix, SentenceDataset, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp
from trainer import Trainer

t_start = time.time()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

model_classes = {'dagcn': GCNClassifier,
                 'dagcnbert': GCNBertClassifier}

dataset_files = {
    'restaurant': {
        'train': './dataset/Restaurants_corenlp/train.json',
        'test': './dataset/Restaurants_corenlp/test.json',
    },
    'laptop': {
        'train': './dataset/Laptops_corenlp/train.json',
        'test': './dataset/Laptops_corenlp/test.json'
    },
    'twitter': {
        'train': './dataset/Tweets_corenlp/train.json',
        'test': './dataset/Tweets_corenlp/test.json',
    }
}

input_colses = {
    'dagcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj_reshape'],
    'dagcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix',
                    'src_mask', 'aspect_mask']
}

initializers = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal_,
}

optimizers = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD,
}

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='dagcn', type=str, help=', '.join(model_classes.keys()))
parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--l2reg', default=1e-4, type=float)
parser.add_argument('--num_epoch', default=1, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--log_step', default=5, type=int)
parser.add_argument('--embed_dim', default=300, type=int)
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
parser.add_argument('--polarities_dim', default=3, type=int, help='3')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
parser.add_argument('--loop', default=True)

parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
parser.add_argument('--max_length', default=85, type=int)
parser.add_argument('--multi_hop', default=False, type=bool)
parser.add_argument('--max_hop', default=4, type=int)
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
parser.add_argument('--vocab_dir', type=str, default='./dataset/Tweets_corenlp')
parser.add_argument('--pad_id', default=0, type=int)
parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
parser.add_argument('--cuda', default='0', type=str)

parser.add_argument('--fusion', default=True, type=bool,
                    help='fuse distance based weighted matrices belonging to different aspects')
parser.add_argument('--alpha', default=0.8, type=float, help='the weight of distance')
parser.add_argument('--beta', default=0.4, type=float, help='the threshold that whether link aspect with words directly')
parser.add_argument('--gama', default=1.2, type=float, help='the weight of kl divergence loss')
parser.add_argument('--distance_matrix_debug', default=False, type=bool, help='debug mode')

# * bert
parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--bert_dim', type=int, default=768)
parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
parser.add_argument('--diff_lr', default=False, action='store_true')
parser.add_argument('--bert_lr', default=2e-5, type=float)
opt = parser.parse_args()

opt.model_class = model_classes[opt.model_name]
opt.dataset_file = dataset_files[opt.dataset]
opt.inputs_cols = input_colses[opt.model_name]
opt.initializer = initializers[opt.initializer]
opt.optimizer = optimizers[opt.optimizer]

opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

if 'bert' in opt.model_name:
    tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = opt.model_class(bert, opt).to(opt.device)
    trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
    testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
else:
    logger.info('Building tokenizer...')
    tokenizer = build_tokenizer(
        fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
        max_length=opt.max_length,
        data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset)
    )
    logger.info('Building embedding matrix...')
    embedding_matrix = build_embedding_matrix(
        vocab=tokenizer.vocab,
        embed_dim=opt.embed_dim,
        data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset)
    )

    logger.info("Loading vocab...")
    token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')    # token
    post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')    # position
    pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')      # POS
    dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')      # deprel
    pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')      # polarity
    logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))

    opt.post_size = len(post_vocab)
    opt.pos_size = len(pos_vocab)

    vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
    model = opt.model_class(opt, embedding_matrix).to(opt.device)
    trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
    testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# set random seed
setup_seed(opt.seed)

log_dir = './log/'+opt.dataset
if not os.path.exists(log_dir):
    os.makedirs(log_dir, mode=0o777)
log_file = '{}.log'.format(strftime("%Y-%m-%d_%H:%M:%S", localtime()))
logger.addHandler(logging.FileHandler("%s/%s" % (log_dir, log_file)))

our_trainer = Trainer(opt, model, train_dataloader, test_dataloader, logger)
our_trainer._print_args()
our_trainer.run()
t_end = time.time()
print('running time is ', round(t_end-t_start), 'secs.')
