import sys
import argparse
import h5py
import numpy as np
import torch
from utils.holder import *
from utils.util import *
import qa.pipeline
from transformers import *
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="Path to where HDF5 model to be loaded.", default="")
## dim specs
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
#
parser.add_argument('--output', help="Path to output HuggingFace(HF) format", default='/models/hf')

# Since the models are to be converted, not trained via HF, we will have to "fake" some training options
def artificialize_training_bin():
	params = argparse.Namespace()
	params.adam_epsilon=1e-08
	params.cache_dir=''
	params.config_name=''
	params.data_dir=None
	params.device=torch.device(type='cuda')
	params.do_eval=True
	params.do_lower_case=True
	params.do_train=True
	params.doc_stride=128
	params.eval_all_checkpoints=False
	params.evaluate_during_training=False
	params.fp16=False
	params.fp16_opt_level='O1'
	params.gradient_accumulation_steps=1
	params.lang_id=0
	params.learning_rate=3e-05
	params.local_rank=-1
	params.logging_steps=500
	params.max_answer_length=30
	params.max_grad_norm=1.0
	params.max_query_length=64
	params.max_seq_length=384
	params.max_steps=-1
	params.model_name_or_path='bert-base-uncased'
	params.model_type='bert'
	params.n_best_size=20
	params.n_gpu=1
	params.no_cuda=False
	params.null_score_diff_threshold=0.0
	params.num_train_epochs=2.0
	params.output_dir='NOT_DEFINED'
	params.overwrite_cache=True
	params.overwrite_output_dir=False
	params.per_gpu_eval_batch_size=8
	params.per_gpu_train_batch_size=20
	params.predict_file='NOT_DEFINED'
	params.save_steps=500
	params.seed=3435
	params.server_ip=''
	params.server_port=''
	params.threads=1
	params.tokenizer_name=''
	params.train_batch_size=20
	params.train_file='model_name_or_path'
	params.verbose_logging=False
	params.version_2_with_negative=False
	params.warmup_steps=0
	params.weight_decay=0.0
	return params

def main(args):
	opt = parser.parse_args(args)
	opt.gpuid = -1
	opt.dropout = 0
	shared = Holder()

	# fix some hyperparameters automatically
	if 'base' in opt.transformer_type:
		opt.hidden_size = 768
	elif 'large'in opt.transformer_type:
		opt.hidden_size = 1024

	# load model
	m = qa.pipeline.Pipeline(opt, shared)
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	mlm = AutoModel.from_pretrained(opt.transformer_type)
	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False, use_fast=True)
	training_args = artificialize_training_bin()

	config = mlm.config
	config.num_labels = 2	# 1 for start and 1 for end
	training_args.model_name_or_path = opt.transformer_type

	if 'roberta' in opt.transformer_type:
		training_args.model_type = 'roberta'
		m_hf = RobertaForQuestionAnswering(config)
		# move parameters
		m_hf.roberta = m.encoder.transformer
		m_hf.qa_outputs = m.classifier.linear[1]
	elif 'distilbert' in opt.transformer_type:
		training_args.model_type = 'distilbert'
		m_hf = DistilBertForQuestionAnswering(config)
		# move parameters
		m_hf.distilbert = m.encoder.transformer
		m_hf.qa_outputs = m.classifier.linear[1]
	elif 'bert' in opt.transformer_type:
		training_args.model_type = 'bert'
		m_hf = BertForQuestionAnswering(config)
		# move parameters
		m_hf.bert = m.encoder.transformer
		m_hf.qa_outputs = m.classifier.linear[1]
	else:
		raise Exception('unrecognized model type {0}'.format(opt.transformer_type))
	
	m_hf.save_pretrained(opt.output)
	tokenizer.save_pretrained(opt.output)
	torch.save(training_args, opt.output + '/training_args.bin')


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
