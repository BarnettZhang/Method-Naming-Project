import torch
from transformers import get_linear_schedule_with_warmup, EncoderDecoderModel, RobertaTokenizer
import os
import copy
from time import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


bcolors = {
    'RESULTS': '\033[95m',
    'HEADER': '\033[94m',
    'SUCCESS': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'INFO': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def printc(log, color='HEADER'):
    """
    Prints logs with color according to the dict bcolors
    """
    print(f"{bcolors[color]}{log}{bcolors['ENDC']}")

class LayerDropModuleList(torch.nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m

class Seq2SeqModelInterface(torch.nn.Module):
    def __init__(self,config,device):
        """
        PyTorch Seq2SeqModel interface
        Every model has to inherit from Seq2SeqModelInterface so training and testing run correctly

        At least the methods defined below and which raise NotImplementedError must be implemented
        - self.optimizer
        - self.scheduler
        """
        super(Seq2SeqModelInterface, self).__init__()
        self.device = device
        self.config = config
        
    def initialize_scheduler(self, total_steps=0):
        """
        Creates a scheduler for a given otimizer
        """
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=2, # Default value
                                                        num_training_steps=total_steps)
    def resume(self, config):
        """
        Resumes with a given checkpoint. Loads the saved parameters, optimizer and scheduler.
        """
        printc(f"Resuming with model at {config.resume}...", "INFO")
        path_checkpoint = config.resume
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=self.device)
        
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def step(self, batch):
        """
        Args:
            batch: data from the data loaders (see training.py)
        Output:
            loss (tensor): PyTorch loss
            outputs (batch_size,seq_len,vocab_size): model outputs (raw predictions without softmax)
        
        Examples::
            >>> batch = next(iter(train_loader))
            >>> loss, outputs = model.step(batch)
        """
        raise NotImplementedError

    def forward(self,*args, **kwargs):
        """
        PyTorch nn.Module forward
        It is specific to the model, and the args have no specific format
        """
        raise NotImplementedError

    def evaluate(self, batch, num_sequences=1):
        """
        Args:
            batch: data from the data loaders (similar to training data)
            num_sequences: the number of sequences to output 
        Output:
            top_seqences(batch_size,num_sequences,max_output_seq_len): The top num_sequences predictions
            top_lengths(batch_size,num_sequences): The actual lengths of the top num_sequences predictions
            target_sequences(batch_size,batch_tgt_max_seq_len): The target sequences corresponding to the predicted ones for metrics computation
            target_lengths(batch_size): The actual lengths of the top target sequences
            decoded_sequences List[List[string] * num_sequences]*batch_size: The top num_sequences predictions decoded (as strings)
            outputs_probability (batch_size,num_sequences,max_output_seq_len - 1, vocab_size): model outputs passed through a softmax to turn into probabilities
        
        Examples::
            >>> batch = next(iter(eval_loader))
            >>> (top_seqences,top_lengths,target_sequences, target_lengths, 
                decoded_sequences,outputs_probability) = model.evaluate(batch,num_sequences=num_output_sequences)
        """
        raise NotImplementedError

    def single_inference(self, function_string, num_sequences=1):
        """
        Args:
            function_string: raw text data (i.e a function extracted using ast)
            num_sequences: the number of sequences to output 
        Output:
            decoded_sequence [num_sequences]: The top num_sequences predictions decoded (as strings)
            sequence_scores [num_sequences]: Probability for each sequence
        
        Examples:
            >>> decoded_sequences,sequence_scores = model.single_inference(function_string,num_sequences=num_output_sequences)
        """
        raise NotImplementedError

def apply_layerdrop(model, layerdrop_rate=0.2):
    oldModuleList = model.encoder.encoder.layer
    newModuleList = LayerDropModuleList(0.2)

    # Now iterate over all layers, only keeping the relevant layers.
    for i in range(6):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.encoder.layer = newModuleList

    return copyOfModel

def set_dropout(model, drop_rate=0.1):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)

class CodeBERTaEncoderDecoder(Seq2SeqModelInterface):
    def __init__(self,config,device, layerdrop=False, layerdrop_rate=0):
        """
        RoBERTa to RoBERTa encoder-decoder using HuggingFace pretrained CodeBERTa models trained on the CodeNet challenge dataset on LM tasks.
        Elements in training batch for this model should be tuples (inputs,labels,inputs_lengths,labels_lengths). 
        Inputs and labels do not need any padding.
        """
        super(CodeBERTaEncoderDecoder, self).__init__(config,device)
        self.config = config
        assert config.model_type == "CodeBERTa", 'Error: Wrong model type!'

        self.model_name = config.model_name
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.model_name, self.model_name).to(self.device)
        if layerdrop == True:
            self.model = apply_layerdrop(self.model, layerdrop_rate)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        self.max_output_seq_len = config.max_output_seq_len
        self.learning_rate = config.learning_rate
        self.max_grad_norm = config.max_grad_norm

        self.optimizer = Adam(self.model.parameters(), lr = self.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        if config.resume:
            self.resume(config)

        self.drop_rate = config.drop_rate
        if self.drop_rate:
            set_dropout(self.model, drop_rate=self.drop_rate)
            
    def apply_layerdrop(model, layerdrop_rate=0.2):
        oldModuleList = model.encoder.encoder.layer
        newModuleList = LayerDropModuleList(0.2)

        # Now iterate over all layers, only keeping the relevant layers.
        for i in range(6):
            newModuleList.append(oldModuleList[i])

        # create a copy of the model, modify it with the new list, and return
        copyOfModel = copy.deepcopy(model)
        copyOfModel.encoder.encoder.layer = newModuleList

        return copyOfModel

    def get_models_inputs_from_pair_batch(self,batch):
        batch_size = len(batch)
        unzipped = list(zip(*batch))
        inputs,targets,inputs_lengths,targets_lengths,inputs_raw = unzipped[0],unzipped[1],unzipped[2],unzipped[3],unzipped[4]

        PAD_token = self.tokenizer.pad_token_id

        #Build input tensor and pad
        inputs_lengths_tensor = torch.LongTensor(inputs_lengths)
        inputs_tensor = torch.ones(batch_size,inputs_lengths_tensor.max()).long() * PAD_token
        for idx, (seq, seqlen) in enumerate(zip(inputs, inputs_lengths_tensor)):
            inputs_tensor[idx,:seqlen] = torch.LongTensor(seq)

        inputs_attention_mask = (inputs_tensor != PAD_token) * 1

        #Build target tensor and pad
        targets_lengths_tensor = torch.LongTensor(targets_lengths)
        targets_tensor = torch.ones(batch_size,targets_lengths_tensor.max()).long() * PAD_token

        for idx, (seq, seqlen) in enumerate(zip(targets, targets_lengths_tensor)):
            targets_tensor[idx,:seqlen] = torch.LongTensor(seq)

        targets_attention_mask = (targets_tensor != PAD_token) * 1

        return (inputs_tensor, targets_tensor,targets_lengths_tensor,inputs_attention_mask,targets_attention_mask,inputs_raw)

    def step(self, batch):
        """
        Args:
            batch: a batch of training data in the form described above in init
        Output:
            loss (tensor): PyTorch loss
            outputs (batch_size,seq_len,vocab_size): model outputs (predictions or something else)
        """
        #Unpack batch data
        src_seqs,tgt_seqs,tgt_lens,src_mask,tgt_mask,_ = self.get_models_inputs_from_pair_batch(batch)
        
        src_seqs = src_seqs.to(self.device)
        tgt_seqs = tgt_seqs.to(self.device)

        tgt_lens = tgt_lens.to(self.device)

        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        # -------------------------------------
        # Training mode (enable dropout)
        # -------------------------------------
        self.model.train()    

        loss,outputs = self.forward(src_seqs,tgt_seqs,src_mask,tgt_mask)
        # -------------------------------------
        # Backward and optimize
        # -------------------------------------
        # Backward to get gradients w.r.t parameters in model.
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.max_grad_norm)
        
        # Update parameters with optimizer
        self.optimizer.step()

        return loss,outputs
            

    def forward(self,src_seqs,tgt_seqs,src_mask,tgt_mask):
        # if self.is_quantized == True:
        #     src_seqs = self.quant(src_seqs)
        #     tgt_seqs = self.quant(tgt_seqs)
        #     src_mask = self.quant(src_mask)
        #     tgt_mask = self.quant(tgt_mask)
        output = self.model(input_ids=src_seqs,decoder_input_ids=tgt_seqs,labels=tgt_seqs,encoder_attention_mask=src_mask,decoder_attention_mask=tgt_mask)
        return output.loss,output.logits

    def evaluate(self, eval_batch, max_seq_len=None,num_return_sequences=None,num_beams=5):
        """
        Args:
            eval_batch: batch data in the same form as train data (described in init)
            num_sequences: the number of sequences to output 
            max_seq_len: Maximum output sequence length
            num_beams: Number of beams for beam search. 
        Output:
            top_seqences(batch_size,num_sequences,max_seq_len): The top num_sequences predictions
            top_lengths(batch_size,num_sequences): The actual lengths of the top num_sequences predictions
            target_sequences(batch_size,batch_tgt_max_seq_len): The target sequences corresponding to the predicted ones for metrics computation
            target_lengths(batch_size): The actual lengths of the top target sequences
            decoded_sequences List[List[string] * num_sequences]*batch_size: The top num_sequences predictions decoded (as strings)
            outputs_probability (batch_size,num_sequences,max_seq_len - 1, vocab_size): model outputs passed through a softmax to turn into probabilities
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_output_seq_len
        if num_return_sequences is None:
            num_return_sequences = self.config.num_return_sequences
        with torch.no_grad():
            batch_size = len(eval_batch)

            #Unpack batch data
            src_seqs,tgt_seqs,tgt_lens,_,_,inputs_raw = self.get_models_inputs_from_pair_batch(eval_batch)

            src_seqs = src_seqs.to(self.device)
            tgt_seqs = tgt_seqs.to(self.device)
            tgt_lens = tgt_lens.to(self.device)


            # -------------------------------------
            # Eval mode mode (disable dropout)
            # -------------------------------------
            self.model.eval()

            # -------------------------------------
            # Forward model
            # -------------------------------------
            start_beam_search = time()
            beam_output = self.model.generate(
                                src_seqs, 
                                max_length=self.max_output_seq_len, 
                                num_beams=num_beams, 
                                num_return_sequences=num_return_sequences, 
                                early_stopping=True,
                                output_scores = True,
                                return_dict_in_generate=True,
                                no_repeat_ngram_size = 1,
                                eos_token_id = self.tokenizer.eos_token_id,
                                pad_token_id = self.tokenizer.eos_token_id
                            )
            beam_search_time = time() - start_beam_search
            #top_sequence = (batch_size,num_sequences,max_seq_len)
            #top_length = (batch_size,num_sequences)
            top_sequence = beam_output["sequences"].view(batch_size,num_return_sequences,beam_output["sequences"].size(1))
            # non zero values mask
            eos_mask = top_sequence == self.tokenizer.eos_token_id

            # operations on the mask to find first EOS_token in the rows
            mask_max_values, eos_index = torch.max(eos_mask, dim=2)
            # Actual length is one more than the index
            top_length = eos_index + 1

            # if the max-mask is zero, there is no pad index in the row, the length is the length of the sequence
            top_length[mask_max_values == 0] = top_sequence.size(2)

            #get output probabilites
            outputs = torch.stack(beam_output['scores']).transpose(0,1).view(batch_size,num_return_sequences,beam_output["sequences"].size(1) - 1,self.tokenizer.vocab_size)
            output_prob = torch.nn.functional.softmax(outputs,dim=3)
            #decode sequences and add _
            to_decode_full_batch = []
            for i in range(batch_size):
                to_decode_single_batch = []
                for j in range(num_return_sequences):
                    top_sequence_to_decode = [self.tokenizer.convert_tokens_to_ids("_")] * (len(top_sequence[i][j]) * 2 - 1)
                    top_sequence_to_decode[0::2] = top_sequence[i][j]
                    to_decode_single_batch.append(top_sequence_to_decode)
                to_decode_full_batch.append(to_decode_single_batch)
            

            #decode sequences
            decoded_sequences = [self.tokenizer.batch_decode(to_decode_full_batch[i],skip_special_tokens=True) for i in range(batch_size)]

            del outputs,src_seqs,eos_index,eos_mask,mask_max_values,beam_output

        return top_sequence,top_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences,inputs_raw

    def single_inference(self, function_string,num_return_sequences=None):
        """
        Args:
            function_string: raw text data (i.e a function extracted using ast)
            num_sequences: the number of sequences to output 
        Output:
            decoded_sequence [num_sequences]: The top num_sequences predictions decoded (as strings)
            sequence_scores [num_sequences]: Probability for each sequence
        """
        raise NotImplementedError