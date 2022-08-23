import time
import zmq
import sys
import os
import zipfile
import torch
from Config import Config
from CodeBERTaModel import CodeBERTaEncoderDecoder
from transformers import RobertaTokenizer
import dill


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    src_path = sys.argv[1]
    if not os.path.isfile(src_path + '/OriginalFullDatasetState'):
        with zipfile.ZipFile(src_path + '/OriginalFullDatasetState.zip', 'r') as zip_ref:
            zip_ref.extractall(src_path)
        os.remove(src_path + '/OriginalFullDatasetState.zip')
    t1 = time.time()

    # Load the model to CPU
    device = torch.device("cpu")
    config = Config()
    model = CodeBERTaEncoderDecoder(config,device)
    model.load_state_dict(torch.load(src_path + '/OriginalFullDatasetState', pickle_module=dill, map_location='cpu'), strict=False)
    tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    while True:
        #  Wait for next request from client
        message = socket.recv().decode("utf-8") 

        if message != "ThisIsToStopRunningTheModel":
            selected_code_raw = message
            selected_code = tokenizer.batch_encode_plus([message])["input_ids"]
            underscore_token = tokenizer.get_vocab()["_"]
            selected_code = [[token for token in input if token != underscore_token][:tokenizer.model_max_length] for input in selected_code]
            evaluate_input = [(selected_code[0], [0, 883, 19603, 8588, 2], len(selected_code[0]), 5, selected_code_raw[0])]
            top_sequence,top_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences,inputs_raw = model.evaluate(evaluate_input)
            ind = 1
            message_string = ""
            for name in decoded_sequences[0]:
                message_string += str(ind) + '. ' + name.lstrip("_").rstrip("_") + " "
                ind = ind + 1
            socket.send(str.encode(message_string))
        else:
            socket.send(str.encode("Model is unloaded."))
            break