import numpy as np
import dill
import torch

if __name__ == "__main__" :
    model = torch.load('LayerDropModel', pickle_module=dill, map_location=torch.device('cpu'))
    print(model)
    model.device = torch.device('cpu')
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16)
    print(quantized_model)
    torch.save(model.state_dict(), 'LayerDropState', pickle_module=dill)
    torch.save(quantized_model.state_dict(), 'QuantizeStateFLOAT16', pickle_module=dill)