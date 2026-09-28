from src.get_data import ImageDataStreamer
import snntorch as snn
import torch.nn as nn
import torch

data_dir = "data/mdata"
pixel_size = 15
input_features = int(pixel_size**2)
N_exc = 200
N_inh = 50
batch_size = 100
num_steps = 100
alpha = 30  # synaptic decay
beta = 30  # membrane decay

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def run_dataset(dataset: str):
    # initiate streamer to fetch data
    streamer = ImageDataStreamer(
        data_dir=data_dir,
        batch_size=batch_size,
        pixel_size=pixel_size,
        num_steps=num_steps,
        dataset=dataset,
    )


# define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # W_SE
        self.fc1 = nn.Linear(in_features=input_features, out_features=N_exc, bias=True)
        # W_EE
        self.lif1 = snn.RLeaky(
            beta=beta, alpha=alpha, num_inputs=N_exc, batch_size=batch_size
        )
        # W_EI
        self.fc2 = nn.Linear(in_features=N_exc, out_features=N_inh, bias=True)
        # W_IE
