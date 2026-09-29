"""snnTorch version of the E/I network, trained with surrogate gradients.

Sign is fixed per weight: excitatory weights (W_SE, W_EE, W_EI) stay positive,
W_IE is stored as a positive magnitude and enters the current with a minus sign
(so the inhibitory synapse stays negative), and W_OUT keeps whatever sign it was
initialised with. Signs are enforced twice: the downscaling is multiplicative
in log-magnitude (it cannot cross zero), and a projection after every optimizer
step clamps any weight that the gradient pushed through zero back to a small
magnitude with its original sign.

Usage
-----
  python main.py --dataset mnist --epochs 1
"""

import argparse

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from snntorch import spikegen, surrogate
from torch.utils.data import DataLoader
from torchvision import datasets

pixel_size = 15
N_se = pixel_size**2
N_exc = 200
N_inh = 50
N_out = 10
batch_size = 100
num_steps = 100
tau_syn = 30.0  # synaptic time constant (steps)
tau_mem = 30.0  # membrane time constant (steps)
alpha = float(torch.exp(torch.tensor(-1.0 / tau_syn)))  # snntorch wants decay in (0, 1)
beta = float(torch.exp(torch.tensor(-1.0 / tau_mem)))
decay_lambda = 0.99997  # per-step contraction of log(|w| / w_target)
weight_target = 0.2  # target magnitude (inhibitory target is -weight_target)
w_floor = 1e-4  # smallest magnitude a weight may have after projection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignedLinear(nn.Module):
    """Bias-free linear layer whose weights keep the sign they were born with."""

    def __init__(self, n_in, n_out, scale, mixed_sign=False):
        super().__init__()
        mag = scale * (0.5 + torch.rand(n_out, n_in))
        sign = torch.where(torch.rand(n_out, n_in) < 0.5, -1.0, 1.0) if mixed_sign else torch.ones(n_out, n_in)
        self.weight = nn.Parameter(mag * sign)
        self.register_buffer("sign", sign)

    def forward(self, x):
        return F.linear(x, self.weight)

    @torch.no_grad()
    def project(self):
        """Clamp every weight back onto its own sign (never zero)."""
        self.weight.copy_(self.sign * (self.weight * self.sign).clamp_min(w_floor))

    @torch.no_grad()
    def downscale(self, n_steps):
        """n applications of u <- lambda*u, u = log(|w| / w_target), in one op."""
        power = decay_lambda**n_steps
        mag = (self.weight * self.sign).clamp_min(w_floor)
        self.weight.copy_(self.sign * weight_target * (mag / weight_target) ** power)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_SE = SignedLinear(N_se, N_exc, scale=0.1)
        self.W_EE = SignedLinear(N_exc, N_exc, scale=0.02)
        self.W_EI = SignedLinear(N_exc, N_inh, scale=0.05)
        self.W_IE = SignedLinear(N_inh, N_exc, scale=0.05)  # magnitude; subtracted below
        self.W_OUT = SignedLinear(N_exc, N_out, scale=0.1, mixed_sign=True)
        grad = surrogate.fast_sigmoid()
        self.lif_E = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=grad)
        self.lif_I = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=grad)
        self.layers = [self.W_SE, self.W_EE, self.W_EI, self.W_IE, self.W_OUT]

    def forward(self, x):
        """x: [T, B, N_se] spikes -> [T, B, N_out] readout."""
        T, B, _ = x.shape
        s_E = x.new_zeros(B, N_exc)
        s_I = x.new_zeros(B, N_inh)
        syn_E, mem_E = self.lif_E.init_synaptic()
        syn_I, mem_I = self.lif_I.init_synaptic()
        out = []
        for t in range(T):
            cur_E = self.W_SE(x[t]) + self.W_EE(s_E) - self.W_IE(s_I)
            cur_I = self.W_EI(s_E)
            s_E, syn_E, mem_E = self.lif_E(cur_E, syn_E, mem_E)
            s_I, syn_I, mem_I = self.lif_I(cur_I, syn_I, mem_I)
            out.append(self.W_OUT(s_E))
        return torch.stack(out)

    @torch.no_grad()
    def sleep(self, n_steps):
        for layer in self.layers:
            layer.downscale(n_steps)

    @torch.no_grad()
    def project(self):
        for layer in self.layers:
            layer.project()

    def sign_violations(self):
        return sum(int(((l.weight * l.sign) <= 0).sum()) for l in self.layers)


def get_data(dataset):
    ds_map = {
        "mnist": datasets.MNIST,
        "kmnist": datasets.KMNIST,
        "fmnist": datasets.FashionMNIST,
    }
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((pixel_size, pixel_size)),
            transforms.ToTensor(),
        ]
    )
    root = "../data/torchvision"
    train = ds_map[dataset](root=root, train=True, download=True, transform=transform)
    test = ds_map[dataset](root=root, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True),
    )


def encode(img):
    """Poisson rate code: [B,1,H,W] -> [T,B,N_se]."""
    return spikegen.rate(img.flatten(1), num_steps=num_steps)


@torch.no_grad()
def evaluate(net, loader, max_batches=None):
    net.eval()
    correct = total = 0
    for i, (img, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pred = net(encode(img.to(device))).mean(0).argmax(1)
        correct += int((pred == y.to(device)).sum())
        total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "kmnist", "fmnist"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (smoke test)")
    p.add_argument("--no-sleep", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    train_loader, test_loader = get_data(args.dataset)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        net.train()
        for i, (img, y) in enumerate(train_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            logits = net(encode(img.to(device))).mean(0)
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.project()
            if not args.no_sleep:
                # one downscaling op per batch, equal to num_steps per-step decays
                net.sleep(num_steps)
            if i % 50 == 0:
                print(f"epoch {epoch} batch {i} loss {loss.item():.4f} sign_violations {net.sign_violations()}")
        acc = evaluate(net, test_loader, args.max_batches)
        print(f"epoch {epoch} test acc {acc:.4f} sign_violations {net.sign_violations()}")


if __name__ == "__main__":
    main()
