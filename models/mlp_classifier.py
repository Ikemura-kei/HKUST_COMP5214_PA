import torch

class MLPClassifier(torch.nn.Module):
    def __init__(self, num_hidden_neuron, num_class):
        super(MLPClassifier, self).__init__()
        
        self.net = torch.nn.Sequential( \
            torch.nn.Linear(784, num_hidden_neuron, bias=True), \
            torch.nn.Linear(num_hidden_neuron, num_hidden_neuron, bias=True), \
            torch.nn.Linear(num_hidden_neuron, num_class, bias=False) \
            )
        
    def forward(self, x):
        return self.net(x)