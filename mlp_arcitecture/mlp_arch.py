import torch
import torch.nn as nn

# def zero_initialize_weights(model):
#     for param in model.parameters():
#         if param.requires_grad:
#             nn.init.zeros_(param)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        input_size and out put_size:  single integer
        hidden_size: list of integers

        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        """
        assume x: [batch,T,1]
        """
        x1 = x.permute( (0, 2, 1))
        for layer in self.layers:
            x1 = layer(x1)
        return x1.permute((0, 2, 1))


# class YourSecondModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(YourSecondModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#         # Initializing weights with zeros
#         #self.init_weights()
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#     def init_weights(self):
#         for layer in [self.fc1, self.fc2]:
#             nn.init.zeros_(layer.weight)
#             nn.init.zeros_(layer.bias)

if __name__ == "__main__":
    # Define the sizes of the MLP layers
    input_size = 75
    hidden_sizes = [1024, 1024]
    output_size = 75

    # Create an instance of the MLP model
    model = MLP(input_size, hidden_sizes, output_size)

    # Print the model architecture
    print(model)
    inp = torch.randn(350, 75,1)

    # with torch.no_grad():
    #     out1 = model(inp)
    #     print(out1)
    #     print("----------------------")
    #     zero_initialize_weights(model)
    #     out2 = model(inp)
    #     print(out2)






