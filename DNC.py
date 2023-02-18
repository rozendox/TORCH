import torch
from torch import nn
from dnc.dnc import DNC

# Define o tamanho das entradas e saídas da rede
input_size = 10
output_size = 5

# Define os parâmetros da DNC
memory_size = 100
memory_dim = 20
num_layers = 2
num_hidden = 200

# Cria a DNC
dnc = DNC(input_size, output_size, memory_size, memory_dim, num_layers, num_hidden)

# Define a função de perda
criterion = nn.MSELoss()

# Define o otimizador
optimizer = torch.optim.Adam(dnc.parameters(), lr=0.001)

# Define os dados de entrada e saída
inputs = torch.randn(32, input_size)
targets = torch.randn(32, output_size)

# Treina a rede
for i in range(1000):
    # Forward pass
    outputs, _ = dnc(inputs)

    # Calcula a perda
    loss = criterion(outputs, targets)

    # Backward pass e atualiza os parâmetros
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Imprime o progresso a cada 100 iterações
    if i % 100 == 0:
        print(f"Iteração {i}: perda = {loss.item()}")
