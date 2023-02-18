import torch
import torch.nn as nn

# Define o tamanho da sequência de entrada, o tamanho do vocabulário e o tamanho do embedding
seq_len = 10
vocab_size = 100
embedding_dim = 50

# Define o tamanho da camada oculta e o número de camadas
hidden_size = 100
num_layers = 2

# Cria a rede neural recorrente
rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)

# Define a função de perda
criterion = nn.CrossEntropyLoss()

# Define o otimizador
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

# Define os dados de entrada e saída
inputs = torch.randint(low=0, high=vocab_size, size=(1, seq_len))
targets = torch.randint(low=0, high=vocab_size, size=(1, seq_len))

# Cria um embedding para os dados de entrada
embedding = nn.Embedding(vocab_size, embedding_dim)

# Faz o forward pass e calcula a perda
inputs = embedding(inputs)
outputs, _ = rnn(inputs)
loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

# Faz o backward pass e atualiza os parâmetros
optimizer.zero_grad()
loss.backward()
optimizer.step()
