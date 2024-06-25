import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(
        self,
        dataPath:str
        ):
        with open(dataPath,"r",encoding="utf-8") as f:
            self.dataset = f.read()
        self.generate_vocabulary()

    def generate_vocabulary(
        self,
        ):
        self.char2index = {}
        self.index2char = {}
        """
        TODO:
        """
        unique_chars = list(set(self.dataset))
        for i, char in enumerate(unique_chars):
            self.char2index[char] = i
            self.index2char[i] = char

    def encode(
        self,
        sentence : str,
        ) -> torch.Tensor:
        """
        TODO:
        例子, 假设A-Z 对应的token是1-26, 句子开始，结束符号的token是0。
        input  : "ABCD"
        output : Tensor([0,1,2,3]) 

        注意: 为了后续实验方便，输出Tensor的数据类型dtype 为torch.long。
        """
        return torch.tensor([self.char2index[char] for char in sentence],dtype=torch.long)

    def decode(
        self,
        tokens : torch.Tensor,
        ) -> str:
        """
        TODO:
        例子, 假设A-Z 对应的token是1-26, 句子开始，结束符号的token是0。
        input : Tensor([0,1,2,3]) 
        output : "ABCD"
        """
        return "".join([self.index2char[i] for i in tokens.tolist()])

class ShakespeareDataset(Dataset):
    def __init__(self, filepath, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        self.encoded = self.tokenizer.encode(text)
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.encoded) - self.chunk_size

    def __getitem__(self, idx):
        #TODO: 提取一段文本(长度为 chunk_size）作为输入，以及这段文本的每一个字符的下一个字符作为标签
        # example(not correspond to real text): chunk = tensor([ 0, 20, 49, 58, 59])
        #         label = tensor([20, 49, 58, 59, 19])
        # decoded chunk: "The "
        # decoded label: "he T"
        chunk = self.encoded[idx:idx+self.chunk_size]
        label = self.encoded[idx+1:idx+self.chunk_size+1]

        return chunk, label

tokenizer = Tokenizer(dataPath="input.txt")

def create_dataloader(filepath, tokenizer, chunk_size, batch_size, shuffle=True):
    dataset = ShakespeareDataset(filepath, tokenizer, chunk_size)
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader


train_dataloader,val_dataloader = create_dataloader('input.txt', tokenizer, chunk_size=200, batch_size=2)

class HeadAttention(nn.Module):
    def __init__(self, seq_len:int, embed_size:int, hidden_size:int):
        super().__init__()
        # embed_size: dimension for input embedding vector
        # hidden_size: dimension for hidden vector. eg. x:(..., embed_size) --to_q--> query_vector:(..., hidden_size)

        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))
        
        # TODO: init three matrix, to_q, to_k, to_v.
        self.to_q = nn.Linear(embed_size, hidden_size)
        self.to_k = nn.Linear(embed_size, hidden_size)
        self.to_v = nn.Linear(embed_size, hidden_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        # return (batch_size, seq_len, hidden_size)
        # TODO: implement the attention mechanism
        q = self.to_q(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        # q, k, v: (batch_size, seq_len, hidden_size)
        # attn: (batch_size, seq_len, seq_len)
        attn = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn = attn.masked_fill(self.tril == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)
    
class MultiHeadAttention(nn.Module):
    # MultiHeadAttention is consist of many HeadAttention output.
    # concat all this head attention output o_i, then merge them with a projection matrix W_o, as [o_1, o_2, ...] x W_o
    # The reason for using multi-head attention is that we want each head to be able to extract different features
    def __init__(self, n_heads:int, head_size:int, seq_len:int, embed_size:int):
        # n_heads is the number of head attention
        # head_size is the hidden_size in each HeadAttention
        super().__init__()
        head_size = embed_size // n_heads
        #TODO: implement heads and projection
        self.heads = nn.ModuleList([HeadAttention(seq_len, embed_size, head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, embed_size)


    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size), make sure embed_size=n_heads x head_size
        # return: (batch_size, seq_len, embed_size)
        # TODO:
        # 1. split the input into n_heads, each with size (batch_size, seq_len, head_size)
        # 2. apply each head attention to the corresponding input
        # 3. concatenate all the head attention output
        # 4. apply the projection matrix
        head_outputs = [head(inputs) for head in self.heads]
        return self.projection(torch.cat(head_outputs, dim=-1))
    

class Expert(nn.Module):
    def __init__(self, embed_size:int):
        super().__init__()
        #TODO: init two linear layer
        self.fc1 = nn.Linear(embed_size, 4*embed_size)
        self.fc2 = nn.Linear(4*embed_size, embed_size)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        # -> mid: (batch_size, seq_len, 4 x embed_size)
        # -> outputs: (batch_size, seq_len, embed_size)
        mid = F.relu(self.fc1(inputs))
        return self.fc2(mid)

# First define the top k router module
class TopkRouter(nn.Module):
    def __init__(self, embed_size, num_experts, active_experts):
        ## TODO
        ## embed_size : dimension of embedding 
        ## num_experts : how many Experts per layer
        ## active_experts: only active_experts out of num_experts are selected to process Embeddings per token.
        super(TopkRouter, self).__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.fc = nn.Linear(embed_size, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    

    def forward(self, inputs):
        ## TODO
        ## 完成这部分时，注意使用Softmax()对router_output做标准化。同时注意这部分所用操作的可导性。
        ## 输入值
        ## inputs is the output tensor from multihead self attention block, shape (B:batch size, T: seq_len, C: embed_size)
        ## 返回值
        ## router_output: normalized weight of Experts, 即教程中的 \alpha
        ## indices:   index of selected Experts, 即教程中的 index
        router_output = self.softmax(self.fc(inputs))
        top_values, indices = torch.topk(router_output, self.active_experts, dim=-1)
        return router_output, indices
    
class SparseMoE(nn.Module):
    def __init__(self, embed_size:int, num_experts:int, active_experts:int):
        ## TODO
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.experts = nn.ModuleList([Expert(embed_size) for _ in range(num_experts)])
        self.router = TopkRouter(embed_size, num_experts, active_experts)

    def forward(self, inputs):
        ## TODO
        router_output, indices = self.router(inputs)
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        final_output = torch.einsum('bnte,btn->bte', expert_outputs, router_output)
        return final_output

class Block(nn.Module):
    # Transformer basic block, consist of MultiHeadAttention, FeedForward and layer normalization
    def __init__(self, embed_size:int, n_heads:int, seq_len:int, num_experts:int, active_experts:int):
        super().__init__()
        # TODO: implement block structure
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mha = MultiHeadAttention(n_heads, embed_size//n_heads, seq_len, embed_size)
        self.ff = SparseMoE(embed_size, num_experts, active_experts)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        #TODO: forward with residual connection
        x = self.mha(inputs)
        x = self.norm1(x + inputs)
        x = self.ff(x)
        return self.norm2(x + inputs)

class SparseMoETransformer(nn.Module):
    # Transformer decoder, consist of 
    # token embedding layer and position_embedding(position_embedding 可以理解为对位置编码，感兴趣的同学可以查阅原文，这里可以看为vocab_len = seq_len的Embedding)
    # a stack of Transformer basic block
    # a layernorm and output linear layer
    def __init__(self, vocab_size:int, seq_len:int, embed_size:int, n_layers:int, n_heads:int, num_experts:int, active_experts:int):
        # vocab_size is the number of word in vocabulary dict
        # seq_len is the sequence length/sentence length
        # embed_size is the embedding vector dimension
        super().__init__()
        # TODO: 
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(seq_len, embed_size)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads, seq_len, num_experts, active_experts) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def forward(self, inputs, labels=None):
        # labels: the (ground) true output 
        # TODO: implement the forward function of the transformer

        # inputs:(batch_size, seq_len, )
        batch_size, seq_len, = inputs.shape
        # embedding:(batch_size, seq_len, embed_size)
        embedding = self.token_embedding(inputs) + self.position_embedding(torch.arange(seq_len, device=device))

        # attens:(batch_size, seq_len, embed_size)
        attens = embedding
        for block in self.blocks:
            attens = block(attens)

        # logits:(batch_size, seq_len, vocab_size)
        logits = self.fc(attens)
        logits = F.log_softmax(logits, dim=-1)

        # compute the loss
        
        if labels is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_len, vocab_size)
            labels = labels.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, labels)
        return logits, loss
    def generate(self, inputs, max_new_tokens):
        inputs = torch.tensor(tokenizer.encode(inputs)).unsqueeze(0)
        device = next(self.parameters()).device  
        inputs = inputs.to(device)
        if inputs.size(1) > self.seq_len:
            inputs = inputs[:, :self.seq_len]
        generated = inputs
        for _ in range(max_new_tokens):
            if generated.size(1) > self.seq_len:
                generated_input = generated[:, -self.seq_len:]
            else:
                generated_input = generated
            logits, _ = self.forward(generated_input)
            last_logits = logits[:, -1, :]  
            next_token_ids = torch.argmax(last_logits, dim=-1)  
            next_token_ids = next_token_ids.unsqueeze(-1)  
            generated = torch.cat([generated, next_token_ids], dim=1)  
        return generated
    
def train(model, dataloader, epoch, device):
    # Optimizer 会根据模型的输出和真实标签计算梯度，然后利用反向传播算法更新模型的参数。
    # 在本实验中你可以将 Optimizer 视作黑盒，只需要知道如何使用即可。
    # 找一个合适的 Optimizer。对不同的任务，模型，最适合的优化器是不一样的，你可以先尝试最常用的 Adam，如果有兴趣可以看看其他的优化器。
    # docs see: https://pytorch.org/docs/stable/optim.html 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    total_loss = 0
    from tqdm import tqdm
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # TODO: implement the training process, and compute the training loss and validation loss
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        


    print(f'Epoch {epoch} Loss: {total_loss / len(dataloader)}')

    return total_loss / len(dataloader)

def validate(model, dataloader, epoch, device):
    model.eval()
    # TODO: 实现验证函数。与训练函数类似，但不需要计算梯度。
    total_loss = 0
    from tqdm import tqdm
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    return total_loss / len(dataloader)

dataloader = create_dataloader('input.txt', tokenizer, chunk_size=20, batch_size=512)
model = SparseMoETransformer(vocab_size=len(tokenizer.char2index), seq_len=20, embed_size=64, n_layers=3, n_heads=8, num_experts=8, active_experts=2).to(device)


# 训练模型
def run(model, train_dataloader, valid_dataloader, device, epochs=10):
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, epoch, device)
        valid_loss = validate(model, valid_dataloader, epoch, device)
        print(f'Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}')

#TODO: 用 matplotlib plot 训练过程中的 loss 变化
import matplotlib.pyplot as plt
def plot_loss(train_loss, valid_loss):
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

train_dataloader, val_dataloader = dataloader

run(model, train_dataloader, val_dataloader, device, epochs=1)

# 保存模型
torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))

print(tokenizer.decode(model.generate("I could pick my lance",max_new_tokens=100)[0].tolist()))