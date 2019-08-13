# computed-similarity
Hybrid model: TreeLSTM+ABCNN3
# brief description
- fixed the inception problem: the third convolution's filter width must be 'sentence_length-1'-
- fixed the hybrid model's linear layer: 'self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)',the input vector size is m1:(1 x 924)，but the wh linear layer is (1208 x 50),so change the '2 * self.mem_dim' into 924 or find proper 'self.mem_dim'
- 'data'the data file including clinic corpus and glove is stored on the could workstation
- 'lib' file including standfordparser is stored on the could workstation
- 'checkpoints' file is not uploaded
