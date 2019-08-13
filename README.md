# Computed-similarity
Hybrid model: TreeLSTM+ABCNN3
# Brief description
- fixed the inception problem: the third convolution's filter width must be 'sentence_length-1'
- fixed the hybrid model's linear layer: 'self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)',the input vector size is m1:(1 x 924)，but the wh linear layer is (1208 x 50),so change the '2 * self.mem_dim' into 924 or find proper 'self.mem_dim'
- 'data'the data file including clinic corpus and glove is stored on the could workstation
- 'lib' file including standfordparser is stored on the could workstation
- 'checkpoints' file is not uploaded
# Run the model
1. abmain--------newtrainer------model.ABCNN corresponding to ABCNN3
2. main-----trainer-----------model.SimilarityTreeLSTM corresponding totreelstm
3. hybrid-------hybridtrainer-----------model.Hybrid corresponding to The Hybrid model

- python3 abmain.py   run single abcnn model
- python3 main.py  run single TreeLSTM model
- python3 hybrid.py run hybrid model

