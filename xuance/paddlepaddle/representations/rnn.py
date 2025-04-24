from xuance.common import Sequence, Optional, Union, Callable, Tuple
from xuance.paddlepaddle import Layer, Tensor
from xuance.paddlepaddle.utils import paddle, nn, mlp_block, gru_block, lstm_block, ModuleType


class Basic_RNN(Layer):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalize: Optional[Layer] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int]] = None,
                 **kwargs):
        super(Basic_RNN, self).__init__()
        self.input_shape = input_shape
        self.fc_hidden_sizes = hidden_sizes["fc_hidden_sizes"]
        self.recurrent_hidden_size = hidden_sizes["recurrent_hidden_size"]
        self.N_recurrent_layer = kwargs["N_recurrent_layers"]
        self.dropout = kwargs["dropout"]
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes["recurrent_hidden_size"],)}
        self.mlp, self.rnn, output_dim = self._create_network()
        if self.normalize is not None:
            self.use_normalize = True
            self.input_norm = self.normalize(input_shape, device=device)
            self.norm_rnn = self.normalize(output_dim, device=device)
        else:
            self.use_normalize = False

    def _create_network(self) -> Tuple[Layer, Layer, int]:
        layers = []
        input_shape = self.input_shape
        for h in self.fc_hidden_sizes:
            mlp_layer, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                               device=self.device)
            layers.extend(mlp_layer)
        if self.lstm:
            rnn_layer, input_shape = lstm_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                                self.dropout, self.initialize, self.device)
        else:
            rnn_layer, input_shape = gru_block(input_shape[0], self.recurrent_hidden_size, self.N_recurrent_layer,
                                               self.dropout, self.initialize, self.device)
        return nn.Sequential(*layers), rnn_layer, input_shape

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None):
        if self.use_normalize:
            tensor_x = self.input_norm(paddle.to_tensor(x, dtype=paddle.float32))
        else:
            tensor_x = paddle.to_tensor(x, dtype=paddle.float32)
        mlp_output = self.mlp(tensor_x)
        self.rnn.flatten_parameters()
        if self.lstm:
            output, (hn, cn) = self.rnn(mlp_output, (h, c))
            if self.use_normalize:
                output = self.norm_rnn(output)
            return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": cn.detach()}
        else:
            output, hn = self.rnn(mlp_output, h)
            if self.use_normalize:
                output = self.norm_rnn(output)
            return {"state": output, "rnn_hidden": hn.detach(), "rnn_cell": None}

    def init_hidden(self, batch):
        # hidden_states = torch.zeros(size=(self.N_recurrent_layer, batch, self.recurrent_hidden_size)).to(self.device)
        # cell_states = torch.zeros_like(hidden_states).to(self.device) if self.lstm else None
        hidden_states = paddle.zeros(shape=(self.N_recurrent_layer, batch, self.recurrent_hidden_size), dtype='float32')
        hidden_states = hidden_states.cuda() if self.device == 'gpu' else hidden_states  # 根据设备移动张量

        cell_states = paddle.zeros_like(hidden_states, dtype='float32') if self.lstm else None
        if cell_states is not None and self.device == 'gpu':
            cell_states = cell_states.cuda()  # 如果需要，将 cell_states 移动到 GPU

        return hidden_states, cell_states

    def init_hidden_item(self, indexes: list, *rnn_hidden):
        zeros_size = (self.N_recurrent_layer, len(indexes), self.recurrent_hidden_size)
        if self.lstm:
            # rnn_hidden[0][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            # rnn_hidden[1][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            zeros_tensor = paddle.zeros(shape=zeros_size, dtype=rnn_hidden[0].dtype)
            if self.device == 'gpu':
                zeros_tensor = zeros_tensor.cuda()

            rnn_hidden[0][:, indexes, :] = zeros_tensor  # 注意：PaddlePaddle 的赋值需要确保维度匹配
            rnn_hidden[1][:, indexes, :] = zeros_tensor

            return rnn_hidden
        else:
            # rnn_hidden[0][:, indexes] = torch.zeros(size=zeros_size).to(self.device)
            # 创建零张量
            zeros_tensor = paddle.zeros(shape=zeros_size, dtype=rnn_hidden[0].dtype)

            # 如果需要移动到 GPU
            if self.device == 'gpu':
                zeros_tensor = zeros_tensor.cuda()

            # 索引赋值
            rnn_hidden[0][:, indexes, :] = zeros_tensor
            return rnn_hidden

    def get_hidden_item(self, i, *rnn_hidden):
        return (rnn_hidden[0][:, i], rnn_hidden[1][:, i]) if self.lstm else (rnn_hidden[0][:, i], None)
