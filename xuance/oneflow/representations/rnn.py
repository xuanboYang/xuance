from xuance.common import Sequence, Optional, Union, Callable, Tuple
import oneflow as flow
from xuance.oneflow import Module, Tensor
from xuance.oneflow.utils import nn, mlp_block, gru_block, lstm_block, ModuleType


class Basic_RNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: dict,
                 normalize: Optional[Module] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
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
        else:
            self.use_normalize = False

    def _create_network(self) -> Tuple[Module, Module, int]:
        if self.fc_hidden_sizes:
            mlp, _ = mlp_block(self.input_shape, self.fc_hidden_sizes[-1],
                              self.fc_hidden_sizes[:-1], self.normalize,
                              self.activation, self.initialize, self.device)
            mlp_net = nn.Sequential(*mlp)
            input_dim = self.fc_hidden_sizes[-1]
        else:
            mlp_net = nn.Identity()
            input_dim = self.input_shape

        if self.lstm:
            rnn_net, output_size = lstm_block(input_dim, self.recurrent_hidden_size, self.N_recurrent_layer,
                                              self.dropout, self.device)
        else:
            rnn_net, output_size = gru_block(input_dim, self.recurrent_hidden_size, self.N_recurrent_layer,
                                             self.dropout, self.device)
        return mlp_net, rnn_net, output_size

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None):
        if self.use_normalize:
            x = self.input_norm(x)
        x = self.mlp(x)
        if self.lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                self.rnn.flatten_parameters()
                output, (hn, cn) = self.rnn(x, (h, c))
                return output.squeeze(0), hn, cn
            else:
                self.rnn.flatten_parameters()
                output, (hn, cn) = self.rnn(x, (h, c))
                return output, hn, cn
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                self.rnn.flatten_parameters()
                output, hn = self.rnn(x, h)
                return output.squeeze(0), hn, None
            else:
                self.rnn.flatten_parameters()
                output, hn = self.rnn(x, h)
                return output, hn, None

    def init_hidden(self, batch):
        h = flow.zeros(self.N_recurrent_layer, batch, self.recurrent_hidden_size, device=self.device)
        c = flow.zeros(self.N_recurrent_layer, batch, self.recurrent_hidden_size, device=self.device)
        return h, c

    def init_hidden_item(self, indexes: list, *rnn_hidden):
        if self.lstm:
            h, c = rnn_hidden
            return h[:, indexes, :].clone().detach(), c[:, indexes, :].clone().detach()
        else:
            h = rnn_hidden[0]
            return h[:, indexes, :].clone().detach(), None

    def get_hidden_item(self, i, *rnn_hidden):
        """
        rnn_hidden: (h, c) for LSTM, (h, None) for GRU
        """
        if self.lstm:
            h, c = rnn_hidden
            return h[:, i, :].clone().detach(), c[:, i, :].clone().detach()
        else:
            h = rnn_hidden[0]
            return h[:, i, :].clone().detach(), None
