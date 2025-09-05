from utils.transformer_modules import *
from utils.transformer_modules import _gen_timing_signal, _gen_bias_mask
from utils.hparams import HParams
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
use_cuda = torch.cuda.is_available()
from mamba_ssm import Mamba
from utils.transformer_modules import *
from utils.transformer_modules import _gen_timing_signal, _gen_bias_mask
from utils.hparams import HParams

use_cuda = torch.cuda.is_available()

class self_attention_block(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(self_attention_block, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights
        return y

class bi_directional_self_attention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(bi_directional_self_attention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = self_attention_block(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = self_attention_block(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list

class bi_directional_self_attention_layers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(bi_directional_self_attention_layers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[bi_directional_self_attention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list

class BTC_model(nn.Module):
    def __init__(self, config):
        super(BTC_model, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size'], output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction,second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, weights_list, second


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(MambaLayer, self).__init__()
        # Initialize the Mamba model as part of this layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand

        )
        
        # You can add more layers here if needed
        # e.g., self.linear = nn.Linear(d_model, some_other_dimension)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        # Pass the input through the Mamba model
        x = self.mamba(x)
        
        # Additional operations can go here
        # e.g., x = self.linear(x)
        
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)

        
        self.fc21 =       MambaLayer(
           d_model=dim,
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        )
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
        #print(x.shape)
       
        
        # Begin merged shiftmlp forward logic
        B, C,N = x.shape
        x1 = x.clone()
        x2 = self.fc21(x1)

       
        x1 =x2

        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        
        # Original UCMBlock initializations
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Merged shiftmlp components
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, dim)

        self.act = nn.GELU()
        self.act1 = nn.ReLU()  # Assuming Activation is a placeholder for an actual activation like GELU
        self.fc2 = nn.Linear(dim, dim)
        
        self.drop = nn.Dropout(drop)

        
        

        self.norm4 = norm_layer(dim)
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
        #print(x.shape)
       
        
        # Begin merged shiftmlp forward logic
        B, C,N = x.shape
        x1 = x.clone()

        
        x = self.fc1(x)

        x=  self.act(x)

        
        x = self.drop(x)
        
        x = self.fc2(x)

        x = self.drop(x)
        
        
        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=dim)

        self.weight = nn.Parameter(torch.ones(108))
        self.bias = nn.Parameter(torch.zeros(108))

       

    def forward(self, x, H, W):

        B, C,N = x.shape
        x = x.view(B, C, H, W)
       
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2)

        return x

class BMACE(nn.Module):
    def __init__(self, config):
        super(BMACE, self).__init__()

        self.timestep = config['timestep']

        # 🔑 강제로 False로 고정 (추론용)
        self.probs_out = False  

        params = (
            config['feature_size'],
            config['hidden_size'],
            config['num_layers'],
            config['num_heads'],
            config['total_key_depth'],
            config['total_value_depth'],
            config['filter_size'],
            config['timestep'],
            config['input_dropout'],
            config['layer_dropout'],
            config['attention_dropout'],
            config['relu_dropout']
        )

        self.norm1 = nn.LayerNorm(144)
        self.fc1 = nn.Linear(144, config['hidden_size_mace'])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.layer1 = MambaBlock(config['hidden_size_mace'])
        self.layer2 = MambaBlock(config['hidden_size_mace'])

        # 🔑 probs_out을 False로 강제 적용
        self.output_layer = SoftmaxOutputLayer(
            hidden_size=config['hidden_size_mace'] * 2,
            output_size=config['num_chords'],
            probs_out=False
        )

        
   
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size_mace']*2, output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels=None):  # labels 기본값 None으로
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.dropout1(x)

        x1 = x.clone().detach()
        out_forward = self.layer1(x, 12, 12) + x1

        x_reversed = torch.flip(x, dims=[1])
        out_backward = self.layer2(x_reversed, 12, 12)
        out_backward = out_backward + x_reversed.clone().detach()
        out_backward = torch.flip(out_backward, dims=[1])

        out_combined = torch.cat((out_forward, out_backward), dim=-1)
        x = self.dropout1(out_combined)
        self_attn_output = x

        # 🔴 추론 경로: logits만 반환하고 즉시 종료
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            return logits

        # 🔵 학습 경로: labels가 있을 때만 view/손실계산
        logits, second = self.output_layer(self_attn_output)
        prediction = logits.view(-1)
        second = second.view(-1)

        loss = None
        if labels is not None:
            labels = labels.view(-1, self.timestep)
            loss = self.output_layer.loss(self_attn_output, labels)

        return prediction, loss, [], second
    
class MACE_V(nn.Module):
    def __init__(self, config):
        super(MACE_V, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])
        self.norm1 = nn.LayerNorm(144)
        self.fc1 =  nn.Linear(144,128)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.layer1 =  MambaBlock( config['hidden_size'])
        self.layer2 =  MambaBlock( config['hidden_size'])

        
     
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size'], output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        #print(x.shape)
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        #self_attn_output, weights_list = self.self_attn_layers(x)
        x = self.norm1 (x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x1 = x.clone().detach()
        out_forward = self.layer1(x,12,12)
        out_forward+=x1
        
        # Reverse the input sequence for the backward pass
       # x_reversed = torch.flip(x, dims=[1])  # Assuming the sequence is along dimension 1
        out_backward = self.layer2(out_forward,12,12)
        x_reversed1 = out_forward.clone().detach()
        out_backward+=x_reversed1
        
  
        out_combined =out_backward #out_backwardtorch.cat((out_forward, out_backward), dim=-1)
        

        x = self.dropout1( out_combined)
        self_attn_output=x
        

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction,second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, [], second    
    
class MACE_H(nn.Module):
    def __init__(self, config):
        super(MACE_H, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])
        self.norm1 = nn.LayerNorm(144)
        self.fc1 =  nn.Linear(144,128)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.layer1 =  MambaBlock( config['hidden_size'])
        self.layer2 =  MambaBlock( config['hidden_size'])

        
       
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size']*2, output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        #print(x.shape)
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        #self_attn_output, weights_list = self.self_attn_layers(x)
        x = self.norm1 (x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x1 = x.clone().detach()
        out_forward = self.layer1(x,12,12)
        out_forward+=x1
        
        # Reverse the input sequence for the backward pass
      
        out_forward1 = self.layer2(x,12,12)
        x_reversed1 = x.clone().detach()
        out_forward1+=x_reversed1
        
        # Reverse the backward output to match the forward output's temporal alignment

    
        out_combined = torch.cat((out_forward, out_forward1), dim=-1)

        x = self.dropout1( out_combined)
        self_attn_output=x
        

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction,second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, [], second
if __name__ == "__main__":
    config = HParams.load("run_config.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1
    timestep = 108
    feature_size = 144
    num_chords = 25

    features = torch.randn(batch_size,timestep,feature_size,requires_grad=True).to(device)
    chords = torch.randint(25,(batch_size*timestep,)).to(device)

    model = BMACE(config=config.model).to(device)

    prediction, loss, weights_list, second = model(features, chords)
    
    print(features.shape)
    print(prediction.size())
    print(loss)
    total_params = sum(p.numel() for p in model.parameters() )

    print(f"Ours Total number of trainable parameters: {total_params}")

    model = MACE_H(config=config.model).to(device)

    prediction, loss, weights_list, second = model(features, chords)
    
    print(features.shape)
    print(prediction.size())
    print(loss)
    total_params = sum(p.numel() for p in model.parameters() )

    print(f"BTC Total number of trainable parameters: {total_params}")


