# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size,input_dim,hidden_dim,kernel_size,bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width))



class ConvLSTM(nn.Module):

    def __init__(self, input_size,              # exp:(200,200)
                 input_dim,                     # exp:1
                 hidden_dim,                    # exp:32
                 kernel_size,                   # exp:(3,3)
                 return_one=False,
                 bias=True):

        super(ConvLSTM, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.use_gpu = torch.cuda.is_available()
        self.return_one = return_one
        self.bias = bias

        self.cell = ConvLSTMCell(input_size=(self.height,self.width),
                                 input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 bias=self.bias)


    # input_tensor with shape (batchsize,steps,channels,height,width)
    # input hidden_state is [h,c] list
    # for h and c with shape (batchsize,out_channels,height,width)
    # if return_one = True, return (batchsize,channels,height,width)
    # if return_one = False, return (batchsize,steps,channels,height,width)
    def forward(self, input_tensor,
                hidden_state=None):

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        seq_len = input_tensor.size(1)

        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :],
                             cur_state=[h, c])
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)

        if self.return_one:
            return layer_output[:,-1,:,:,:],[h,c]
        else:
            return layer_output, [h,c]


    def _init_hidden(self, batch_size):
        return self.cell.init_hidden(batch_size)






class FSCONV2D(nn.Module):
    def __init__(self,in_channels,                                  #Input channels of the samples
                 out_channels,                                      #Output channels of the samples
                 kernel_size,                                       #Kernel size of the convolution operation
                 stride,                                            #Stride of the convolution operation
                 padding,                                           #Padding of the convolution operation
                 bias=True):                                             #If use bias for every kernel

        super(FSCONV2D,self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x):                                           # input x with shape:(batchsize,steps,channels,width,height)
        # x: (B, T, C, H, W) or (B, C, H, W)
        if x.dim() == 5:
            # process each timestep independently, keep time dim
            out_list = []
            for t in range(x.size(1)):
                xi = x[:, t]          # (B, C, H, W)
                out_list.append(self.conv(xi))
            return torch.stack(out_list, dim=1)  # (B, T, C_out, H_out, W_out)
        elif x.dim() == 4:
            return self.conv(x)  # (B, C_out, H_out, W_out)
        else:
            raise ValueError(f"FSCONV2D: unsupported input dim {x.dim()} for shape {x.shape}")








class FSDCONV2D(nn.Module):
    def __init__(self,in_channels,                                  #Input channels of the samples
                 out_channels,                                      #Output channels of the samples
                 kernel_size,                                       #Kernel size of the convolution operation
                 stride,                                            #Stride of the convolution operation
                 padding,                                           #Padding of the convolution operation
                 bias=True):                                             #If use bias for every kernel

        super(FSDCONV2D,self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.ConvTranspose2d(in_channels=self.input_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x):                                           # input x with shape:(batchsize,steps,channels,width,height)
        # x: (B, T, C, H, W) or (B, C, H, W)
        if x.dim() == 5:
            out_list = []
            for t in range(x.size(1)):
                xi = x[:, t]   # (B, C, H, W)
                out_list.append(self.conv(xi))
            return torch.stack(out_list, dim=1)  # (B, T, C_out, H_out, W_out)
        elif x.dim() == 4:
            return self.conv(x)  # (B, C_out, H_out, W_out)
        else:
            raise ValueError(f"FSDCONV2D: unsupported input dim {x.dim()} for shape {x.shape}")







"The Implement of First Seprate Pooling Network"
class FSPOOL2D(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=(2,2)):
        super(FSPOOL2D, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)

    def forward(self, x):
        """
        Returns:
          out: (B, T, C, Hp, Wp) or (B, C, Hp, Wp)
          ind: (B, T, C, Hp, Wp) or (B, C, Hp, Wp)
        """
        is_4d = False
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W)
            is_4d = True
        if x.dim() != 5:
            raise ValueError(f"FSPOOL2D: unsupported input dim {x.dim()} for shape {x.shape}")

        out_list = []
        ind_list = []
        for t in range(x.size(1)):
            xi = x[:, t]        # (B, C, H, W)
            o, i = self.pooling(xi)  # both (B, C, Hp, Wp)
            out_list.append(o)
            ind_list.append(i)

        out = torch.stack(out_list, dim=1)  # (B, T, C, Hp, Wp)
        ind = torch.stack(ind_list, dim=1)  # (B, T, C, Hp, Wp)

        if is_4d:
            out = out[:, 0]   # (B, C, Hp, Wp)
            ind = ind[:, 0]
        return out, ind



class FSUNPOOLING(nn.Module):
    def __init__(self,kernel_size=(2,2), stride=None):
        super(FSUNPOOLING,self).__init__()
        self.kernel_size = kernel_size

        self.unpooling = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride if stride else kernel_size)

    def forward(self, x, ind):
        """
        x: (B,T,C,H,W) or (B,C,H,W)
        ind: (B,T,C,Hp,Wp) or (B,C,Hp,Wp)
        returns: (B,T,C,Hout,Wout) or (B,C,Hout,Wout)
        """
        # normalize to time dimension
        x_was_4d = False
        if x.dim() == 4:
            x = x.unsqueeze(1)   # (B,1,C,H,W)
            x_was_4d = True
        if ind.dim() == 4:
            ind = ind.unsqueeze(1)  # (B,1,C,Hp,Wp)

        if x.dim() != 5 or ind.dim() != 5:
            raise ValueError(f"FSUNPOOLING: shapes incompatible x={x.shape}, ind={ind.shape}")

        out_list = []
        for t in range(x.size(1)):
            xi = x[:, t]   # (B, C, H, W)
            ii = ind[:, t] # (B, C, Hp, Wp)
            # sanity: channels must match
            if xi.size(1) != ii.size(1):
                raise RuntimeError(f"FSUNPOOLING: channel mismatch at timestep {t}: x channels {xi.size(1)} vs ind channels {ii.size(1)}")
            out_list.append(self.unpooling(xi, ii))
        out = torch.stack(out_list, dim=1)  # (B, T, C, Hout, Wout)

        if x_was_4d:
            out = out[:, 0]
        return out



class FORECASTER_LOSS(nn.Module):
    def __init__(self):
        super(FORECASTER_LOSS,self).__init__()

    def forward(self, output,ground):
        output = output.view(-1)
        ground = ground.view(-1)
        gap = torch.abs(output-ground)
        weight = (output+ground-gap)/2
        weight = 1-weight/255.0
        weight = torch.exp(weight)
        loss = torch.mean(weight*(output-ground)*(output-ground))
        return loss
