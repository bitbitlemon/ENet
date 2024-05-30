import torch
from torch import nn
from torch.optim import Optimizer


class InitialBlock(nn.Module):
    """The initial block for ENet."""

    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()

        activation = nn.ReLU if relu else nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )

        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottleneck for ENet."""

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
                 dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("internal_ratio out of range.")

        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size),
                          padding=(0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                          padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation()
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottleneck for ENet."""

    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        self.return_indices = return_indices

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("internal_ratio out of range.")

        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation()
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        if main.is_cuda:
            padding = padding.cuda()

        main = torch.cat((main, padding), 1)
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """Upsampling bottleneck for ENet."""

    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("internal_ratio out of range.")

        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels, internal_channels, kernel_size=2, stride=2, bias=bias
        )
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)

        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):
    """ENet model."""

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)

        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0,
                             output_size=torch.Size([x.size(0), x.size(1) // 2, x.size(2) * 2, x.size(3) * 2]))
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0,
                             output_size=torch.Size([x.size(0), x.size(1) // 2, x.size(2) * 2, x.size(3) * 2]))
        x = self.regular5_1(x)

        # Final conv
        x = self.fullconv(x)

        return x


class LBFGS_Adam(Optimizer):
    """Implements a combination of L-BFGS and Adam algorithms."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, max_iter=20, history_size=100):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        max_iter=max_iter, history_size=history_size)
        super().__init__(params, defaults)

    def step(self, closure=None):
        assert closure is not None, 'LBFGS_Adam requires a closure for now'

        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr, betas, eps = group['lr'], group['betas'], group['eps']
        weight_decay = group['weight_decay']
        max_iter, history_size = group['max_iter'], group['history_size']

        state = self.state[self.param_groups[0]['params'][0]]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = []
            state['exp_avg_sq'] = []
            state['s'] = []
            state['y'] = []
            state['ro'] = []

        state['step'] += 1
        beta1, beta2 = betas

        for p in self.param_groups[0]['params']:
            if p.grad is None:
                continue

            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('LBFGS_Adam does not support sparse gradients')

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            if weight_decay != 0:
                grad = grad.add(p.data, alpha=weight_decay)

            state['exp_avg'].append(torch.zeros_like(p.data))
            state['exp_avg_sq'].append(torch.zeros_like(p.data))

            exp_avg[-1].mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq[-1].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq[-1].sqrt().add_(eps)
            p.data.addcdiv_(exp_avg[-1], denom, value=-lr)

        def evaluate(params, closure):
            for param in params:
                param.requires_grad_(False)
            loss = closure()
            for param in params:
                param.requires_grad_(True)
            return loss

        loss = evaluate(self.param_groups[0]['params'], closure)

        old_dirs, old_stps = state['s'], state['y']
        ro, H_diag = state['ro'], state['H_diag']

        flat_grad = self._gather_flat_grad()
        loss.backward()

        flat_grad.add_(state['weight_decay'], flat_params)

        if state['n_iter'] == 0:
            H_diag = 1. / (flat_grad.norm() + eps)

        state['n_iter'] += 1

        d = flat_grad
        alpha = []
        q = flat_grad

        for i in range(len(old_dirs) - 1, -1, -1):
            alpha_i = ro[i] * old_dirs[i].dot(q)
            q.add_(old_stps[i], alpha=-alpha_i)
            alpha.append(alpha_i)

        r = torch.mul(q, H_diag)
        for i in range(len(old_dirs)):
            beta = ro[i] * old_stps[i].dot(r)
            r.add_(old_dirs[i], alpha=alpha[i] - beta)

        p.data.add_(r, alpha=-lr)
        return loss


# Example usage
if __name__ == "__main__":
    model = ENet(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS_Adam(model.parameters(), lr=0.001)


    def closure():
        optimizer.zero_grad()
        output = model(torch.randn(1, 3, 224, 224))
        loss = criterion(output, torch.randint(0, 10, (1, 224, 224)))
        loss.backward()
        return loss


    optimizer.step(closure)
