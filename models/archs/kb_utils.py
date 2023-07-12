import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LayerNormFunction(paddle.autograd.PyLayer):
    @staticmethod
    
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.shape #size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        # weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        #y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        y = paddle.reshape(weight, [1, C, 1, 1]) * y + paddle.reshape(bias, [1, C, 1, 1])
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / paddle.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Layer):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        #self.register_parameter('weight', nn.Parameter(paddle.ones(channels), requires_grad=requires_grad))
        x=paddle.ones(shape=[channels])
        self.weight = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        #self.register_parameter('bias', nn.Parameter(paddle.zeros(channels), requires_grad=requires_grad))
        x=paddle.zeros(shape=[channels])
        self.bias = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Layer):
    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1) #dim
        return x1 * x2


class KBAFunction(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = paddle.reshape(att,[B, nset, H * W])
        att = paddle.transpose(att,[0,2,1])

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = paddle.nn.functional.unfold(x, selfk, selfk // 2)

        # for unfold att / less memory cost
        uf = paddle.reshape(uf,[B, selfg, selfc // selfg * KK, H * W]).permute(0, 3, 1, 2)
        attk = paddle.reshape(attk,[B, H * W, selfg, selfc // selfg, selfc // selfg * KK])

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = paddle.transpose(x,[0,2,1])
        x = paddle.reshape(x,[B, selfc, H, W])
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = paddle.reshape(grad_output,[B, selfc, H * W])
        dbias = paddle.transpose(dbias,[0,2,1])

        dselfb = paddle.transpose(att,[0,2,1])
        dselfb = dselfb @ dbias
        datt = dbias @ paddle.transpose(dselfb,[0,2,1])

        attk = att @ selfw
        uf = F.Unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        #dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)
        dx = paddle.reshape(dbias, [B, H*W, selfg, selfc // selfg, 1])

        dattk = dx @ paddle.reshape(uf, [B, H*W, selfg, 1, selfc // selfg * KK])
        duf = paddle.transpose(dattk,[0,1,2,4,3]) @ dx
        del attk, uf

        dattk=paddle.reshape(dattk, [B, H * W, -1])
        datt += dattk @ paddle.transpose(selfw,[0,2,1])
        dselfw = paddle.transpose(att,[0,2,1]) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = paddle.transpose(datt,[0,2,1])
        datt = paddle.reshape(datt,[B, nset, H, W])

        return dx, datt, None, None, dselfb, dselfw
