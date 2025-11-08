import math
import sys
sys.path.append('/workspace/test')
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import vgg19

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .layers import MetaNeXtStage, CheckerboardMaskedConv2d
from .base import CompressionModel
from .utils import conv, deconv, gconv, quantize_ste, Demultiplexer, Multiplexer, CharbonnierLoss, VGGLoss_ESRGAN


class ScalingNet(nn.Module):
    def __init__(self, channel):
        super(ScalingNet, self).__init__()
        self.channel = int(channel)
        self.fc1 = nn.Linear(1, channel // 2, bias=True)
        self.fc2 = nn.Linear(channel // 2, channel, bias=True)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, lambda_rd):
        b, c, _, _ = x.size()
        scaling_vector = torch.exp(10 * self.fc2(F.relu(self.fc1(lambda_rd))))
        scaling_vector = scaling_vector.view(b, c, 1, 1)
        x_scaled = x * scaling_vector.expand_as(x)
        return x_scaled

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mae = nn.L1Loss()

    def gram_matrix(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def forward(self,target_feature, inputs):
        G = self.gram_matrix(inputs)
        target = self.gram_matrix(target_feature)
        self.loss = self.mae(G, target)
        return self.loss 

class HyperSD(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel) -> None:
        super().__init__()
        self.in_c = in_channel
        self.mid_c = mid_channel
        self.out_c = out_channel
        self.conv1 = nn.ConvTranspose2d(in_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.conv4(x)
        x = self.dequant(x)
        return x



class TinyLIC(CompressionModel):
    r"""Mobile Lossy image compression framework
    "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation"

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=320, w_adv=0.05, w_vgg=1, model_size = "80M"):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        self.num_slices = 5
        self.w_adv = w_adv
        self.w_vgg = w_vgg
        self.model_size = model_size

        if self.model_size == "40M":
            N = 64
            M = 64
            self.M = M
            in_ch_list = [0, 4, 8, 16, 32]
            out_ch_list = [4, 4, 8, 16, self.M - 32]

        elif self.model_size == "50M":
            self.num_slices = 4
            N = 64
            M = 128
            self.M = M
            in_ch_list = [0, 16, 32, 64]
            out_ch_list = [16, 16, 32, self.M - 64]

        elif self.model_size == "80M":
            N = 96
            M = 128
            self.M = M
            in_ch_list = [0, 8, 16, 32, 64]
            out_ch_list = [8, 8, 16, 32, self.M - 64]

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = MetaNeXtStage(dim=N, depth=depths[0])
        self.g_a_scale0 = ScalingNet(channel=N)

        self.g_a2 = conv(N, N * 3 // 2, kernel_size=3, stride=2)
        self.g_a3 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[1])
        self.g_a_scale1 = ScalingNet(channel=N * 3 // 2)

        self.g_a4 = conv(N * 3 // 2, N * 2, kernel_size=3, stride=2)
        self.g_a5 = MetaNeXtStage(dim=N * 2, depth=depths[2])
        self.g_a_scale2 = ScalingNet(channel=N * 2)

        self.g_a6 = conv(N * 2, M, kernel_size=3, stride=2)
        self.g_a7 = MetaNeXtStage(dim=M, depth=depths[3])
        self.g_a_scale3 = ScalingNet(channel=M)

        self.h_a0 = conv(M, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a1 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[4])
        self.h_a2 = conv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_a3 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[5])

        depths = depths[::-1]
        self.h_s0 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[0])
        self.h_s1 = deconv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=2)
        self.h_s2 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[1])
        self.h_s3 = deconv(N * 3 // 2, M * 2, kernel_size=3, stride=2)

        self.h_sd = HyperSD(N * 3 // 2, M * 5 // 4, M)

        self.g_s_scale0 = ScalingNet(channel=M)
        self.g_s_mr0 = ScalingNet(channel=M)
        self.g_s0 = MetaNeXtStage(dim=M, depth=depths[2])
        self.g_s1 = deconv(M, N * 2, kernel_size=3, stride=2)

        self.g_s_scale1 = ScalingNet(channel=N * 2)
        self.g_s_mr1 = ScalingNet(channel=N * 2)
        self.g_s2 = MetaNeXtStage(dim=N * 2, depth=depths[3])
        self.g_s3 = deconv(N * 2, N * 3 // 2, kernel_size=3, stride=2)

        self.g_s_scale2 = ScalingNet(channel=N * 3 // 2)
        self.g_s_mr2 = ScalingNet(channel=N * 3 // 2)
        self.g_s4 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[4])
        self.g_s5 = deconv(N * 3 // 2, N, kernel_size=3, stride=2)

        self.g_s_scale3 = ScalingNet(channel=N)
        self.g_s_mr3 = ScalingNet(channel=N)
        self.g_s6 = MetaNeXtStage(dim=N, depth=depths[5]) #gelu leakyrelu no layernorm
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N * 3 // 2)
        self.gaussian_conditional = GaussianConditional(None)

        # self.cc_transforms = nn.ModuleList(
        #     nn.Sequential(
        #         conv(2 * M + in_ch_list[i], 256, stride=1, kernel_size=5),#gconv leaky256
        #         nn.LeakyReLU(),
        #         conv(256, 128, stride=1, kernel_size=5),
        #         nn.LeakyReLU(),
        #         conv(128, 2 * out_ch_list[i], stride=1, kernel_size=3),
        #     ) for i in range(self.num_slices)
        # )
        
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                gconv(2 * M + in_ch_list[i], 256, stride=1, kernel_size=5),#gconv leaky256
                nn.LeakyReLU(),
                gconv(256, 128, stride=1, kernel_size=5),
                nn.LeakyReLU(),
                conv(128, 2 * out_ch_list[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.sc_transforms = nn.ModuleList(
            CheckerboardMaskedConv2d(
                out_ch_list[i], 2 * out_ch_list[i], kernel_size=5, padding=2, stride=1
            ) for i in range(self.num_slices)
        )
        # self.entropy_parameters = nn.ModuleList(
        #     nn.Sequential(
        #         conv(2 * M + 12 // 3 * out_ch_list[i], 10 // 3 * out_ch_list[i], 1, 1),
        #         nn.GELU(),
        #         conv(10 // 3 * out_ch_list[i], 8 // 3 * out_ch_list[i], 1, 1),
        #         nn.GELU(),
        #         conv(8 // 3 * out_ch_list[i], 6 // 3 * out_ch_list[i], 1, 1),
        #     ) for i in range(self.num_slices)
        # )
        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                conv(2 * M + 12 // 3 * out_ch_list[i], 10 // 3 * out_ch_list[i], 1, 1),
                nn.LeakyReLU(),
                conv(10 // 3 * out_ch_list[i], 8 // 3 * out_ch_list[i], 1, 1),
                nn.LeakyReLU(),
                conv(8 // 3 * out_ch_list[i], 3 // 3 * out_ch_list[i], 1, 1),
            ) for i in range(self.num_slices)
        )


    def g_a(self, x, lambda_rd):
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a_scale0(x, lambda_rd)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a_scale1(x, lambda_rd)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a_scale2(x, lambda_rd)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a_scale3(x, lambda_rd)
        return x

    def g_s(self, x, lambda_rd):
        x = self.g_s_scale0(x, lambda_rd)
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s_scale1(x, lambda_rd)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s_scale2(x, lambda_rd)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s_scale3(x, lambda_rd)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a1(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def h_s(self, x):
        x = self.h_s0(x)
        x = self.h_s1(x)
        x = self.h_s2(x)
        x = self.h_s3(x)
        return x
       
    # 正常编码器和解码器一起训练
    def forward(self, x, lambda_rd):
        y = self.g_a(x, lambda_rd)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.h_s(z_hat)
        scales_hat = self.h_sd(z_hat)
        scales_hat = scales_hat.exp()#

        if self.model_size == "40M":
            y_slices = y.split([4, 4, 8, 16, self.M - 32], 1)
        elif self.model_size == "50M":
            y_slices = y.split([16, 16, 32, self.M - 64], 1)
        elif self.model_size == "80M":
            y_slices = y.split([8, 8, 16, 32, self.M - 64], 1)
            scales_hat_slices = scales_hat.split([8, 8, 16, 32, self.M - 64], 1)#

        y_hat_slices = []
        y_likelihood = []

        for slice_index, (scales_hat, y_slice) in enumerate(zip(scales_hat_slices, y_slices)):
            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)
            means_hat = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
            )
            means_hat_anchor, _ = Demultiplexer(means_hat)
            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            y_half = y_hat_slice.clone()
            y_half[:, :, 0::2, 0::2] = 0
            y_half[:, :, 1::2, 1::2] = 0

            sc_params = self.sc_transforms[slice_index](y_half)
            sc_params[:, :, 0::2, 1::2] = 0
            sc_params[:, :, 1::2, 0::2] = 0

            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)
            means_hat = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            means_hat = Multiplexer(means_hat_anchor, means_hat_non_anchor)

            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
            y_hat_slices.append(y_hat_slice)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
            y_likelihood.append(y_slice_likelihood)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # Generate the image reconstruction.
        x_hat = self.g_s(y_hat, lambda_rd)


        return {"x_hat": x_hat,"y_hat": y_hat,"likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}
        # return {"x_hat": x_hat,"likelihoods": {"y": y_likelihoods, "z": z_likelihoods},}

    def encode(self, x, lambda_rd, sps_id=0, output_file=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        compressed = self.compress(x, lambda_rd)
        bit_stream = compressed['bit_stream']
# 遍历每个字符串列表
        written = write_ip(output_file, True, sps_id, bit_stream)
        result = {
            'bit': written * 8,
            'x_hat': compressed['x_hat'],
        }
        return result

    def compress(self, x, lambda_rd):
        y = self.g_a(x, lambda_rd)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        scales_hat = self.h_sd(z_hat)
        scales_hat = scales_hat.exp()
        #scale_hat = 

        if self.model_size == "40M":
            y_slices = y.split([4, 4, 8, 16, self.M - 32], 1)
        elif self.model_size == "50M":
            y_slices = y.split([16, 16, 32, self.M - 64], 1)
        elif self.model_size == "80M":
            y_slices = y.split([8, 8, 16, 32, self.M - 64], 1)
            scales_hat_slices = scales_hat.split([8, 8, 16, 32, self.M - 64], 1)

        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, (scales_hat, y_slice) in enumerate(zip(scales_hat_slices, y_slices)):
            scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
            y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)

            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)

            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)

            # scales_hat_anchor, _ = Demultiplexer(scales_hat)
            # means_hat_anchor, _ = Demultiplexer(means_hat)

            means_hat = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
            )
            means_hat_anchor, _ = Demultiplexer(means_hat)

            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
            y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

            symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
            indexes_list.extend(index_anchor.reshape(-1).tolist())

            y_half = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))

            sc_params = self.sc_transforms[slice_index](y_half)
            sc_params[:, :, 0::2, 1::2] = 0
            sc_params[:, :, 1::2, 0::2] = 0

            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)

            # _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            # _, means_hat_non_anchor = Demultiplexer(means_hat)
            means_hat = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
            )
            _, means_hat_non_anchor = Demultiplexer(means_hat)

            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols",
                                                                      means_hat_non_anchor)
            y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

            symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        x_hat = self.g_s(y_hat, lambda_rd)

        return {
                "strings": [y_strings, z_strings],
                "bit_stream": y_strings + z_strings,
                "shape": z.size()[-2:],
                "x_hat": x_hat,
                }

    def decompress(self, strings, shape, lambda_rd):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)
        scales_hat = self.h_sd(z_hat)
        scales_hat = scales_hat.exp()

        # if self.model_size == "40M":
        #      = y.split([4, 4, 8, 16, self.M - 32], 1)
        # elif self.model_size == "50M":
        #     y_slices = y.split([16, 16, 32, self.M - 64], 1)
        #if self.model_size == "80M":

        scales_hat_slices = scales_hat.split([8, 8, 16, 32, self.M - 64], 1)


        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)
        indexes_list = []

        for slice_index,scales_hat in enumerate(scales_hat_slices):
            scales_hat_anchor, scales_hat_non_anchor = Demultiplexer(scales_hat)
            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)

            # scales_hat_anchor, _ = Demultiplexer(scales_hat)
            # means_hat_anchor, _ = Demultiplexer(means_hat)
            means_hat = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
            )
            means_hat_anchor, _ = Demultiplexer(means_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)

            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

            y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            sc_params = self.sc_transforms[slice_index](y_hat_slice)
            # gaussian_params = self.entropy_parameters[slice_index](
            #     torch.cat((params, sc_params, cc_params), dim=1)
            # )
            # scales_hat, means_hat = gaussian_params.chunk(2, 1)

            # _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            # _, means_hat_non_anchor = Demultiplexer(means_hat)
            means_hat = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
            )
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        
        x_hat = self.g_s(y_hat, lambda_rd)
        # x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}

    def compress_split(self, x, lambda_rd, slices):
        y = self.g_a(x, lambda_rd)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        if self.model_size == "40M":
            y_slices = y.split([4, 4, 8, 16, self.M - 32], 1)
        elif self.model_size == "50M":
            y_slices = y.split([16, 16, 32, self.M - 64], 1)
        elif self.model_size == "80M":
            y_slices = y.split([8, 8, 16, 32, self.M - 64], 1)

        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)

            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index < slices:
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
                y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

                symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_half = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))

                sc_params = self.sc_transforms[slice_index](y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0

                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)

                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols",
                                                                          means_hat_non_anchor)
                y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

                symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
                indexes_list.extend(index_non_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            else:
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)

                y_half = Multiplexer(means_hat_anchor, torch.zeros_like(means_hat_anchor))

                sc_params = self.sc_transforms[slice_index](y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0

                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)

                y_hat_slice = Multiplexer(means_hat_anchor, means_hat_non_anchor)

            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress_split(self, strings, shape, lambda_rd, slices):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            if slice_index < slices:
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
                y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
                sc_params = self.sc_transforms[slice_index](y_hat_slice)
                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)

                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
                y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            else:
                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)

                y_half = Multiplexer(means_hat_anchor, torch.zeros_like(means_hat_anchor))

                sc_params = self.sc_transforms[slice_index](y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0

                gaussian_params = self.entropy_parameters[slice_index](
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)

                y_hat_slice = Multiplexer(means_hat_anchor, means_hat_non_anchor)
            y_hat_slices.append(y_hat_slice)


        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, lambda_rd).clamp_(0, 1)

        return {"x_hat": x_hat}

