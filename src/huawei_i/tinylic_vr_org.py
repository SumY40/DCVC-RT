import math

import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .layers import ConvNeXtBlockAdaLN, SequentialWithEmbedding, AdaptiveAffineBCHW, CheckerboardMaskedConv2d
from .base import CompressionModel
from .utils import conv, deconv, quantize_ste, Demultiplexer, Multiplexer


class VariableRateLossyVAE(CompressionModel):
    r"""Mobile Lossy image compression framework

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    log2_e = math.log2(math.e)
    MAX_LOG_LMB = math.log(8192)

    def __init__(self, config: dict):
        super().__init__()

        self.num_slices = 5
        self.N = config['N']
        self.M = config['M']
        
        self.g_a = nn.ModuleList(config.pop('g_a'))
        self.h_a = nn.ModuleList(config.pop('h_a'))
        self.h_s = nn.ModuleList(config.pop('h_s'))
        self.g_s = nn.ModuleList(config.pop('g_s'))

        self.entropy_bottleneck = EntropyBottleneck(self.N*3//2)
        self.gaussian_conditional = GaussianConditional(None)

        in_ch_list = [0, 16, 32, 64, 128]
        out_ch_list = [16, 16, 32, 64, self.M-128]
        self.cc_transforms = nn.ModuleList(
            SequentialWithEmbedding(
                conv(2*self.M + in_ch_list[i], 224, stride=1, kernel_size=5),
                nn.GELU(),
                AdaptiveAffineBCHW(224),
                conv(224, 128, stride=1, kernel_size=5),
                nn.GELU(),
                AdaptiveAffineBCHW(128),
                conv(128, 2*out_ch_list[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.sc_transforms = nn.ModuleList(
            CheckerboardMaskedConv2d(
                out_ch_list[i], 2*out_ch_list[i], kernel_size=5, padding=2, stride=1
            ) for i in range(self.num_slices)
        )
        self.entropy_parameters = nn.ModuleList(
            SequentialWithEmbedding(
                conv(2*self.M + 12//3*out_ch_list[i], 10//3*out_ch_list[i], 1, 1),
                nn.GELU(),
                AdaptiveAffineBCHW(10//3*out_ch_list[i]),
                conv(10//3*out_ch_list[i], 8//3*out_ch_list[i], 1, 1),
                nn.GELU(),
                AdaptiveAffineBCHW(8//3*out_ch_list[i]),
                conv(8//3*out_ch_list[i], 6//3*out_ch_list[i], 1, 1),
            ) for i in range(self.num_slices)
        )

        self._setup_lmb_embedding(config)

    def _setup_lmb_embedding(self, config):
        _low, _high = config['lmb_range']
        self.lmb_range = (float(_low), float(_high))
        self.default_lmb = self.lmb_range[1]
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
            nn.Linear(self.lmb_embed_dim[0], self.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(self.lmb_embed_dim[1], self.lmb_embed_dim[1]),
        )
        self._sin_period = config['sin_period']

    def sample_lmb(self, n: int):
        low, high = self.lmb_range # original lmb space, 16 to 1024
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device="cuda")
        lmb = torch.pow(transformed_lmb, exponent=p)
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (n,)
        return lmb
    
    def sinusoidal_embedding(self, values: torch.Tensor, dim=256, max_period=64):
        assert values.dim() == 1 and (dim % 2) == 0
        exponents = torch.linspace(0, 1, steps=(dim // 2))
        freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
        args = values.view(-1, 1) * freqs.view(1, dim//2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def get_lmb_embedding(self, lmb: torch.Tensor):
        assert isinstance(lmb, torch.Tensor) and lmb.dim() == 1
        scaled = torch.log(lmb) * self._sin_period / self.MAX_LOG_LMB
        embedding = self.sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0],
                                                max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def forward(self, x):
        B = x.shape[0]
        if self.training:
            lmb = self.sample_lmb(n=B)
        else:
            lmb = torch.full((B,), self.default_lmb, device=x.device)
        assert lmb.shape == (B,)
        lmb_emb = self.get_lmb_embedding(lmb)

        y = x
        for _, block in enumerate(self.g_a):
            if getattr(block, 'requires_embedding', False):
                y = block(y, lmb_emb)
            else:
                y = block(y)
        
        z = y
        for _, block in enumerate(self.h_a):
            if getattr(block, 'requires_embedding', False):
                z = block(z, lmb_emb)
            else:
                z = block(z)
            
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        
        params = z_hat
        for _, block in enumerate(self.h_s):
            if getattr(block, 'requires_embedding', False):
                params = block(params, lmb_emb)
            else:
                params = block(params)
        
        y_slices = y.split([16, 16, 32, 64, self.M-128], 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices, lmb_emb)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1), lmb_emb
                )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            y_half = y_hat_slice.clone()
            y_half[:, :, 0::2, 0::2] = 0
            y_half[:, :, 1::2, 1::2] = 0

            sc_params = self.sc_transforms[slice_index](y_half)
            sc_params[:, :, 0::2, 1::2] = 0
            sc_params[:, :, 1::2, 0::2] = 0
            
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1), lmb_emb
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
            y_hat_slices.append(y_hat_slice)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
            y_likelihood.append(y_slice_likelihood)
       
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # Generate the image reconstruction.
        x_hat = y_hat
        for _, block in enumerate(self.g_s):
            if getattr(block, 'requires_embedding', False):
                x_hat = block(x_hat, lmb_emb)
            else:
                x_hat = block(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lmb":lmb,
        }
        
        
    def compress(self, x, testlmb):
        
        B = x.shape[0]
        # lmb = torch.full((B,), testlmb, device=x.device)
        lmb = torch.full((B,), testlmb, device='cuda')
        assert lmb.shape == (B,)
        lmb_emb = self.get_lmb_embedding(lmb)

        y = x
        for _, block in enumerate(self.g_a):
            if getattr(block, 'requires_embedding', False):
                y = block(y, lmb_emb)
            else:
                y = block(y)
        
        z = y
        for _, block in enumerate(self.h_a):
            if getattr(block, 'requires_embedding', False):
                z = block(z, lmb_emb)
            else:
                z = block(z)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = z_hat
        for _, block in enumerate(self.h_s):
            if getattr(block, 'requires_embedding', False):
                params = block(params, lmb_emb)
            else:
                params = block(params)

        y_slices = y.split([16, 16, 32, 64, self.M-128], 1)
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
            cc_params = self.cc_transforms[slice_index](support_slices,lmb_emb)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1),lmb_emb
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
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
                torch.cat((params, sc_params, cc_params), dim=1),lmb_emb
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
            y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

            symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:]
                }

    def decompress(self, strings, shape,testlmb):

        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        lmb = torch.full(z_hat.shape[0:1], testlmb,device='cuda')
        lmb_emb = self.get_lmb_embedding(lmb)


        params = z_hat
        for _, block in enumerate(self.h_s):
            if getattr(block, 'requires_embedding', False):
                params = block(params, lmb_emb)
            else:
                params = block(params)

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices,lmb_emb)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1),lmb_emb
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            scales_hat_anchor, _ = Demultiplexer(scales_hat)
            means_hat_anchor, _ = Demultiplexer(means_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
            y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

            y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            sc_params = self.sc_transforms[slice_index](y_hat_slice)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1),lmb_emb
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)
            
            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = y_hat
        for _, block in enumerate(self.g_s):
            if getattr(block, 'requires_embedding', False):
                x_hat = block(x_hat, lmb_emb)
            else:
                x_hat = block(x_hat)

        return {"x_hat": x_hat}



def tinylic(lmb_range=(32,8192)):
    cfg = dict()

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    N = 128
    M = 320
    depths = [2, 2, 6, 2, 2, 2]

    res_block = ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    cfg['g_a'] = [
        conv(3, N, kernel_size=5, stride=2),
        *[res_block(N) for _ in range(depths[0])],
        conv(N, N*3//2, kernel_size=3, stride=2),
        *[res_block(N*3//2) for _ in range(depths[1])],
        conv(N*3//2, N*2, kernel_size=3, stride=2),
        *[res_block(N*2) for _ in range(depths[2])],
        conv(N*2, M, kernel_size=3, stride=2),
        *[res_block(M) for _ in range(depths[3])],
    ]
    cfg['h_a'] = [
        conv(M, N*3//2, kernel_size=3, stride=2),
        *[res_block(N*3//2, kernel_size=3) for _ in range(depths[4])],
        conv(N*3//2, N*3//2, kernel_size=3, stride=2),
        *[res_block(N*3//2, kernel_size=3) for _ in range(depths[5])],
    ]

    depths = depths[::-1]
    cfg['h_s'] = [
        *[res_block(N*3//2, kernel_size=3) for _ in range(depths[0])],
        deconv(N*3//2, N*3//2, kernel_size=3, stride=2),
        *[res_block(N*3//2, kernel_size=3) for _ in range(depths[1])],
        deconv(N*3//2, M*2, kernel_size=3, stride=2),
    ]
    cfg['g_s'] = [
        *[res_block(M, kernel_size=3) for _ in range(depths[2])],
        deconv(M, N*2, kernel_size=3, stride=2),
        *[res_block(N*2, kernel_size=3) for _ in range(depths[3])],
        deconv(N*2, N*3//2, kernel_size=3, stride=2),
        *[res_block(N*3//2, kernel_size=3) for _ in range(depths[4])],
        deconv(N*3//2, N, kernel_size=3, stride=2),
        *[res_block(N, kernel_size=3) for _ in range(depths[5])],
        deconv(N, 3, kernel_size=5, stride=2),
    ]
    cfg['N'] = N
    cfg['M'] = M

    model = VariableRateLossyVAE(cfg)

    return model





