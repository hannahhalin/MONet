'''
Adapted from the official implementation of the IRR-PWC model.
https://github.com/visinf/irr.git
'''

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import conv, upsample2d_as, downsample2d_as, rescale_flow, initialize_msra, concat_and_sync_dims, compute_cost_volume
from .pwc_modules import WarpingLayer, DownWarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense, OccContextNetwork, OccEstimatorDense
from .pwc_modules import OccFusionNetwork
from .irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc

import copy


class PWCNet(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(PWCNet, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        # The feature extractor output feature pyramid from coarse to 
        # fine resolutions, in total 6.
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in_flo = self.dim_corr + 32 + 2
        self.num_ch_in_occ = self.dim_corr + 32 + 1

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in_flo)
        self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
        self.occ_estimators = OccEstimatorDense(self.num_ch_in_occ)
        self.occ_context_networks = OccContextNetwork(self.num_ch_in_occ + 448 + 1)
        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1)])

        self.conv_1x1_1 = conv(16, 3, kernel_size=1, stride=1, dilation=1)

        self.refine_flow = RefineFlow(2 + 1 + 32)
        self.refine_occ = RefineOcc(1 + 32 + 32)

        self.fuse_occs = OccFusionNetwork(self.output_level + 1)
        
        self.corr_params = {
            "pad_size": self.search_range, "kernel_size": 1, 
            "max_disp": self.search_range, "stride1": 1, 
            "stride2": 1, "corr_multiply": 1}

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        batch_size, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        output_dict_eval = {}
        flows = []
        occs = []

        _, _, h0_x1, w0_x1, = x1_pyramid[0].size()
        _, _, hp_x1, wp_x1, = x1_pyramid[self.output_level].size()
        if self.args.cuda:
            flow_f = torch.zeros(batch_size, 2, h0_x1, w0_x1).float().cuda()
            flow_b = torch.zeros(batch_size, 2, h0_x1, w0_x1).float().cuda()
            occ_f = torch.zeros(batch_size, 1, hp_x1, wp_x1).float().cuda()
            occ_b = torch.zeros(batch_size, 1, hp_x1, wp_x1).float().cuda()
        else:
            flow_f = torch.zeros(batch_size, 2, h0_x1, w0_x1).float()
            flow_b = torch.zeros(batch_size, 2, h0_x1, w0_x1).float()
            occ_f = torch.zeros(batch_size, 1, hp_x1, wp_x1).float()
            occ_b = torch.zeros(batch_size, 1, hp_x1, wp_x1).float()

        # Predicts flow.
        out_corr_relu_fs, out_corr_relu_bs = [], []
        x1_1by1s, x2_1by1s = [], []
        x1_1by1_warps, x2_1by1_warps = [], []
        for l in range(self.output_level+1):
            x1, x2 = x1_pyramid[l], x2_pyramid[l]

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)

            # correlation
            out_corr_f = compute_cost_volume(x1, x2_warp, self.corr_params)
            out_corr_b = compute_cost_volume(x2, x1_warp, self.corr_params)
                
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # Records for occ prediction.
            out_corr_relu_fs.append(out_corr_relu_f)
            out_corr_relu_bs.append(out_corr_relu_b)

            if l != self.output_level:
                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)
            else:
                x1_1by1 = x1
                x2_1by1 = x2

            # Records for occ prediction.
            x1_1by1s.append(x1_1by1)
            x2_1by1s.append(x2_1by1)

            # concat and estimate flow
            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

            x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
            flow_est_f = flow_f + flow_res_f
            flow_est_b = flow_b + flow_res_b

            flow_cont_f = flow_est_f + self.context_networks(torch.cat([x_intm_f, flow_est_f], dim=1))
            flow_cont_b = flow_est_b + self.context_networks(torch.cat([x_intm_b, flow_est_b], dim=1))
            
            # refinement
            img1_resize = upsample2d_as(x1_raw, flow_f, mode="bilinear")
            img2_resize = upsample2d_as(x2_raw, flow_b, mode="bilinear")
            img2_warp = self.warping_layer(img2_resize, rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)
            img1_warp = self.warping_layer(img1_resize, rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)

            # flow refine
            flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
            flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)

            flow_cont_f = rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False)
            flow_cont_b = rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False)
            flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
            flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

            # occ refine
            x2_1by1_warp = self.warping_layer(x2_1by1, flow_f, height_im, width_im, self._div_flow)
            x1_1by1_warp = self.warping_layer(x1_1by1, flow_b, height_im, width_im, self._div_flow)

            x2_1by1_warps.append(x2_1by1_warp)
            x1_1by1_warps.append(x1_1by1_warp)

            flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])

        # Predicts occlusion.
        for l in range(self.output_level, -1, -1):
            x1, x2 = x1_pyramid[l], x2_pyramid[l]

            # warping
            if l != self.output_level:
                occ_f = downsample2d_as(occ_f, x1, mode="bilinear")
                occ_b = downsample2d_as(occ_b, x2, mode="bilinear")

            # correlation
            out_corr_relu_f = out_corr_relu_fs[l]
            out_corr_relu_b = out_corr_relu_bs[l]

            x1_1by1 = x1_1by1s[l]
            x2_1by1 = x2_1by1s[l]

            # occ estimation
            x_intm_occ_f, occ_res_f = self.occ_estimators(torch.cat([out_corr_relu_f, x1_1by1, occ_f], dim=1))
            x_intm_occ_b, occ_res_b = self.occ_estimators(torch.cat([out_corr_relu_b, x2_1by1, occ_b], dim=1))
            occ_est_f = occ_f + occ_res_f
            occ_est_b = occ_b + occ_res_b

            occ_cont_f = occ_est_f + self.occ_context_networks(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
            occ_cont_b = occ_est_b + self.occ_context_networks(torch.cat([x_intm_occ_b, occ_est_b], dim=1))

            # occ refine
            x2_1by1_warp = x2_1by1_warps[l]
            x1_1by1_warp = x1_1by1_warps[l]

            occ_f = self.refine_occ(occ_cont_f.detach(), x1_1by1, x1_1by1 - x2_1by1_warp)
            occ_b = self.refine_occ(occ_cont_b.detach(), x2_1by1, x2_1by1 - x1_1by1_warp)

            occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])

        # Computes the concatenated intermediate predicted flows and occs.
        fused_occs = []
        for idx in range(4):
            fused_occs.append(
                self.fuse_occs(concat_and_sync_dims([occ[idx] for occ in occs])))
        # Gurantee the order is pred6,5,4,3,2
        occs.reverse()
        # The order becomes pred6,5,4,3,2,fused
        occs.append(fused_occs)

        occ_f = fused_occs[2]
        occ_b = fused_occs[3]

        for l in range(self.output_level+1, self.num_levels):
            x1, x2 = x1_pyramid[l], x2_pyramid[l]

            flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
            flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
            flows.append([flow_f, flow_b])

            x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
            x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
            flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im, self._div_flow)
            flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im, self._div_flow)

            if l != self.num_levels-1:
                x1_in = self.conv_1x1_1(x1)
                x2_in = self.conv_1x1_1(x2)
                x1_w_in = self.conv_1x1_1(x1_warp)
                x2_w_in = self.conv_1x1_1(x2_warp)
            else:
                x1_in = x1
                x2_in = x2
                x1_w_in = x1_warp
                x2_w_in = x2_warp

            occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([x1_in, x2_w_in, flow_f, flow_b_warp], dim=1))
            occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([x2_in, x1_w_in, flow_b, flow_f_warp], dim=1))

            occs.append([occ_f, occ_b])

        output_dict_eval['flow'] = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        output_dict_eval['occ'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")
        
        for i, occ_pair in enumerate(occs):
            idx = 10
            if len(occ_pair) == 4:
                idx = 2
            elif len(occ_pair) == 2:
                idx = 0
            output_dict_eval['occ_' + str(self.num_levels-i)] = upsample2d_as(occ_pair[idx], x1_raw, mode="bilinear")

        output_dict['flow'] = flows
        output_dict['occ'] = occs

        if self.training:
            return output_dict
        else:
            return output_dict_eval