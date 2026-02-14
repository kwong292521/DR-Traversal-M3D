import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from experiments.config import cfg

def decode_detections(dets, info, calibs, cls_mean_size, threshold, use_3d_filter=True, to_std_offset=None, affine_T=None):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    use_3d_filter = cfg['tester'].get('use_3d_filter', True)
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            filter_score = dets[i, j, 1] if use_3d_filter else dets[i, j, -1]
            if filter_score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            if to_std_offset is not None:
                bbox = list(np.array(bbox) - np.tile(to_std_offset[i], 2))

            if affine_T is not None:
                bbox = np.array([[bbox[0], bbox[1], 1],
                                 [bbox[2], bbox[3], 1]]).T
                bbox = list((affine_T[i] @ bbox).reshape(-1))

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x3d)

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs_oracle(outputs, targets, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    bs, c, h, w = heatmap.shape
    heading = outputs['heading']
    depth = outputs['depth'][:, 0:1, :, :]
    log_sigma = outputs['depth'][:, 1:2, :, :]
    size_3d = outputs['size_3d']
    offset_3d = outputs['offset_3d']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.

    detections = torch.zeros((bs, K, 37)).cuda()
    for i in range(bs):
        n_gt = targets['mask_2d'][i].sum()
        if n_gt != 0:
            _js = torch.where(targets['mask_2d'][i])[0]
            for j in _js:
                ind = targets['indices'][i, j:j+1]
                ys = ind // w
                xs = ind % w

                offset_2d_gt = targets['offset_2d'][i, j]
                xs2d = xs + offset_2d_gt[0]
                ys2d = ys + offset_2d_gt[1]

                offset_3d_gt = targets['offset_3d'][i, j]
                xs3d = xs + offset_3d_gt[0]
                ys3d = ys + offset_3d_gt[1]

                size_2d_gt = targets['size_2d'][i, j]
                heading_bin_gt = targets['heading_bin'][i, j]
                heading_res_gt = targets['heading_res'][i, j]
                heading_gt = torch.zeros((24)).cuda()
                heading_gt[heading_bin_gt] = 1
                heading_gt[heading_bin_gt+12] = heading_res_gt
                size_3d_gt = targets['size_3d'][i, j]
                cls_ids_gt = targets['cls_ids'][i, j:j+1]

                _depth = depth[i, 0, ys, xs]
                _log_sigma = log_sigma[i, 0, ys, xs]
                _scores_2d = heatmap[i, cls_ids_gt, ys, xs]

                _scores_3d = (-_log_sigma.exp()).exp()
                _scores = _scores_2d * _scores_3d

                # detections[i, j] = torch.cat([cls_ids_gt, _scores, xs2d, ys2d, size_2d_gt, _depth, heading_gt, size_3d_gt, xs3d, ys3d, _scores_2d])
                detections[i, j] = torch.cat([cls_ids_gt, torch.tensor([1]).cuda(), xs2d, ys2d, size_2d_gt, _depth, heading_gt, size_3d_gt, xs3d, ys3d, torch.tensor([1]).cuda()])

    return detections


def extract_dets_from_outputs(outputs, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    heading = outputs['heading']
    depth = outputs['depth'][:, 0:1, :, :]
    log_sigma = outputs['depth'][:, 1:2, :, :]
    size_3d = outputs['size_3d']
    offset_3d = outputs['offset_3d']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.

    batch, channel, height, width = heatmap.size() # get shape

    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores_2d, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    offset_3d = _transpose_and_gather_feat(offset_3d, inds)
    offset_3d = offset_3d.view(batch, K, 2)
    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    heading = _transpose_and_gather_feat(heading, inds)
    heading = heading.view(batch, K, 24)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    log_sigma = _transpose_and_gather_feat(log_sigma, inds)
    log_sigma = log_sigma.view(batch, K, 1)
    size_3d = _transpose_and_gather_feat(size_3d, inds)
    size_3d = size_3d.view(batch, K, 3)
    cls_ids = cls_ids.view(batch, K, 1).float()
    scores_2d = scores_2d.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)
    
    # score fusion 
    if cfg['model']['obj_confi'] == 'dle':
        scores_3d = torch.exp(-log_sigma)
    elif cfg['model']['obj_confi'] == 'gup':
        scores_3d = (-log_sigma.exp()).exp()
    scores = scores_2d * scores_3d

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, scores_2d], dim=2)

    return detections



############### auxiliary function ############
def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask.to(torch.bool)]  # B*K*C --> M * C    


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from torch.utils.data import DataLoader

    dataset = KITTI_Dataset('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
