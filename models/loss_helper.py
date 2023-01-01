import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
    
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.
    """

    # 读取霍夫投票的输出
    batch_size = end_points['seed_xyz'].shape[0]  # B(8), num_seed(1024), 3
    num_seed = end_points['seed_xyz'].shape[1]
    vote_xyz = end_points['vote_xyz']             # B, num_seed, 3
    seed_inds = end_points['seed_inds'].long()    # B, num_seed (index in [0,num_points-1])

    # 读取 groundtruth
    # B, num_seed (bool)
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    # B, num_seed, 3 (repeat 3 times)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3)
    # B, num_seed, 3 (dx, dy, dz)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    # B, num_seed, 3 (x, y, z)
    seed_gt_votes += end_points['seed_xyz']

    # 计算 vote_xyz 的 loss
    # B*num_seed, 1, 3
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, 1, 3)
    # B*num_seed, 1, 3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, 1, 3)
    # B*num_seed, 1
    _, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    # B, num_seed
    votes_dist = dist2.view(batch_size, num_seed)
    # average loss
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ 反映的是 proposal 对预测的自信程度.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
    """ 
    # 计算 aggregated_vote_xyz 的 loss
    # 从 1024 个 vote 里 fps 采样出 256 个 proposal。此处 xyz 是 vote_xyz，不再是点云坐标。
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']  # B(8), num_proposal(256), 3
    gt_center = end_points['center_label'][:,:,0:3]          # B, MAX_NUM_OBJ(64), 3
    B = gt_center.shape[0]            # B
    K = aggregated_vote_xyz.shape[1]  # num_proposal
    # B, num_proposal
    dist1, _, _, _ = nn_distance(aggregated_vote_xyz, gt_center)

    # 计算 label 和 mask
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    # 距离 <0.3 为 1
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    # 距离 <0.3 or >0.6 为 1，损失函数 mask
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # 计算 objectness loss
    # B, num_proposal, 2
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    return objectness_loss, objectness_label, objectness_mask

def compute_box_loss(end_points, config):
    """ Compute 3D bounding box loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
    """

    # compute center loss
    # B, num_proposal, 3
    pred_center = end_points['center']
    batch_size = pred_center.shape[0]
    num_proposal = pred_center.shape[1]
    # B, MAX_NUM_OBJ(64), 3
    gt_center = end_points['center_label'][:,:,0:3]
    batch_size = gt_center.shape[0]
    # dist1: (B,num_proposal), dist2: (B,MAX_NUM_OBJ)
    dist1, _, dist2, _ = nn_distance(pred_center, gt_center)
    objectness_label = end_points['objectness_label'].float()
    # 离 center 近的 proposal 的预测必须要准
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    # 每个物体都必须有预测
    centroid_reg_loss2 = torch.sum(dist2)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # compute heading class loss
    num_heading_bin = config.num_heading_bin  # 12
    # B, num_proposal
    heading_class_label = end_points['heading_class_label'].view(batch_size,1).repeat(1,num_proposal)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label)
    # 离 center 近的 proposal 的预测必须要准
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    
    # compute heading residual loss
    # B, num_proposal
    heading_residual_label = end_points['heading_residual_label'].view(batch_size,1).repeat(1,num_proposal)
    # 将 -15~15*np.pi/180 转换到 -1~1
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)
    # B, num_proposal, num_heading_bin
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, num_proposal, num_heading_bin).zero_()
    # num_heading_bin 的行向量为 one-hot 形式.
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)
    # B, num_proposal, num_heading_bin 逐元素相乘，再对最后一维求和。
    heading_residual_normalized_loss = huber_loss(torch.sum(
        end_points['heading_residuals_normalized']*heading_label_one_hot, -1)
        - heading_residual_normalized_label)
    # 离 center 近的 proposal 的预测必须要准
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss
    center_loss, heading_cls_loss, heading_reg_loss = \
        compute_box_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
