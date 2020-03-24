import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from fcos.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from fcos.utils.comm import reduce_sum
from fcos.layers import ml_nms
from detectron2.layers import interpolate

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores
    
"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOSOutputs(object):
    def __init__(
            self,
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            controllers, 
            masks,
            gt_instances=None,
    ):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.ctrness_pred = ctrness_pred
        self.locations = locations

        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr
        self.controllers = controllers
        self.masks = masks

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, self.gt_instances, loc_to_size_range
        )

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        matched_idxes = []
        im_idxes = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            matched_idxes.append(locations_to_gt_inds)
            im_idxes.append(torch.tensor([im_i]*len(labels_per_im)).to(locations_to_gt_inds.device))
        return {"labels": labels, "reg_targets": reg_targets, "matched_idxes": matched_idxes, "im_idxes": im_idxes}

    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets, matched_idxes, im_idxes = training_targets["labels"], training_targets["reg_targets"], training_targets["matched_idxes"], training_targets["im_idxes"]

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.logits_pred
            ], dim=0,)
        reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.reg_pred
            ], dim=0,)
        ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.ctrness_pred
            ], dim=0,)

        labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels
            ], dim=0,)

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0,)
        
        matched_idxes = cat(
            [
                x.reshape(-1) for x in matched_idxes
            ], dim=0,)

        im_idxes = cat(
            [
                x.reshape(-1) for x in im_idxes
            ], dim=0,)

        controllers_pred = cat(
            [
                x.permute(0, 2, 3, 1).reshape(-1, 169) for x in self.controllers
            ], dim=0,)

        return self.fcos_losses(
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            ctrness_pred,
            controllers_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            matched_idxes,
            im_idxes
        )

    def predict_proposals(self):
        sampled_boxes = []

        bundle = (
            self.locations, self.logits_pred,
            self.reg_pred, self.ctrness_pred,
            self.strides
        )

        for i, (l, o, r, c, s) in enumerate(zip(*bundle)):
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            r = r * s
            controller = self.controllers[i]
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, controller, self.image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        # for CondInst
        boxlists = self.forward_for_mask(boxlists)

        return boxlists

    def forward_for_mask(self, boxlists):
        N, dim, h, w = self.masks.shape
        grid_x = torch.arange(w).view(1,-1).float().repeat(h,1).cuda() / (w-1) * 2 - 1
        grid_y = torch.arange(h).view(-1,1).float().repeat(1,w).cuda() / (h-1) * 2 - 1
        x_map = grid_x.view(1, 1, h, w).repeat(N, 1, 1, 1)
        y_map = grid_y.view(1, 1, h, w).repeat(N, 1, 1, 1)
        masks_feat = torch.cat((self.masks, x_map, y_map), dim=1)
        o_h = int(h * self.strides[0])
        o_w = int(w * self.strides[0])
        for im in range(N):
            boxlist = boxlists[im]
            input_h, input_w = boxlist.image_size
            mask = masks_feat[None, im]
            ins_num = boxlist.controllers.shape[0]
            weights1 = boxlist.controllers[:,:80].reshape(-1,8,10).reshape(-1,10).unsqueeze(-1).unsqueeze(-1)
            bias1 = boxlist.controllers[:, 80:88].flatten()
            weights2 = boxlist.controllers[:, 88:152].reshape(-1,8,8).reshape(-1,8).unsqueeze(-1).unsqueeze(-1)
            bias2 = boxlist.controllers[:, 152:160].flatten()
            weights3 = boxlist.controllers[:, 160:168].unsqueeze(-1).unsqueeze(-1)
            bias3 = boxlist.controllers[:,168:169].flatten()
            
            conv1 = F.conv2d(mask,weights1,bias1).relu()
            conv2 = F.conv2d(conv1, weights2, bias2, groups = ins_num).relu()
            masks_per_image = F.conv2d(conv2, weights3, bias3, groups = ins_num).sigmoid()
            masks = interpolate(masks_per_image, size = (o_h,o_w), mode="bilinear", align_corners=False)
            masks = masks[:, :, :input_h, :input_w].permute(1,0,2,3)
            boxlist.pred_masks = masks
        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness, controller, image_sizes
    ):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()
        controller = controller.view(N, 169, H, W).permute(0, 2, 3, 1)
        controller = controller.reshape(N, -1, 169)
        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_controller = controller[i]
            per_controller = per_controller[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_controller = per_controller[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.controllers = per_controller

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def prepare_masks(self, m_h, m_w, r_h, r_w, targets_masks):
        masks = []
        for im_i in range(len(targets_masks)):
            mask_t = targets_masks[im_i]
            if len(mask_t) == 0:
                masks.append(mask_t.new_tensor([]))
                continue
            n, h, w = mask_t.shape
            mask = mask_t.new_zeros((n, r_h, r_w))
            mask[:, :h, :w] = mask_t
            resized_mask = interpolate(
                input=mask.float().unsqueeze(0), size=(m_h, m_w), mode="bilinear", align_corners=False,
                )[0].gt(0)
            masks.append(resized_mask)
        return masks

    def dice_loss(self,input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))



    def fcos_losses(
        self,
        labels,
        reg_targets,
        logits_pred,
        reg_pred,
        ctrness_pred,
        controllers_pred,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,
        matched_idxes,
        im_idxes
    ):
        num_classes = logits_pred.size(1)
        labels = labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            logits_pred,
            class_target,
            alpha=focal_loss_alpha,
            gamma=focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        reg_pred = reg_pred[pos_inds]
        reg_targets = reg_targets[pos_inds]
        ctrness_pred = ctrness_pred[pos_inds]
        controllers_pred = controllers_pred[pos_inds]
        matched_idxes = matched_idxes[pos_inds]
        im_idxes = im_idxes[pos_inds]

        ctrness_targets = compute_ctrness_targets(reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        reg_loss = iou_loss(
            reg_pred,
            reg_targets,
            ctrness_targets
        ) / ctrness_norm

        ctrness_loss = F.binary_cross_entropy_with_logits(
             ctrness_pred,
             ctrness_targets,
             reduction="sum"
         ) / num_pos_avg

        # for CondInst
        N, C, h, w = self.masks.shape 
        grid_x = torch.arange(w).view(1,-1).float().repeat(h,1).cuda() / (w-1) * 2 - 1
        grid_y = torch.arange(h).view(-1,1).float().repeat(1,w).cuda() / (h-1) * 2 - 1
        x_map = grid_x.view(1, 1, h, w).repeat(N, 1, 1, 1)
        y_map = grid_y.view(1, 1, h, w).repeat(N, 1, 1, 1)
        masks_feat = torch.cat((self.masks, x_map, y_map), dim=1)
        r_h = int(h * self.strides[0])
        r_w = int(w * self.strides[0])
        targets_masks = [target_im.gt_masks.tensor for target_im in self.gt_instances]
        masks_t = self.prepare_masks(h, w, r_h, r_w, targets_masks)
        mask_loss = masks_feat[0].new_tensor(0.0)
        batch_ins = im_idxes.shape[0] 
        # for each image
        for i in range(N):
            inds = (im_idxes==i).nonzero().flatten()
            ins_num = inds.shape[0]
            if ins_num > 0:
                controllers = controllers_pred[inds]
                mask_feat = masks_feat[None, i]
                weights1 = controllers[:, :80].reshape(-1,8,10).reshape(-1,10).unsqueeze(-1).unsqueeze(-1)
                bias1 = controllers[:, 80:88].flatten()            
                weights2 = controllers[:, 88:152].reshape(-1,8,8).reshape(-1,8).unsqueeze(-1).unsqueeze(-1)
                bias2 = controllers[:, 152:160].flatten()
                weights3 = controllers[:, 160:168].unsqueeze(-1).unsqueeze(-1)
                bias3 = controllers[:,168:169].flatten()
                conv1 = F.conv2d(mask_feat,weights1,bias1).relu()
                conv2 = F.conv2d(conv1, weights2, bias2, groups = ins_num).relu()
                masks_per_image = F.conv2d(conv2, weights3, bias3, groups = ins_num)[0].sigmoid()
            
                for j in range(ins_num):
                    ind = inds[j]
                    mask_gt = masks_t[i][matched_idxes[ind]].float()
                    mask_pred = masks_per_image[j]
                    mask_loss += self.dice_loss(mask_pred, mask_gt)
            
        if batch_ins > 0:
            mask_loss = mask_loss / batch_ins
              

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_ctr": ctrness_loss,
            "loss_mask": mask_loss
        }
        return losses, {}

