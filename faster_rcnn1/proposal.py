
from proposal_layer import gen_proposal
from proposal_target_layer import gen_target

class Proposal:
    def __init__(self, num_class, is_training, weights_regularizer, batch_size):
    #cls_scoure_pred = np.zeros([1,2,2,3,2])
    #bbox_pred = np.zeros([1, 2,2,3,4])
        self.num_class = num_class
        self.is_training = is_training
        self.weights_regularizer = weights_regularizer
        self.batch_size = batch_size

    def build(self, rpn_cls_pred, rpn_bbox_pred, gt_box, im_info, num_anchors, anchor_list):

        bois, scores = gen_proposal(rpn_cls_pred, rpn_bbox_pred, im_info, anchor_list, num_anchors)

        self.bois = bois

        labels, rois, roi_scores, bbox_targets, in_weights, out_weights = gen_target(bois, scores, gt_box, self.num_class, self.batch_size)

        self.rois = rois
        self.cls_label = labels

        self.bbox_target = bbox_targets
        self.bbox_target_in_weight = in_weights
        self.bbox_target_out_weight = out_weights


