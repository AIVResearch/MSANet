""" Visualize model predictions """
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

def to_cpu(tensor):
    return tensor.detach().clone().cpu()


class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255), 'green':(50,255,50)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = './vis_'
        # if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, pred_meta_mask_b, base_out, cls_id_b, batch_idx, iou_b=None, dir_name="./"):
        spt_img_b = to_cpu(spt_img_b)
        spt_mask_b = to_cpu(spt_mask_b)
        qry_img_b = to_cpu(qry_img_b)
        qry_mask_b = to_cpu(qry_mask_b)
        pred_mask_b = to_cpu(pred_mask_b)
        pred_meta_mask_b = to_cpu(pred_meta_mask_b)
        base_out = to_cpu(base_out)
        cls_id_b = to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, pred_meta_mask, base_out, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, pred_meta_mask_b, base_out, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, pred_meta_mask, base_out, cls_id, batch_idx, sample_idx, True, iou, dir_name)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, pred_meta_mask, base_out, cls_id, batch_idx, sample_idx, label, iou=None, dir_name="./"):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['green']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        spt_image_pils = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        s_mask_b = []
        for spt_mask in spt_masks:
            spt_mask[spt_mask==255] = 0
            s_mask_b.append(spt_mask*255)

        s_mask_b = [Image.fromarray(spt_mask) for spt_mask in s_mask_b]
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_meta_mask = cls.to_numpy(pred_meta_mask, 'mask')

        base_out = cls.to_numpy(base_out, 'mask')
        base_oot_with_output = base_out
        base_oot_with_output[pred_mask==1] = 255
        base_oot_with_output_concat = np.repeat(base_oot_with_output[:, :, np.newaxis], 3, axis=2)   
        base_oot_with_output_concat[base_oot_with_output==1] = (67, 3, 38)
        base_oot_with_output_concat[base_oot_with_output==2] = (66, 8, 109)
        base_oot_with_output_concat[base_oot_with_output==3] = (106, 81, 224)
        base_oot_with_output_concat[base_oot_with_output==4] = (79, 120, 185)
        base_oot_with_output_concat[base_oot_with_output==5] = (9, 100, 245)
        base_oot_with_output_concat[base_oot_with_output==6] = (9, 245, 241)
        base_oot_with_output_concat[base_oot_with_output==7] = (13, 78, 53)
        base_oot_with_output_concat[base_oot_with_output==8] = (15, 203, 87)
        base_oot_with_output_concat[base_oot_with_output==9] = (121, 175, 96)
        base_oot_with_output_concat[base_oot_with_output==10] = (133, 94, 11)
        base_oot_with_output_concat[base_oot_with_output==11] = (133, 54, 11)
        base_oot_with_output_concat[base_oot_with_output==12] = (133, 28, 11)
        base_oot_with_output_concat[base_oot_with_output==13] = (226, 132, 172)
        base_oot_with_output_concat[base_oot_with_output==14] = (211, 226, 132)
        base_oot_with_output_concat[base_oot_with_output==15] = (140, 164, 179)
        base_oot_with_output_concat[base_oot_with_output==255] = (255,255,255)
        base_out_bg = (base_out == 0)
        base_out[base_out_bg] = 255
        base_out_fg = (base_out < 16)
        base_out[base_out_fg] = 1
        base_out_bg = (base_out == 255)
        base_out[base_out_bg] = 0


        query_img = Image.fromarray(qry_img.astype(np.uint8))
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        pred_mask_b = Image.fromarray(pred_mask.astype(np.uint8)*255)
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        qry_mask_b = qry_mask.astype(np.uint8)
        qry_mask_b[qry_mask_b==255] = 0
        qry_mask_b = Image.fromarray(qry_mask_b*255)
        qry_masked_meta_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_meta_mask.astype(np.uint8), pred_color))
        qry_mask_meta_b = Image.fromarray(pred_meta_mask.astype(np.uint8)*255)
        base_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), base_out.astype(np.uint8), pred_color))
        base_mask_b = Image.fromarray(base_out.astype(np.uint8)*255)
        generalized_mask = Image.fromarray(base_oot_with_output_concat.astype(np.uint8))
        
        # merged_pil = cls.merge_image_pair(spt_masked_pils + [qry_masked_meta_pil, base_masked_pil, pred_masked_pil, qry_masked_pil] + spt_image_pils +s_mask_b + [query_img, qry_mask_meta_b, base_mask_b, pred_mask_b, qry_mask_b])
        merged_pil = cls.merge_image_pair(spt_masked_pils + [qry_masked_meta_pil, base_masked_pil, pred_masked_pil, qry_masked_pil, generalized_mask])

        iou = iou.item() if iou else 0.0
        if not os.path.exists(cls.vis_path + dir_name): os.makedirs(cls.vis_path + dir_name)
        merged_pil.save(cls.vis_path + dir_name + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        # mask = F
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
