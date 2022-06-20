r""" Extracts intermediate features from given backbone network & layer ids """


from collections import Counter
import numpy as np
def extract_feat_vgg(img, backbone_layers, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feat_ids_1 = [0, 3, 6]
    feats = []
    layers = []
    feat = img
    feat = backbone_layers[0](feat)
    layers.append(feat)
    feat = backbone_layers[1](feat)
    layers.append(feat)
    feat = backbone_layers[2](feat)
    layers.append(feat)

    for layers_34 in [backbone_layers[3], backbone_layers[4]]:
        for lid, module in enumerate(layers_34):
            feat = module(feat)
            if lid in feat_ids_1:
                feats.append(feat.clone())
        layers.append(feat)
    feats.append(feat.clone())
    return feats, layers

# from collections import Counter
# import numpy as np
# def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
#     r""" Extract intermediate features from VGG """
#     feats = []
#     feat = img
#     for lid, module in enumerate(backbone.features):
#         feat = module(feat)
#         if lid in feat_ids:
#             feats.append(feat.clone())
#     return feats

# def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
#     r""" Extract intermediate features from ResNet"""

#     feats = []

#     # Layer 0
#     feat = backbone.conv1.forward(img)
#     feat = backbone.bn1.forward(feat)
#     feat = backbone.relu.forward(feat)
#     feat = backbone.maxpool.forward(feat)
#     layers = []
#     layers.append(feat)
#     layer_nums = np.cumsum(list(Counter(lids).values()))
#     layer_nums_iter = iter(layer_nums)
#     layer_id = next(layer_nums_iter)
#     # Layer 1-4
#     for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
#         res = feat
#         feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
#         feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

#         if bid == 0:
#             res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

#         feat += res

#         if hid + 1 in feat_ids:
#             feats.append(feat.clone())

#         feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

#         if hid + 1 == layer_id:
#             if layer_id != layer_nums[-1]:
#                 layer_id = next(layer_nums_iter)
#             layers.append(feat.clone())

#     return feats, layers

def extract_feat_res(img, backbone_layers, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []
    feats
    # Layer 0
    feat = backbone_layers[0](img) #.conv1.forward(img)
    
    layer_nums = np.cumsum(list(Counter(lids).values()))
    layer_nums_iter = iter(layer_nums)
    layer_id = next(layer_nums_iter)
    layers = [feat]
    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone_layers[lid][bid].conv1.forward(feat)
        feat = backbone_layers[lid][bid].bn1.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv2.forward(feat)
        feat = backbone_layers[lid][bid].bn2.forward(feat)
        feat = backbone_layers[lid][bid].relu.forward(feat)
        feat = backbone_layers[lid][bid].conv3.forward(feat)
        feat = backbone_layers[lid][bid].bn3.forward(feat)

        if bid == 0:
            res = backbone_layers[lid][bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone_layers[lid][bid].relu.forward(feat)
        
        if hid + 1 == layer_id :
            if layer_id != layer_nums[-1]:
                layer_id = next(layer_nums_iter)
            layers.append(feat.clone())

    return feats, layers
    
def extract_feat_res_sup(img, backbone, feat_ids, bottleneck_ids, lids,shot=1):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res#bchw

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats