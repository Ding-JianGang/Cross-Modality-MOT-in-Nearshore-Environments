_base_ = [
    '../../_base_/models/deformable-detr.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

batch_size = 8

model = dict(
    type='SP-Tracker',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,),
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=4)
    ),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='SP-Tracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
# optimizer
# default 8 gpu
lr = 0.001 / 8 * batch_size

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
# dataset
data_root = '../data/flir'
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        data_root=data_root,
        load_as_video=False,
        visibility_thr=-1,
        ann_file='annotations/thermal_train_annotations.json',
        data_prefix=dict(img_path='train'),
        # 3 1 2 17
        metainfo=dict(CLASSES=('car', 'person', 'bicycle', 'dog')),
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=0,
            filter_key_img=True,
            method='uniform'),))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler'),
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/thermal_val_annotations.json',
        data_prefix=dict(img_path='train'),
        metainfo=dict(CLASSES=('car', 'person', 'bicycle', 'dog')),
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=1,
            frame_range=0,
            filter_key_img=True,
            method='uniform'),
        load_as_video=False,
        test_mode=True,))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
test_evaluator = val_evaluator

# some hyper parameters
# training settings
total_epochs = 300
num_last_epochs = 10
resume_from = None
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(interval=1))
# evaluator
val_evaluator = dict(postprocess_tracklet_cfg=[
    dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
])
test_evaluator = val_evaluator
