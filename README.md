## Official Code for Joint Deformable Attention and Spatial Position Tracker for Cross-Modality Multiple Object Tracking in Nearshore Environments.
Our dual-modal data is loaded through Dual_stream_data_loading.py, please copy this file to mmtracking-1.0.0rc1\configs\_base_\datasets.The Cross-Modality Feature Fusion module is integrated in Resnet_Bi-attention.py. Copy it into the mmcv repository to train.
![7](https://user-images.githubusercontent.com/88175740/234747830-ceecfb79-9a57-4b5a-8339-41b87c4c7c8f.jpg)


SP-Tracker is a simple and strong online multi-object tracker in Nearshore Environments.We built SP-Tracker on the source code of OC-SORT, as shown in SP-Tracker/SP_with_OC.py. We evaluate the entire pipeline through the TrackEval library, which is also uploaded in the repository.
![8](https://user-images.githubusercontent.com/88175740/234749270-b635c012-e6e0-4923-b46a-bc983521b90a.jpg)

