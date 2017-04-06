This repository contains some changes that I've made to DRML in order to make it compatible with newer versions of Caffe.

These are the changes:

Separate layers in different files .hpp, .cpp, e .cu

Add *BoxParameter box_param* in *message LayerParameter { ... }* in caffe.proto

Add BoxParameter definition
    message BoxParameter {
      required uint32 width = 1;
      required uint32 height = 2;

      repeated uint32 xcoord = 3;
      repeated uint32 ycoord = 4;
    }

Add the parameter *optional uint32 multilabel_num = 13;* in *message ImageDataParameter {* in caffe.proto

Changed *this->JoinPrefetchThread();* to *this->StopInternalThread();* in multilabel_image_data_layer.cpp

Changed *this->prefetch_data_* and *this->prefetch_label_* to:
_Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");_
*batch->data_ and batch->label_*

Changed *InternalThreadEntry* to _load_batch(Batch<Dtype>* batch)_ in multilabel_image_data_layer.cpp
Changed *InternalThreadEntry* to _load_batch(Batch<Dtype>* batch)_ in multilabel_image_data_layer.hpp

Changed caffe.proto adding:
message SpliceParameter {
  repeated uint32 xcoord = 1;
  repeated uint32 ycoord = 2;
}
and changing *LayerParameter* to add
optional SpliceParameter splice_param = 148;

Summary:

Created files:
- include/caffe/layers/multilabel_image_data_layer.hpp
- src/caffe/layers/multilabel_image_data_layer.cpp
- include/caffe/layers/multi_sigmoid_cross_entropy_loss_layer.hpp
- src/caffe/layers/multi_sigmoid_cross_entropy_loss_layer.cpp
- src/caffe/layers/multi_sigmoid_cross_entropy_loss_layer.cu
- include/caffe/layers/box_layer.hpp
- src/caffe/layers/box_layer.cpp
- src/caffe/layers/box_layer.cu
- include/caffe/layers/splice_layer.hpp
- src/caffe/layers/splice_layer.cpp
- src/caffe/layers/splice_layer.cu

Changed files:
- src/caffe/proto/caffe.proto
- include/caffe/layers/multilabel_image_data_layer.hpp
- src/caffe/layers/multilabel_image_data_layer.cpp

This is the original READE.md

Intro
-----

This repository provides the codes for the CVPR16 paper, “[Deep Region and Multi-Label Learning for Facial Action Unit Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf)".
This code aims for training a convolutional network that contains a *region layer* for specializing the learned kernels on different facial regions, and meanwhile utilizes a multi-label cross-entropy to jointly learn 12 AUs.
This implementation is based on [Caffe Toolbox](https://github.com/BVLC/caffe).


File structure
--------------

Based on the caffe toolbox, we organize the source files as follows:

- `include/caffe/`: Header files that contains the declaration of our implemented layers

- `prototxt/`: Network architecture we used to compuare and report in our paper

- `src/caffe/layers/`: Source files of our implemented layers

    - `box_layer.*`: Slice a 160x160 response map into an 8x8 uniform grid.

    - `image_data_layer_multilabel.cpp`: Load multiple labels for one image.

    - `multi_sigmoid_cross_entropy_loss_layer.*`: Multi-label loss.

    - `splice.*`: Concatenate 20 8x8 uniform grids to a 160x160 feature map.



More info
---------

- **Contact**:  Please send comments to Kaili Zhao (kailizhao@bupt.edu.cn)
- **Citation**: If you use this code in your paper, please cite the following:
```
@inproceedings{zhao2016deep,
  title={Deep Region and Multi-Label Learning for Facial Action Unit Detection},
  author={Zhao, Kaili and Chu, Wen-Sheng and Zhang, Honggang},
  booktitle={CVPR},
  year={2016}
}
```
