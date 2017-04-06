#ifndef CUSTOM_MULTILABEL_IMAGE_DATA_LAYER_HPP_
#define CUSTOM_MULTILABEL_IMAGE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MultilabelImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
    public:
        explicit MultilabelImageDataLayer(const LayerParameter& param)
          : BasePrefetchingDataLayer<Dtype>(param) {}
        
        virtual ~MultilabelImageDataLayer();
        
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "MultilabelImageData"; }
        
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        
        //virtual inline int ExactNumTopBlobs() const { return 3; }
        virtual inline int MaxTopBlobs() const { return 2; } // Do not support weight!

    protected:
        shared_ptr<Caffe::RNG> prefetch_rng_;
        virtual void ShuffleImages();
        virtual void load_batch(Batch<Dtype>* batch);

    //vector<std::pair<std::string, int> > lines_;
    vector<std::pair<std::string, std::vector<float> > > lines_;
    int lines_id_;
};

}

#endif