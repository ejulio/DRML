#ifndef CUSTOM_BOX_LAYER_HPP_
#define CUSTOM_BOX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BoxLayer : public Layer<Dtype> {
    public:
        explicit BoxLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Box"; }
        
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        
        virtual inline int MinTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, 
            const vector<Blob<Dtype>*>& bottom);
        
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, 
            const vector<Blob<Dtype>*>& bottom);

    int count_;
    vector <int> xcoord_;
    vector <int> ycoord_;
    int width_;
    int height_;
    int num_;
    int channels_;
};

}

#endif