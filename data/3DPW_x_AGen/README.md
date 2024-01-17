3DPW dataset x AGen model

This dataset is a subset of the 3DPW dataset selected for the training, validation and testing of the AGen model. 
Not all of the videos container in the 3DPW have been included, since some of them are out of scope for the problem considered (e.g. multiple subjects)
This dataset has already been preprocessed to be ready to be used by the AGen model.
The structure of the dataset is the following:

* data 
    * test
        * Sequence_name
            * confs
                * video_metainfo.yaml
            * image
                * 0000.png
                * 0001.png
                * ...
            * mask
                * 0000.png
                * 0001.png
                * ...
            * cameras_normalize.npz
            * cameras.npz
            * mean_shape.npy
            * normalize_trans.npy
            * poses.npy
    *train 
        * Sequence_name
            * confs
                    * video_metainfo.yaml
                * image
                    * 0000.png
                    * 0001.png
                    * ...
                * mask
                    * 0000.png
                    * 0001.png
                    * ...
                * cameras_normalize.npz
                * cameras.npz
                * mean_shape.npy
                * normalize_trans.npy
                * poses.npy
    *valid 
        * Sequence_name
            * confs
                    * video_metainfo.yaml
                * image
                    * 0000.png
                    * 0001.png
                    * ...
                * mask
                    * 0000.png
                    * 0001.png
                    * ...
                * cameras_normalize.npz
                * cameras.npz
                * mean_shape.npy
                * normalize_trans.npy
                * poses.npy