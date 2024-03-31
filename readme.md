Our code is based on the improvement of Luo Hao, more details can be referred to http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
The modules we designed can be found in folder reid//modeling//backbone
We named the backbone network sraresnet50, and You can also modify some configurations under reid//config//defaults.py 
## Train
You can run these commands in  `.sh ` files for training different datasets of differernt loss.  You can also directly run code `sh *.sh` to run our demo after your custom modification.

1. Market1501, cross entropy loss + triplet loss

```bash
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

2. DukeMTMC-reID, cross entropy loss + triplet loss + center loss


```bash
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

## Test
You can test your model's performance directly by running these commands in `.sh ` files after your custom modification. You can also change the configuration to determine which feature of BNNeck is used and whether the feature is normalized (equivalent to use Cosine distance or Euclidean distance) for testing.

Please replace the data path of the model and set the `PRETRAIN_CHOICE` as 'self' to avoid time consuming on loading ImageNet pretrained model.

1. Test with Euclidean distance using feature before BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
2. Test with Cosine distance using feature after BN without re-ranking,.

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
```
3. Test with Cosine distance using feature after BN with re-ranking

```bash
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.RE_RANKING "('yes')" TEST.WEIGHT "('your path to trained checkpoints')"
```