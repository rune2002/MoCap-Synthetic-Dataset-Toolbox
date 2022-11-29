# MoCap-Synthetic-Dataset-Toolbox

## Prepare dataset
* Please refer to [`DATASETS.md`](./DATASETS.md) for the preparation of the dataset files. 

## Save a npz file
The size of pose is (Frames, Joints, XYZ) and the submission data is dictionary which has keys of video name and values of pose. If xyz values of a joint are invalid, set x=y=z=0, so that evaluation metrics can be calculated properly. Then you save the data to npz files and submit the file.
```python
from utils.data_utils import save_npz

d = dict()
print(np.shape(p1)) # (Frames, Joints, 3)
d['M01_01'] = p1

print(np.shape(p2)) # (Frames, Joints, 3)
d['M01_02'] = p2
# ============== Add additional pose data ==============
save_npz('submission.npz', d)
```

## Run evaluation code
Before you make a submission, you can check the npz file is created properly.
```sh
python evaluate.py --submission '/path/to/npz' --target '/path/to/npz'
```

## Visualize
```sh
python visualize.py --submission '/path/to/npz'
```

## Issues
* RGB image to IR image pipeline

## Comment:
* Please feel free to make a pull request if you can fix errors or improve the toolbox.