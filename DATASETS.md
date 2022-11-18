# Dataset setup
# HKMC-MoCap-Synthetic-Dataset
Contact rune2002@hyundai.com to download the dataset and unzip the file in the repo.  
Then, you will have a data folder:
   ```
   ${Repo}
   ├── data
      ├── config.json
      ├── target.npz
      ├── C01
         ├── videos
            ├── M01_01.mp4
            ├── M01_02.mp4
            ...
         ├── calib.json
      ├── C02
         ├── videos
            ├── M01_01.mp4
            ├── M01_02.mp4
            ...
         ├── calib.json
   ```

## Extract images from videos
```sh
python extract_images.py
```
Then, you will have a data folder:
   ```
   ${Repo}
   ├── data
      ├── config.json
      ├── target.npz
      ├── C01
         ├── videos
            ├── M01_01.mp4
            ├── M01_02.mp4
            ...
         ├── images
            ├── M01_01
            ├── M01_02
            ...
         ├── calib.json
      ├── C02
         ├── videos
            ├── M01_01.mp4
            ├── M01_02.mp4
            ...
         ├── images
            ├── M01_01
            ├── M01_02
            ...
         ├── calib.json
   ```