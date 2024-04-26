# RiskNET_original

Original dev. code by Tim Schoonbeek. Reproducability test! <br>

Change the dataset's path in Line 114 in dataset.py <br>
<br>
To train and val on the entire dataset : python main.py --run_name Tim --input mask --n_epochs 20 --lr 0.000001 --batch_size 32 --train --backbone ResNext24 --h_flip --dataset constant_radius <br>
<br>
To train and val on a small dataset : python main.py --input mask --dataset extended --tiny_dataset --train --n_epochs 10 <br>
(Change the "--dataset" as required) <br>
<br>
<br>
To test on the fully trained Tim's set : python main.py --run_name Tim --input mask  --backbone ResNext24 --batch_size 32 --dataset real_world_08 --mask_method "case4"
