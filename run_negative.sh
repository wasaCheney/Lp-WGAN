dataset='lsun'
dataroot=${HOME}'/data/'${dataset}
norm='vanilla l1 l2 linfty'
 
python negative.py --clamp_upper 100 --dataset ${dataset} --dataroot ${dataroot} --norm 'linfty' --niter 65 --cuda

