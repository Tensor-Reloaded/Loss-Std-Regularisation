# Bellow you have the parameters used for running each experiment:
- Run 1: 
  - Baseline: python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --mean_pen=1.0 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG19')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-19/runs/run_1/baseline/metrics"
  - Std Loss Regularized: python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --mean_pen=1.0 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG19')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-19/runs/run_1/metrics" -std --std_pen=0.25 --std_pen_milestones 30 60 90 180 240 290 300 --std_pen_gamma=1.65

- Run 2: 
  - Baseline: python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --mean_pen=1.0 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG19')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-19/runs/run_2/baseline/metrics"
  - Std Loss Regularized: python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG19')" --epoch=300 --train_batch_size=128 --save_path="results/CIFAR-10/VGG-19/runs/run_2/metrics" -std --std_pen=0.15 --std_pen_milestones 30 60 90 180 250 300 --std_pen_gamma=2.0