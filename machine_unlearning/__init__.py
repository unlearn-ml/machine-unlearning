from machine_unlearning import cifar100
from machine_unlearning import metrics
from machine_unlearning import reproducibility
from machine_unlearning import resnet18
from machine_unlearning import rosu
from machine_unlearning import splits
from machine_unlearning import training

__version__ = '0.0.1'

ROSU = rosu.ROSU
ResNet18 = resnet18.ResNet18
Split = splits.Split
random_forgetting = splits.random_forgetting
PairedBatches = training.PairedBatches
unlearn = training.unlearn
train = training.train
evaluate = metrics.evaluate
make_deterministic = reproducibility.make_deterministic
