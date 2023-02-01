"""Plusieurs loss sont implémentés dans la librairie :
- keypointLoss
- totalVariation
- perceptualVGG
- perceptualClassifier"""

from .common import totalVariation, keypointLoss
from .perceptualVGG import perceptualVGG
from .perceptualClassifier import perceptualClassifier,getTrainedModel
from .classifierUtils import generate_label, generate_label_plain
from .gan import ganLoss