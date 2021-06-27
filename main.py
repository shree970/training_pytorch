from models.resnet import resnet18
from utils.helper import model_summary

model_summary(resnet18(), (3, 32, 32))
