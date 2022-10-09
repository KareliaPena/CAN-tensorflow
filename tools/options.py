import argparse
import os

class Options():
    def __init__(self):
        
        self.parser  = argparse.ArgumentParser(description="parser for Tenforflow CAN")
        
        self.parser.add_argument("--epochs", type=int, default=2,
                                help="number of training epochs, default is 2")
        self.parser.add_argument("--batch-size", type=int, default=5,
                                help="batch size for training, default is 32")
        self.parser.add_argument("--train-set", type=str, default="",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
        self.parser.add_argument("--val-set", type=str, default="",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
        self.parser.add_argument("--test-set", type=str, default="",
                                help="path to training dataset, the path should point to a folder "
                                "containing another folder with all the training images")
        self.parser.add_argument("--log-interval", type=int, default=1000,
                                help="number of images after which the training loss is logged, default is 1000")
        self.parser.add_argument("--resume_dir", type=str, default="./training_checkpoints",
                                help="resume if needed")
        self.parser.add_argument("--checkpoint-dir", type=str, default="./training_checkpoints",
                                help="resume if needed")                                
        self.parser.add_argument("--viz", type=int, default=1,
                                help="visualize output using matplotlib during training")
        self.parser.add_argument("--log-dir", type=str, default='./logs/',
                                help="visualize output using matplotlib during training")
        self.parser.add_argument("--output-dir", type=str, default='./results/',
                                help="visualize output using matplotlib during training")
    def parse(self):
        return self.parser.parse_args()


