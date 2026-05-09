import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser

import time
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time=time.time()
    elapsed_time = end_time - start_time
    
    # 转换为分钟和小时
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600
    
    # 打印结果
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Execution time: {elapsed_minutes:.2f} minutes")
    print(f"Execution time: {elapsed_hours:.2f} hours")
