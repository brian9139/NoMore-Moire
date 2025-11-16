import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--stage', type=str, choices=['spec', 'peaks', 'mask', 'restore', 'eval'])

    args = parser.parse_args()

    stage = args.stage
    if stage == 'spec':
        raise NotImplementedError
    elif stage == 'peaks':
        raise NotImplementedError
    elif stage == 'mask':
        raise NotImplementedError
    elif stage == 'restore':
        raise NotImplementedError
    elif stage == 'eval':
        raise NotImplementedError


if __name__ == '__main__':
    main()