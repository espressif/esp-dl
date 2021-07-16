import argparse
from utils import Convert

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model generator tool')
    parser.add_argument('-t', '--target_chip', help='esp32, esp32s2, esp32s3, esp32c3')
    parser.add_argument('-i', '--input_root', help="npy files root")
    parser.add_argument('-n', '--name', help="name of generated files")
    parser.add_argument('-o', '--output_root', help="generated source files root")
    args = parser.parse_args()

    if args.input_root is None or args.name is None or args.output_root is None:
        parser.print_help()
        quit()

    print(f'Generating {args.output_root}/{args.name} on {args.target_chip}...', end='')
    convert = Convert(target_chip=args.target_chip, input_root=args.input_root, output_root=args.output_root,
                      name=args.name)
    convert()
    print(' Finish')
