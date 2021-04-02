import argparse
from tool import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model generator tool')
    parser.add_argument('-i', '--input_root', help="npy files root")
    parser.add_argument('-n', '--name', help="name of generated files")
    parser.add_argument('-o', '--output_root', help="generated source files root")
    args = parser.parse_args()

    if args.input_root is None or args.name is None or args.output_root is None:
        parser.print_help()
        quit()

    convert = Convert(input_root=args.input_root, output_root=args.output_root, name=args.name)
    convert()
