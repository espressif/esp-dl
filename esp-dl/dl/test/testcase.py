import os
import time
from pathlib import Path
import tensorflow as tf
import numpy
import sys

sys.path.append(os.getcwd() + "/../..")
from convert.utils import Export, quantize, CHIP_LIST, ACTIVATION_LIST


def get_sign_width(t: str):
    assert t in ("u8", "s8", "s16")
    return "u" if t[0] == "u" else "", eval(t[1:])


def get_range(s: str, w: int):
    if s == "u":
        return 0, 2**w - 1
    else:
        return -(2 ** (w - 1)), 2 ** (w - 1) - 1


class TestDL:
    def __init__(
        self, target_chip: str, step: int = 25, total: int = 100, quant_method: int = 0
    ):
        assert target_chip in CHIP_LIST

        self.target_chip = target_chip
        self.root = "./"
        self.step = step
        self.total = total
        self.case = []
        self.quant_method = quant_method

        with open(f"{self.root}/CMakeLists.txt", "r") as file:
            self.backup = file.read()
        return

    def delete_case(self):
        while len(self.case) > 0:
            case = self.case.pop()
            while not Path(f"{self.root}/{case}.cpp").exists():
                print(".", end="")
                time.sleep(0.5)
            os.remove(f"{self.root}/{case}.cpp")
        return

    def wait(self) -> bool:
        with open(f"{self.root}/CMakeLists.txt", "w") as file:
            all_cpp = ".cpp\n            ".join(self.case)
            file.writelines(
                (
                    f"set(srcs \n            {all_cpp}.cpp\n)\n",
                    "set(requires unity dl)\n\n",
                    "idf_component_register(SRCS ${srcs} REQUIRES ${requires})\n\n",
                )
            )

        input_str = ""
        while not input_str.isnumeric() and not input_str == "exit":
            input_str = input(
                f'>>> {len(self.case)} TestCases were generated\nInput a number to continue or "exit" to exit: '
            )

        self.delete_case()

        if input_str.isnumeric() and input_str != "0":
            self.step = eval(input_str)
        else:
            return False

        return True

    def testcase(
        self,
        name: str,
        feature_type: str,
        input_shape: tuple,
        filter_shape: tuple,
        stride: tuple,
        dilation: tuple,
        operation: str,
        activation_types: tuple,
    ):
        """

        :param name:
        :param feature_type:
        :param filter_shape:
        :param stride:          (1, stride_y, stride_x, 1)
        :param dilation:        (dilation_y, dilation_x)
        :param operation:
        :return:
        """
        test_unaligned = (True, False)[
            numpy.random.randint(0, 2)
        ]  # actually must choose one

        print("input_shape: %s, filter_shape: %s" % (input_shape, filter_shape))
        output_channel = filter_shape[3] if operation == "conv2d" else input_shape[-1]
        padding = ("VALID", "SAME")[numpy.random.randint(0, 2)]

        if operation in ("conv2d", "depthwise_conv2d"):
            with_bias = (True, False)[numpy.random.randint(0, 2)]
        else:
            with_bias = False

        feature_sign, feature_element_width = get_sign_width(feature_type)
        feature_low, feature_high = get_range(feature_sign, feature_element_width)
        bias_low, bias_high = get_range(feature_sign, feature_element_width // 2)

        activation_type = activation_types[
            numpy.random.randint(0, len(activation_types))
        ]

        if (
            operation not in ("conv2d", "depthwise_conv2d")
            or feature_element_width == 16
        ):
            self.quant_method = 0
        if self.quant_method == 1 and feature_element_width == 8:
            bias_low, bias_high = get_range(feature_sign, 16 // 2)

        # input
        input_exponent = -feature_element_width
        input_q = numpy.random.randint(feature_low, feature_high, input_shape).astype(
            float
        )
        input_f = input_q * 2**input_exponent

        # filter
        if self.quant_method == 0:  # per-layer
            filter_exponent = -feature_element_width
            filter_q = numpy.random.randint(
                feature_low, feature_high, filter_shape
            ).astype(float)
            filter_f = filter_q * 2**filter_exponent
        elif self.quant_method == 1:  # int8 per-channel
            filter_exponent = numpy.random.randint(-7, -4, output_channel).astype(float)
            filter_q = numpy.random.randint(
                feature_low, feature_high, filter_shape
            ).astype(float)
            filter_f = filter_q * 2**filter_exponent

        # output
        output_f = None
        if operation == "conv2d":
            output_f = tf.nn.conv2d(
                input_f, filter_f, stride, padding, dilations=dilation
            )
        elif operation == "depthwise_conv2d":
            output_f = tf.nn.depthwise_conv2d(
                input_f, filter_f, stride, padding, dilations=dilation
            )
        elif operation == "avg_pool2d":
            output_f = tf.nn.avg_pool2d(input_f, filter_shape, stride, padding)
        elif operation == "global_avg_pool2d":
            output_f = tf.nn.avg_pool2d(input_f, filter_shape, 1, "VALID")

        if output_f is not None:
            _, output_exponent, _ = quantize(operation, output_f, feature_element_width)
        else:
            output_exponent = input_exponent

        if operation in ("conv2d", "depthwise_conv2d"):
            mac_shift = input_exponent + filter_exponent - output_exponent
        else:
            mac_shift = input_exponent - output_exponent

        if operation == "conv2d":
            output_q = tf.nn.conv2d(
                input_q, filter_q, stride, padding, dilations=dilation
            )
        elif operation == "depthwise_conv2d":
            output_q = tf.nn.depthwise_conv2d(
                input_q, filter_q, stride, padding, dilations=dilation
            )
        elif operation == "max_pool2d":
            output_q = tf.nn.max_pool2d(input_q, filter_shape, stride, padding)
        elif operation == "avg_pool2d":
            output_q = tf.nn.avg_pool2d(input_q, filter_shape, stride, padding)
        elif operation == "global_max_pool2d":
            output_q = tf.nn.max_pool2d(input_q, filter_shape, 1, "VALID")
        elif operation == "global_avg_pool2d":
            output_q = tf.nn.avg_pool2d(input_q, filter_shape, 1, "VALID")

        if self.quant_method == 1 and feature_element_width == 8 and with_bias:
            output_q = output_q.numpy().astype(int)
            output_q = output_q >> 4
            # output_q = numpy.clip(output_q, -2 ** 15, 2 ** 15 - 1).astype(int)

            bias_exponent = input_exponent + filter_exponent + 4
            bias_q = numpy.random.randint(bias_low, bias_high, output_channel).astype(
                int
            )
            output_q += bias_q
            output_q = numpy.floor(output_q * (2 ** (mac_shift + 4)))
            output_q = numpy.clip(output_q, feature_low, feature_high).astype(int)
        else:
            output_q = output_q.numpy()
            output_q = output_q * (2**mac_shift)
            output_q = numpy.clip(output_q, feature_low, feature_high)

            # bias
            bias_q = None
            bias_exponent = output_exponent
            if with_bias and "conv" in operation:
                bias_q = numpy.random.randint(
                    bias_low, bias_high, output_channel
                ).astype(float)
                output_q += bias_q
            output_q = numpy.clip(output_q, feature_low, feature_high)

        # activation
        alpha_q = None
        alpha_e = None
        if "conv" in operation:
            if activation_type == "ReLU":
                output_q = numpy.where(output_q < 0, 0, output_q)
            elif activation_type == "LeakyReLU":
                alpha_q = numpy.random.randint(feature_low, feature_high, 1)
                alpha_e = -feature_element_width
                output_q = numpy.where(
                    output_q < 0, output_q * alpha_q[0] * (2**alpha_e), output_q
                )
                alpha_q = alpha_q.astype(int)
            elif activation_type == "PReLU":
                alpha_q = numpy.random.randint(
                    feature_low, feature_high, output_channel
                )
                alpha_e = -feature_element_width
                for i in range(output_channel):
                    output_q[..., i] = numpy.where(
                        output_q[..., i] < 0,
                        output_q[..., i] * alpha_q[i] * 2**alpha_e,
                        output_q[..., i],
                    )
                alpha_q = alpha_q.astype(int)
            else:  # 'Linear'
                pass

        output_q = numpy.clip(output_q, feature_low, feature_high)

        # export to c
        padding_dl = "PADDING_VALID" if padding == "VALID" else "PADDING_SAME_END"

        if test_unaligned:
            aligned = False
            print_addr = '    printf("input_element: %p\\n", &input_element);\n'
        else:
            aligned = True
            print_addr = ""

        with open(f"{self.root}/{name}.cpp", "w") as file:
            file.writelines(
                (
                    "#include <stdint.h>\n",
                    "\n",
                    '#include "dl_constant.hpp"\n',
                    '#include "dl_variable.hpp"\n',
                    '#include "dl_tool.hpp"\n',
                    '#include "dl_define.hpp"\n',
                    f'#include "dl_nn_{operation.lower()}.hpp"\n\n',
                    "#include <limits.h>\n",
                    '#include "unity.h"\n\n',
                    "using namespace dl;\n",
                    "using namespace nn;\n",
                    "using namespace tool;\n\n",
                )
            )
            tool = Export(target_chip=self.target_chip, source_file=file, indent="")

            tool.export_element(
                name="input",
                array=input_q.astype(int),
                array_type=feature_type,
                aligned=aligned,
            )
            tool.export_element(
                name="output",
                array=output_q.astype(int),
                array_type=feature_type,
                aligned=aligned,
            )

            if "conv2d" in operation:
                tool(
                    name="layer",
                    operation=operation,
                    feature_type=feature_type,
                    filter_element=filter_q.astype(int),
                    filter_exponent=filter_exponent,
                    stride=stride,
                    dilation=dilation,
                    bias_element=None if bias_q is None else bias_q.astype(int),
                    bias_exponent=bias_exponent,
                    activation_type=activation_type,
                    activation_element=None if alpha_q is None else alpha_q.astype(int),
                    activation_exponent=alpha_e,
                    quant_granularity=self.quant_method,
                )

            if self.quant_method == 1 and feature_element_width == 8:
                bias_str = "&layer_bias" if with_bias else f"(Bias<int16_t> *)NULL"
            else:
                bias_str = (
                    "&layer_bias"
                    if with_bias
                    else f"(Bias<int{feature_element_width}_t> *)NULL"
                )
            activation_str = (
                "&layer_activation"
                if activation_type
                else f"(Activation<int{feature_element_width}_t> *)NULL"
            )
            input_shape_str = (
                input_shape[1:].__str__().replace("(", "{").replace(")", "}")
            )
            filter_shape_str = (
                filter_shape.__str__().replace("(", "{").replace(")", "}")
            )
            if operation.count("conv2d") > 0:
                file.writelines(
                    (
                        f'\nTEST_CASE("{name}", "[{operation}]")\n',
                        "{\n",
                        f"    Tensor<{feature_sign}int{feature_element_width}_t> input;\n",
                        f"    input.set_element(({feature_sign}int{feature_element_width}_t *)input_element).set_exponent({input_exponent}).set_shape({input_shape_str}).set_auto_free(false);\n\n",
                        f"{print_addr}",
                        f"    Tensor<{feature_sign}int{feature_element_width}_t> output = {operation}({output_exponent}, input, layer_filter, {stride[1]}, {stride[2]}, {padding_dl}, {bias_str}, {activation_str});\n",
                        f"    TEST_ASSERT(output.check_element(({feature_sign}int{feature_element_width}_t *)output_element, 2, false));\n",
                        "}\n",
                    )
                )
            elif operation.count("global") > 0:
                file.writelines(
                    (
                        f'\nTEST_CASE("{name}", "[{operation}]")\n',
                        "{\n",
                        f"    Tensor<{feature_sign}int{feature_element_width}_t> input;\n",
                        f"    input.set_element(({feature_sign}int{feature_element_width}_t *)input_element).set_exponent({input_exponent}).set_shape({input_shape_str}).set_auto_free(false);\n\n",
                        f"{print_addr}",
                        (
                            f"    Tensor<{feature_sign}int{feature_element_width}_t> output = {operation}(input);\n"
                            if operation.count("max") > 0
                            else f"    Tensor<{feature_sign}int{feature_element_width}_t> output = {operation}({output_exponent}, input);\n"
                        ),
                        f"    TEST_ASSERT(output.check_element(({feature_sign}int{feature_element_width}_t *)output_element, 5, false));\n",
                        "}\n",
                    )
                )
            elif operation.count("pool2d") > 0:
                file.writelines(
                    (
                        f'\nTEST_CASE("{name}", "[{operation}]")\n',
                        "{\n",
                        f"    Tensor<{feature_sign}int{feature_element_width}_t> input;\n",
                        f"    input.set_element(({feature_sign}int{feature_element_width}_t *)input_element).set_exponent({input_exponent}).set_shape({input_shape_str}).set_auto_free(false);\n\n",
                        f"{print_addr}",
                        (
                            f"    Tensor<{feature_sign}int{feature_element_width}_t> output = {operation}(input, {filter_shape_str}, {stride[1]}, {stride[2]}, {padding_dl});\n"
                            if operation.count("max") > 0
                            else f"    Tensor<{feature_sign}int{feature_element_width}_t> output = {operation}({output_exponent}, input, {filter_shape_str}, {stride[1]}, {stride[2]}, {padding_dl});\n"
                        ),
                        f"    TEST_ASSERT(output.check_element(({feature_sign}int{feature_element_width}_t *)output_element, 8, false));\n",
                        "}\n",
                    )
                )

            else:
                raise ValueError(f"operation {operation} is not supported.")
        return

    def __call__(
        self,
        operation: str,
        feature_type: str,
        input_shape: tuple = None,
        filter_shape: tuple = None,
        stride: tuple = None,
        dilation: tuple = None,
        activation: tuple = None,
    ):
        assert operation in (
            "conv2d",
            "depthwise_conv2d",
            "max_pool2d",
            "avg_pool2d",
            "global_max_pool2d",
            "global_avg_pool2d",
        )
        assert feature_type in ("s16", "s8")

        is_continue = True
        i = 0
        for _ in range(self.total):
            i += 1

            if operation == "conv2d":
                if filter_shape is None:
                    _filter_shape = tuple(numpy.random.randint(1, 8, 2)) + tuple(
                        numpy.random.randint(1, 48, 2)
                    )
                elif len(filter_shape) == 2:
                    _filter_shape = filter_shape + tuple(numpy.random.randint(1, 48, 2))
                elif len(filter_shape) == 4:
                    _filter_shape = filter_shape
                else:
                    raise ValueError(f"filter_shape can not be {filter_shape}.")

                if dilation is None:
                    _dilation = tuple(numpy.random.randint(1, 5, 2))
                else:
                    _dilation = dilation

                if stride is None:
                    _stride = (1,) + tuple(numpy.random.randint(1, 5, 2)) + (1,)
                else:
                    _stride = stride

            elif operation == "depthwise_conv2d":
                if filter_shape is None:
                    _filter_shape = tuple(numpy.random.randint(1, 8, 2)) + (
                        numpy.random.randint(1, 256),
                        1,
                    )
                elif len(filter_shape) == 2:
                    _filter_shape = (
                        filter_shape + tuple(numpy.random.randint(1, 256, 1)) + (1,)
                    )
                elif len(filter_shape) == 4:
                    _filter_shape = filter_shape
                else:
                    raise ValueError(f"filter_shape can not be {filter_shape}.")

                if dilation is None:
                    _dilation = tuple(numpy.random.randint(1, 5, 2))
                else:
                    _dilation = dilation

                if _dilation[0] == _dilation[1] == 1:
                    if stride is None:
                        s = numpy.random.randint(1, 5)
                        _stride = (1, s, s, 1)
                    else:
                        _stride = stride
                else:
                    if stride is None:
                        _stride = (1, 1, 1, 1)
                    elif stride[1] == stride[2] == 1:
                        _stride = stride
                    else:
                        raise ValueError(
                            "When dilation_x != 1 or dilation_y !=1, TF only support stride = (1, 1, 1, 1)"
                        )

            elif operation in ("max_pool2d", "avg_pool2d"):
                if filter_shape is None:
                    _filter_shape = tuple(numpy.random.randint(1, 8, 2))
                elif len(filter_shape) == 2:
                    _filter_shape = filter_shape
                else:
                    raise ValueError(f"filter_shape can not be {filter_shape}.")

                _dilation = (1, 1)
                if stride is None:
                    s = numpy.random.randint(1, 5)
                    _stride = (1, s, s, 1)
                else:
                    _stride = stride

            if operation in ("global_max_pool2d", "global_avg_pool2d"):
                _stride = 1
                _dilation = (1, 1)
                if input_shape is None:
                    if filter_shape is None:
                        _filter_shape = tuple(numpy.random.randint(1, 8, 2))
                    elif len(filter_shape) == 2:
                        _filter_shape = filter_shape
                    else:
                        raise ValueError(f"filter_shape can not be {filter_shape}.")

                    input_height = _filter_shape[0]
                    input_width = _filter_shape[1]
                    input_channel = numpy.random.randint(1, 256)
                    _input_shape = (1, input_height, input_width, input_channel)
                else:
                    if filter_shape is None:
                        _filter_shape = tuple(numpy.random.randint(1, 8, 2))
                        _filter_shape[0] = input_shape[0]
                        _filter_shape[1] = input_shape[1]
                    elif len(filter_shape) == 2:
                        assert _filter_shape[0] == input_shape[0]
                        assert _filter_shape[1] == input_shape[1]
                    else:
                        raise ValueError(f"filter_shape can not be {filter_shape}.")

                    input_channel = numpy.random.randint(1, 256)
                    _input_shape = (1, input_height, input_width, input_channel)
            elif input_shape is None:
                input_shape_min = max(
                    (_filter_shape[0] - 1) * _dilation[0] + 1,
                    (_filter_shape[1] - 1) * _dilation[1] + 1,
                )
                input_shape_max = (
                    max(
                        ((_filter_shape[0] - 1) * _dilation[0] + 1) * _stride[1],
                        ((_filter_shape[1] - 1) * _dilation[1] + 1) * _stride[2],
                    )
                    * 2
                )
                input_height = numpy.random.randint(input_shape_min, input_shape_max)
                input_width = numpy.random.randint(input_shape_min, input_shape_max)
                if operation in ("conv2d", "depthwise_conv2d"):
                    _input_shape = (1, input_height, input_width, _filter_shape[2])
                else:
                    input_channel = numpy.random.randint(1, 256)
                    _input_shape = (1, input_height, input_width, input_channel)
            else:
                if operation in ("conv2d", "depthwise_conv2d"):
                    _input_shape = (1,) + input_shape + (_filter_shape[2],)
                else:
                    input_channel = numpy.random.randint(1, 256)
                    _input_shape = (1,) + input_shape + (input_channel,)

            self.testcase(
                name=f"test_{i}",
                feature_type=feature_type,
                input_shape=_input_shape,
                filter_shape=_filter_shape,
                stride=_stride,
                dilation=_dilation,
                operation=operation,
                activation_types=ACTIVATION_LIST if activation is None else activation,
            )
            self.case.append(f"test_{i}")

            # CMakeLists.txt
            if len(self.case) == self.step:
                is_continue = self.wait()

            if not is_continue:
                break

        if is_continue and len(self.case):
            self.wait()

        with open(f"{self.root}/CMakeLists.txt", "w") as file:
            file.write(self.backup)
        return

    def __del__(self):
        self.delete_case()
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Assistant")
    parser.add_argument("--target_chip", help=f"{CHIP_LIST}")
    parser.add_argument(
        "--operation",
        help="conv2d, depthwise_conv2d, max_pool2d, avg_pool2d, global_max_pool2d, global_avg_pool2d",
    )
    parser.add_argument("--feature_type", help="s16, s8", default="s16")
    parser.add_argument("--input_shape", help='"(H, W)"', default="None")
    parser.add_argument("--filter_shape", help='"(H, W), (H, W, C, N)"', default="None")
    parser.add_argument("--stride", help='"(1, y, x, 1)"', default="None")
    parser.add_argument("--dilation", help='"(y, x)"', default="None")
    parser.add_argument(
        "--activation",
        help='"(None, \\"ReLU\\", \\"LeakyReLU\\", \\"PReLU\\")"',
        default="None",
    )
    parser.add_argument(
        "--step", help="Wait for every this number of testcases", default="20"
    )
    parser.add_argument("--total", help="The total of testcases", default="100")
    parser.add_argument(
        "--quant", help="The quantization method of filter", default="0"
    )
    args = parser.parse_args()

    if all((args.target_chip, args.operation)):
        test_dl = TestDL(
            target_chip=args.target_chip,
            step=eval(args.step),
            total=eval(args.total),
            quant_method=eval(args.quant),
        )
        test_dl(
            operation=args.operation,
            feature_type=args.feature_type,
            input_shape=eval(args.input_shape),
            filter_shape=eval(args.filter_shape),
            stride=eval(args.stride),
            dilation=eval(args.dilation),
            activation=eval(args.activation),
        )
    else:
        parser.print_help()
        quit()
