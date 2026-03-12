from esp_ppq.parser.espdl.export_patterns import (
    AddLUTPattern,
    Operation,
    BaseGraph,
    QuantableOperation,
    EspQuantType,
    ExporterPatternInfo,
)


def apply_addlut_patch():
    # Save original method just in case
    _original_export = AddLUTPattern.export

    def patched_export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        quant_type = op.attributes.get("quant_type", None)
        if (
            quant_type == None
            or quant_type == EspQuantType.F32
            or not isinstance(op, QuantableOperation)
        ):
            return op

        info = ExporterPatternInfo()

        if self.check_op(op):
            lut = None
            if quant_type == EspQuantType.S8:
                lut = self.calculate_lut(op, info, 127, -128, 1)
            elif quant_type == EspQuantType.S16:
                # [CHANGE]: Prioritize op-specific step logic assigned during Fusion over static defaults
                current_step = op.attributes.get("int16_lut_step", self.int16_step)

                # Sanity check: Ensure step is valid
                if current_step is None or current_step <= 0:
                    current_step = 256

                if current_step > 0:
                    lut = self.calculate_lut(
                        op, info, 2**15 - 1, -(2**15), int(current_step)
                    )

            if lut != None:
                lut_name = self.get_lut_name(op, info)
                op.attributes["lut"] = lut_name
                info.add_lut(lut_name, lut, info.get_var_exponents(op.outputs[0].name))

        return op

    print("Applying fix to AddLUTPattern.export for correct LUT step propagation...")
    AddLUTPattern.export = patched_export
