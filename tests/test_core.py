# SPDX-License-Identifier: MulanPSL-2.0
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musaopst.compare import compare_outputs
from musaopst.config import ConfigError, expand_config, load_config
from musaopst.data import clone_inputs, make_inputs, quantize_bfloat16, reference_inputs, tensor_metadata


def example_case():
    return {"name": "add", "backend": "cpu", "entry": "musaopst.examples.cpu_ops:add",
            "inputs": [{"shape": [2, 3], "dtype": "float32"}],
            "outputs": [{"shape": [2, 3], "dtype": "float32"}],
            "reference": "musaopst.examples.references:add"}


def expand(case):
    return expand_config({"schema_version": 1, "cases": [case]})


class ConfigTests(unittest.TestCase):
    def test_defaults_and_no_mutation(self):
        original = example_case()
        before = copy.deepcopy(original)
        result = expand(original)[0]
        self.assertEqual(original, before)
        self.assertEqual(result["compare"]["max_mismatch_ratio"], 0)
        self.assertEqual(result["inputs"][0]["layout"], "contiguous")
        self.assertFalse(result["benchmark"]["enabled"])
        self.assertTrue(result["required"])

    def test_expansion_cartesian_and_independent(self):
        case = example_case()
        case["parameters"] = {"shape": [[2, 3], [0, 7]], "dtype": ["float16", "float32"]}
        case["inputs"][0].update(shape="${shape}", dtype="${dtype}")
        case["outputs"][0].update(shape="${shape}", dtype="${dtype}")
        expanded = expand(case)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(len(set(item["name"] for item in expanded)), 4)
        self.assertEqual(expanded[-1]["parameters_selected"], {"shape": [0, 7], "dtype": "float32"})
        expanded[0]["inputs"][0]["shape"][0] = 99
        self.assertEqual(expanded[1]["inputs"][0]["shape"], [2, 3])

    def test_parameter_combination_validation(self):
        case = example_case()
        case["parameters"] = {"n": [3, -1]}
        case["inputs"][0]["shape"] = ["${n}"]
        with self.assertRaises(ConfigError):
            expand(case)

    def test_unresolved_and_partial_placeholders(self):
        for value in ("${missing}", "float${precision}"):
            case = example_case()
            case["inputs"][0]["dtype"] = value
            with self.subTest(value=value), self.assertRaises(ConfigError):
                expand(case)

    def test_empty_parameters_rejected(self):
        case = example_case()
        case["parameters"] = {"n": []}
        with self.assertRaises(ConfigError):
            expand(case)

    def test_reference_cannot_be_tested_entry(self):
        case = example_case()
        case["reference"] = case["entry"]
        with self.assertRaisesRegex(ConfigError, "independent"):
            expand(case)

    def test_unknown_fields_rejected_at_all_levels(self):
        paths = [(), ("inputs", 0), ("outputs", 0), ("compare",), ("benchmark",), ("repeat",)]
        for path in paths:
            case = expand(example_case())[0]
            target = case
            for key in path:
                target = target[key]
            target["typo"] = 3
            with self.subTest(path=path), self.assertRaisesRegex(ConfigError, "unknown fields"):
                expand(case)

    def test_nonfinite_negative_tolerances_and_ratio_rejected(self):
        for key, value in (("atol", -1), ("atol", float("nan")), ("rtol", float("inf")),
                           ("max_mismatch_ratio", 1.1), ("max_failures", -1)):
            case = example_case()
            case["compare"] = {key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                expand(case)

    def test_boolean_is_not_integer(self):
        for field in ("seed", "timeout_s"):
            case = example_case()
            case[field] = True
            with self.subTest(field=field), self.assertRaises(ConfigError):
                expand(case)
        case = example_case()
        case["inputs"][0]["shape"] = [True]
        with self.assertRaises(ConfigError):
            expand(case)

    def test_dtype_override(self):
        case = example_case()
        case["compare"] = {"atol": 0.1, "rtol": 0.2}
        case["compare_by_dtype"] = {"float32": {"atol": 0.03}, "float16": {"atol": 1}}
        policy = expand(case)[0]["compare"]
        self.assertEqual(policy["atol"], 0.03)
        self.assertEqual(policy["rtol"], 0.2)

    def test_invalid_nonselected_dtype_policy_rejected(self):
        case = example_case()
        case["compare_by_dtype"] = {"float16": {"atol": -1}}
        with self.assertRaises(ConfigError):
            expand(case)

    def test_distribution_validation(self):
        distributions = [{"kind": "uniform", "low": 1, "high": 1}, {"kind": "normal", "std": -1},
                         {"kind": "normal", "low": 0}, {"kind": "fixed", "values": [[1], [2, 3]]},
                         {"kind": "fixed", "values": [1, 2]}]
        for distribution in distributions:
            case = example_case()
            case["inputs"][0]["distribution"] = distribution
            with self.subTest(distribution=distribution), self.assertRaises(ConfigError):
                expand(case)

    def test_scalar_strided_rejected(self):
        case = example_case()
        case["inputs"][0].update(shape=[], layout="strided")
        with self.assertRaises(ConfigError):
            expand(case)

    def test_empty_cases_duplicate_names_and_version_rejected(self):
        configs = [{"schema_version": True, "cases": [example_case()]},
                   {"schema_version": 3, "cases": [example_case()]},
                   {"schema_version": 1, "cases": []},
                   {"schema_version": 1, "cases": [example_case(), example_case()]}]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ConfigError):
                expand_config(config)

    def test_load_rejects_nonstandard_json_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text in ('{"schema_version":1,"schema_version":1,"cases":[]}',
                         '{"schema_version":NaN,"cases":[]}', 'not json'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ConfigError):
                    load_config(path)


class DataTests(unittest.TestCase):
    def test_seed_determinism_independent_allocations(self):
        case = expand(example_case())[0]
        left, right = make_inputs(case)[0], make_inputs(case)[0]
        np.testing.assert_array_equal(left, right)
        self.assertFalse(np.shares_memory(left, right))
        case["seed"] += 1
        self.assertFalse(np.array_equal(left, make_inputs(case)[0]))

    def test_strided_generation_clone_and_reference(self):
        case = example_case()
        case["inputs"][0]["layout"] = "strided"
        original = make_inputs(expand(case)[0])[0]
        duplicate = clone_inputs([original])[0]
        reference = reference_inputs([original])[0]
        self.assertEqual(tensor_metadata(original)["strides"], [6, 2])
        self.assertFalse(original.flags.c_contiguous)
        self.assertEqual(duplicate.strides, original.strides)
        self.assertEqual(tensor_metadata(reference)["strides"], [6, 2])
        self.assertEqual(reference.dtype, np.dtype("float64"))
        original.fill(99)
        np.testing.assert_array_equal(duplicate, reference)
        self.assertFalse(np.shares_memory(duplicate, original))
        self.assertFalse(np.shares_memory(reference, original))

    def test_strides_for_zero_and_singleton_shapes(self):
        for shape in ([0], [1], [2, 0], [0, 3]):
            case = example_case()
            case["inputs"][0].update(shape=shape, layout="strided")
            original = make_inputs(expand(case)[0])[0]
            duplicate = clone_inputs([original])[0]
            self.assertEqual(original.shape, tuple(shape))
            self.assertEqual(original.strides, duplicate.strides)
            self.assertEqual(tensor_metadata(original)["strides"][-1], 2)

    def test_negative_stride_snapshot(self):
        original = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1, ::-2]
        duplicate = clone_inputs([original])[0]
        np.testing.assert_array_equal(duplicate, original)
        self.assertEqual(duplicate.strides, original.strides)
        self.assertFalse(np.shares_memory(duplicate, original))

    def test_fixed_values_and_scalar_fill(self):
        for values in ([[1, 2, 3], [4, 5, 6]], 1.25):
            case = example_case()
            case["inputs"][0]["distribution"] = {"kind": "fixed", "values": values}
            array = make_inputs(expand(case)[0])[0]
            np.testing.assert_array_equal(array, np.broadcast_to(values, (2, 3)))

    def test_fixed_empty_shape(self):
        case = example_case()
        case["inputs"][0].update(shape=[3, 0, 2], distribution={"kind": "fixed", "values": []})
        self.assertEqual(make_inputs(expand(case)[0])[0].shape, (3, 0, 2))

    def test_fp16_reference_uses_quantized_values(self):
        case = example_case()
        case["inputs"][0].update(dtype="float16", distribution={"kind": "fixed", "values": 1.0003})
        actual = make_inputs(expand(case)[0])[0]
        reference = reference_inputs([actual])[0]
        self.assertEqual(float(reference[0, 0]), float(np.float16(1.0003)))
        self.assertNotEqual(float(reference[0, 0]), 1.0003)

    def test_bfloat16_rounding_ties_even_and_specials(self):
        bits = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000,
                         0x7F800000, 0xFF800000, 0x7F800001], dtype=np.uint32)
        actual = quantize_bfloat16(bits.view(np.float32))
        self.assertEqual(actual.view(np.uint32)[:4].tolist(), [0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000])
        self.assertTrue(np.isposinf(actual[4]))
        self.assertTrue(np.isneginf(actual[5]))
        self.assertTrue(np.isnan(actual[6]))

    def test_integer_reference_not_promoted_to_float(self):
        original = np.array([2**63-1], dtype=np.int64)
        reference = reference_inputs([original])[0]
        self.assertEqual(reference.dtype, np.dtype("int64"))
        self.assertEqual(int(reference[0]), 2**63-1)


class CompareTests(unittest.TestCase):
    def compare(self, actual, expected, policy=None, shape=None, dtype="float32"):
        actual = np.asarray(actual, dtype=dtype)
        expected = np.asarray(expected, dtype=np.float64)
        return compare_outputs([actual], [expected], [{"shape": shape or list(actual.shape), "dtype": dtype}], policy or {})

    def test_correct_values_pass(self):
        result = self.compare([1, 2, 3], [1, 2, 3])
        self.assertTrue(result["passed"])
        self.assertEqual(result["outputs"][0]["mismatch_count"], 0)

    def test_small_wrong_values_do_not_pass_boolean_all(self):
        result = self.compare([1, 2.1, 3], [1, 2, 3])
        self.assertFalse(result["passed"])
        self.assertEqual(result["outputs"][0]["mismatch_count"], 1)
        self.assertEqual(result["outputs"][0]["failures"][0]["index"], [1])

    def test_shape_contract_prevents_broadcast(self):
        result = self.compare([[1, 2]], [1, 2], shape=[2])
        self.assertFalse(result["passed"])
        self.assertTrue(result["contract_errors"])
        self.assertEqual(result["outputs"], [])

    def test_output_count_contract(self):
        result = compare_outputs([], [np.array([1])], [{"shape": [1], "dtype": "float32"}], {})
        self.assertFalse(result["passed"])
        self.assertIn("count", result["contract_errors"][0])

    def test_reference_count_and_shape_contract(self):
        contract = [{"shape": [2], "dtype": "float32"}]
        for expected in ([], [np.ones((1, 2))]):
            result = compare_outputs([np.ones(2, np.float32)], expected, contract, {})
            self.assertFalse(result["passed"])
            self.assertEqual(result["outputs"], [])

    def test_wrong_actual_dtype_fails_but_reference_precision_is_free(self):
        result = compare_outputs([np.ones(2, np.float16)], [np.ones(2, np.float64)],
                                 [{"shape": [2], "dtype": "float32"}], {})
        self.assertFalse(result["passed"])
        self.assertIn("dtype", result["contract_errors"][0])
        self.assertTrue(self.compare([1, 2], [1, 2])["passed"])

    def test_bfloat16_logical_dtype_metadata(self):
        array = quantize_bfloat16([1, 2])
        result = compare_outputs([array], [array.astype(np.float64)],
                                 [{"shape": [2], "dtype": "bfloat16"}], {}, ["bfloat16"])
        self.assertTrue(result["passed"])

    def test_declared_actual_dtype_cannot_hide_wrong_storage(self):
        result = compare_outputs([np.ones(2, np.float64)], [np.ones(2)],
                                 [{"shape": [2], "dtype": "float32"}], {}, ["float32"])
        self.assertFalse(result["passed"])
        self.assertIn("storage dtype", result["contract_errors"][0])

    def test_nan_policy(self):
        self.assertFalse(self.compare([np.nan], [np.nan])["passed"])
        self.assertTrue(self.compare([np.nan], [np.nan], {"nan": "equal"})["passed"])
        self.assertFalse(self.compare([np.nan], [1], {"nan": "equal"})["passed"])

    def test_inf_policy(self):
        self.assertFalse(self.compare([np.inf], [np.inf])["passed"])
        self.assertTrue(self.compare([np.inf, -np.inf], [np.inf, -np.inf], {"inf": "same_sign"})["passed"])
        self.assertFalse(self.compare([np.inf], [-np.inf], {"inf": "same_sign"})["passed"])

    def test_nonfinite_violation_not_hidden_by_mismatch_allowance(self):
        result = self.compare([np.nan], [np.nan], {"max_mismatch_ratio": 1})
        self.assertFalse(result["passed"])

    def test_explicit_mismatch_policy_and_failure_limit(self):
        actual, expected = [2, 3, 1, 1], [1, 1, 1, 1]
        strict = self.compare(actual, expected)
        allowed = self.compare(actual, expected, {"max_mismatch_ratio": 0.5, "max_failures": 1})
        self.assertFalse(strict["passed"])
        self.assertTrue(allowed["passed"])
        output = allowed["outputs"][0]
        self.assertEqual(output["mismatch_count"], 2)
        self.assertEqual(len(output["failures"]), 1)
        self.assertTrue(output["failure_details_truncated"])

    def test_relative_error_definition_at_zero(self):
        result = self.compare([0, 1], [0, 0])
        self.assertEqual(result["outputs"][0]["max_relative_error"], "Infinity")
        self.assertEqual(result["outputs"][0]["max_absolute_error"], 1)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_results_json_safe(self):
        result = self.compare([np.nan, np.inf], [np.inf, -np.inf])
        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["outputs"][0]["max_absolute_error"])

    def test_empty_outputs_pass_with_contract(self):
        result = compare_outputs([np.empty((0, 3), np.float32)], [np.empty((0, 3))],
                                 [{"shape": [0, 3], "dtype": "float32"}], {})
        self.assertTrue(result["passed"])
        self.assertEqual(result["outputs"][0]["total_elements"], 0)
        self.assertIsNone(result["outputs"][0]["max_absolute_error"])

    def test_scalar_failure_index(self):
        result = compare_outputs([np.array(2, np.float32)], [np.array(1.)], [{"shape": [], "dtype": "float32"}], {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["outputs"][0]["failures"][0]["index"], [])

    def test_multiple_outputs_one_wrong_fails_overall(self):
        result = compare_outputs([np.array([1], np.float32), np.array([3], np.int32)],
                                 [np.array([1.]), np.array([2])],
                                 [{"shape": [1], "dtype": "float32"}, {"shape": [1], "dtype": "int32"}], {})
        self.assertFalse(result["passed"])
        self.assertTrue(result["outputs"][0]["passed"])
        self.assertFalse(result["outputs"][1]["passed"])

    def test_int64_difference_cannot_disappear_in_float64(self):
        result = compare_outputs([np.array([2**63 - 1], np.int64)], [np.array([2**63 - 2], np.int64)],
                                 [{"shape": [1], "dtype": "int64"}], {"atol": 0, "rtol": 0})
        self.assertFalse(result["passed"])
        failure = result["outputs"][0]["failures"][0]
        self.assertEqual(failure["actual"], 2**63-1)
        self.assertEqual(failure["reference"], 2**63-2)
        self.assertEqual(failure["absolute_error"], 1)


if __name__ == "__main__":
    unittest.main()
