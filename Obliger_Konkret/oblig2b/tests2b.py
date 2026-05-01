import numbers

import numpy as np


def _run_activation_function_tests(input_class, message_infix, test_cases, message_on_pass=False):
    """
    Internal helper function that loops over test cases for activation function classes.
    Each test case verifies both the forward pass (__call__) and derivative (diff) outputs.

    Args:
        input_class (class): The activation function class to test (e.g., Sigmoid or ReLU).
        message_infix (str): The test name to include in messages.
        test_cases (list): List of tuples (x_data, expectedforward, expected_diff).
        message_on_pass (bool, optional): If `True`, prints message on success.
    """
    for i, (x_data, expectedforward, expected_diff) in enumerate(test_cases, start=1):
        try:
            activation = input_class()

            # Forward pass
            forward = activation(x_data)
            if forward is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`) in forward pass.")
                return
            if not isinstance(forward, np.ndarray):
                print(f"Failed: {message_infix}. Expected `np.ndarray` from forward pass, got `{type(forward)}`.")
                return
            if forward.shape != expectedforward.shape:
                print(
                    f"Failed: {message_infix}. Wrong forward output shape for test `{i}`. "
                    f"Expected `{expectedforward.shape}`, got `{forward.shape}`."
                )
                return
            if not np.allclose(forward, expectedforward, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Wrong forward output for test `{i}`. "
                    f"Expected `{expectedforward}`, got `{forward}`."
                )
                return

            # Derivative
            derivative = activation.diff(x_data)
            if derivative is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`) in derivative.")
                return
            if not isinstance(derivative, np.ndarray):
                print(f"Failed: {message_infix}. Expected `np.ndarray` from derivative, got `{type(derivative)}`.")
                return
            if derivative.shape != expected_diff.shape:
                print(
                    f"Failed: {message_infix}. Wrong derivative output shape for test `{i}`. "
                    f"Expected `{expected_diff.shape}`, got `{derivative.shape}`."
                )
                return
            if not np.allclose(derivative, expected_diff, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Wrong derivative output for test `{i}`. "
                    f"Expected `{expected_diff}`, got `{derivative}`."
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test number `{i}` got unexpected error: `{e}`.")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed.")


def test_sigmoid_class(input_class, message_on_pass=False):
    """
    Tests the `Sigmoid` class for both forward and derivative outputs.
    Ensures that sigmoid and its derivative are computed correctly for
    [n x p] matrix inputs.
    """
    message_infix = "`test_sigmoid_class`"

    test_cases = [
        (
            np.array([[0.0, 1.0], [-1.0, 2.0]]),
            np.array([[0.5, 0.73105858], [0.26894142, 0.88079708]]),
            np.array([[0.25, 0.19661193], [0.19661193, 0.10499359]]),
        ),
        (
            np.array([[-10.0, 10.0], [0.0, -2.0]]),
            np.array([[4.53978687e-05, 9.99954602e-01], [5.00000000e-01, 1.19202922e-01]]),
            np.array([[4.53958077e-05, 4.53958077e-05], [2.50000000e-01, 1.04993590e-01]]),
        ),
        (
            np.array([[2.0, -2.0], [0.0, 1.0]]),
            np.array([[0.88079708, 0.11920292], [0.5, 0.73105858]]),
            np.array([[0.10499359, 0.10499359], [0.25, 0.19661193]]),
        ),
    ]

    _run_activation_function_tests(input_class, message_infix, test_cases, message_on_pass)


def test_relu_class(input_class, message_on_pass=False):
    """
    Tests the `ReLU` class for both forward and derivative outputs.
    Ensures that ReLU and its derivative are computed correctly for
    [n x p] matrix inputs.
    """
    message_infix = "`test_relu_class`"

    test_cases = [
        (
            np.array([[1.0, -1.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            np.array([[-2.0, -3.0], [4.0, 5.0], [0.0, -1.0]]),
            np.array([[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        ),
        (
            np.array([[10.0, -10.0], [-5.0, 2.0]]),
            np.array([[10.0, 0.0], [0.0, 2.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ]

    _run_activation_function_tests(input_class, message_infix, test_cases, message_on_pass)


def test_count_parameters(input_class, message_on_pass=False):
    """
    Tests the `count_parameters()` method of the NeuralNetwork class.
    Verifies that the total number of trainable parameters (weights + biases)
    is computed correctly for different network architectures.

    Args:
        input_class (class): The NeuralNetwork class to test.
        message_on_pass (bool, optional): If `True`, will print a message when all tests pass.
    """
    message_infix = "`test_count_parameters`"

    # Each test case is (layer_sizes, expected_param_count)
    test_cases = [
        # Single hidden layer
        ([2, 1], 3),  # 2 weights + 1 bias = 3
        ([3, 2], 8),  # 3*2 + 2 = 8
        ([4, 3, 2], 23),  # 12 + 3 + 6 + 2 = 23
        ([5, 5, 5], 60),  # 25 + 5 + 25 + 5 = 60
        ([784, 32, 16, 10], 25818),
        ([10, 10, 10, 10, 10], 440),
    ]

    for i, (layer_sizes, expected) in enumerate(test_cases, start=1):
        try:
            neural_network = input_class(
                layer_sizes=layer_sizes,
                activation_functions=[None] * (len(layer_sizes) - 2),
            )

            result = neural_network.count_parameters()
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (`None` returned).")
                return
            if not isinstance(result, numbers.Integral):
                print(f"Failed: {message_infix}. Expected integer result, got `{type(result)}`.")
                return
            if result != expected:
                print(
                    f"Failed: {message_infix}. Test `{i}` for layer_sizes=`{layer_sizes}`. "
                    f"Expected `{expected}`, got `{result}`."
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test `{i}` got unexpected error: `{e}`.")
            return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [{len(test_cases)}/{len(test_cases)}] tests passed.")


def test_forward_pass(input_class, relu_class, message_on_pass=False):
    """
    Tests the `forward()` method of the NeuralNetwork class.
    Verifies that the forward pass produces correct logits, weighted sums,
    and intermediate activations when all weights are initialized to ones and biases to zeros.

    Args:
        input_class (class): The NeuralNetwork class to test.
        relu_class (class): The ReLU activation class to use.
        message_on_pass (bool, optional): If `True`, prints a message if all tests pass. Defaults to False.
    """
    message_infix = "`testforward_pass`"

    # Each test case: (layer_sizes, x_data, expected_weighted_sums, expected_activations, expected_logits)
    test_cases = []

    # --- Test case 1: [2, 2, 1] network ---
    # All weights = 1, all biases = 0, activation = ReLU
    # Input: 2 samples, 2 features
    layer_sizes = [2, 2, 1]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, 2.0], [0.0, -1.0]])  # Shape: [2, 2]

    expected_weighted_sums = [np.array([[3.0, 3.0], [-1.0, -1.0]]), np.array([[6.0], [0.0]])]
    expected_activations = [x_data, np.array([[3.0, 3.0], [0.0, 0.0]]), np.array([[6.0], [0.0]])]
    expected_logits = np.array([[6.0], [0.0]])

    test_cases.append((layer_sizes, x_data, expected_weighted_sums, expected_activations, expected_logits))

    # --- Test case 2: [3, 2, 2] network ---
    # All weights = 1, all biases = 0
    # Input: one sample, three features
    layer_sizes = [3, 2, 2]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, -1.0, 2.0]])  # Shape: [1, 3]

    expected_weighted_sums = [np.array([[2.0, 2.0]]), np.array([[4.0, 4.0]])]
    expected_activations = [x_data, np.array([[2.0, 2.0]]), np.array([[4.0, 4.0]])]
    expected_logits = np.array([[4.0, 4.0]])

    test_cases.append((layer_sizes, x_data, expected_weighted_sums, expected_activations, expected_logits))

    # --- Run all test cases ---
    for i, (layer_sizes, x_data, expected_weighted_sums, expected_activations, expected_logits) in enumerate(
        test_cases, start=1
    ):
        try:
            neural_network = input_class(
                layer_sizes=layer_sizes,
                activation_functions=activation_functions,
                initialization_method="ones",
            )

            result = neural_network.forward(x_data)

            # Check output and types
            if result is None:
                print(f"Failed: {message_infix}. Test `{i}` returned None.")
                return
            if not isinstance(result, np.ndarray):
                print(f"Failed: {message_infix}. Expected np.ndarray return type, got `{type(result)}`.")
                return
            if result.shape != expected_logits.shape:
                print(
                    f"Failed: {message_infix}. Wrong output shape for test `{i}`. "
                    f"Expected `{expected_logits.shape}`, got `{result.shape}`."
                )
                return
            if not np.allclose(result, expected_logits, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Wrong logits values for test `{i}`. "
                    f"Expected `{expected_logits}`, got `{result}`."
                )
                return

            # Check internal variables
            if not hasattr(neural_network, "weighted_sums") or not hasattr(neural_network, "activations"):
                print(f"Failed: {message_infix}. NeuralNetwork missing `weighted_sums` or `activations` attributes.")
                return

            if len(neural_network.weighted_sums) != len(expected_weighted_sums):
                print(
                    f"Failed: {message_infix}. Wrong number of weighted sums. "
                    f"Expected `{len(expected_weighted_sums)}`, got `{len(neural_network.weighted_sums)}`."
                )
                return

            if len(neural_network.activations) != len(expected_activations):
                print(
                    f"Failed: {message_infix}. Wrong number of activations. "
                    f"Expected `{len(expected_activations)}`, got `{len(neural_network.activations)}`."
                )
                return

            # Check weighted_sums and activations layer by layer
            for j, (z_expected, z_actual) in enumerate(zip(expected_weighted_sums, neural_network.weighted_sums)):
                if not np.allclose(z_expected, z_actual, atol=1e-6):
                    print(
                        f"Failed: {message_infix}. Weighted sum mismatch at layer {j + 1} in test `{i}`. "
                        f"Expected `{z_expected}`, got `{z_actual}`."
                    )
                    return

            for j, (a_expected, a_actual) in enumerate(zip(expected_activations, neural_network.activations)):
                if not np.allclose(a_expected, a_actual, atol=1e-6):
                    print(
                        f"Failed: {message_infix}. Activation mismatch at layer {j} in test `{i}`. "
                        f"Expected `{a_expected}`, got `{a_actual}`."
                    )
                    return

        except Exception as e:
            print(f"Failed: {message_infix}. Test `{i}` got unexpected error: `{e}`.")
            return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [{len(test_cases)}/{len(test_cases)}] tests passed.")


def test_predict(input_class, relu_class, message_on_pass=False):
    """
    Tests the `predict()` method of the NeuralNetwork class.
    Verifies that the predicted classes correspond to the maximum logits from the forward pass.

    Args:
        input_class (class): The NeuralNetwork class to test.
        relu_class (class): The ReLU activation class to use.
        message_on_pass (bool, optional): If `True`, prints a message when all tests pass. Defaults to False.
    """
    message_infix = "`test_predict`"

    # Each test case: (layer_sizes, x_data, expected_predictions)
    test_cases = []

    layer_sizes = [2, 2, 2]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, 2.0], [0.0, -1.0], [2.0, 2.0]])  # Shape: [3, 2]
    expected_predictions = np.array([0, 0, 0])
    test_cases.append((layer_sizes, x_data, expected_predictions))

    layer_sizes = [3, 3, 3]
    activation_functions = [relu_class(), relu_class()]
    x_data = np.array([[1.0, -2.0, 3.0]])  # Shape [1, 3]
    expected_predictions = np.array([0])
    test_cases.append((layer_sizes, x_data, expected_predictions))

    # Run all test cases
    for i, (layer_sizes, x_data, expected_predictions) in enumerate(test_cases, start=1):
        try:
            model = input_class(
                layer_sizes=layer_sizes,
                activation_functions=activation_functions,
                initialization_method="ones",
            )

            result = model.predict(x_data)

            # Check type and shape
            if result is None:
                print(f"Failed: {message_infix}. Test `{i}` returned None.")
                return
            if not isinstance(result, np.ndarray):
                print(f"Failed: {message_infix}. Expected np.ndarray return type, got `{type(result)}`.")
                return
            if result.ndim != 1:
                print(f"Failed: {message_infix}. Expected 1D output, got shape `{result.shape}`.")
                return
            if result.shape[0] != x_data.shape[0]:
                print(
                    f"Failed: {message_infix}. Wrong number of predictions. "
                    f"Expected `{x_data.shape[0]}`, got `{result.shape[0]}`."
                )
                return
            if not np.array_equal(result, expected_predictions):
                print(
                    f"Failed: {message_infix}. Wrong predictions in test `{i}`. "
                    f"Expected `{expected_predictions}`, got `{result}`."
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test `{i}` got unexpected error: `{e}`.")
            return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [{len(test_cases)}/{len(test_cases)}] tests passed.")


def _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass=False):
    """
    Internal helper function that loops over test cases for a scalar loss function.
    Verifies that each test case returns the correct numeric value.
    Also works for accuracy, even though it is not usually a loss function.

    Args:
        input_function (callable): The loss function to test.
        message_infix (str): The test name to include in messages.
        test_cases (list): List of tuples (y_data, predictions, expected).
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes. If not,
            prints nothing if all the tests passes. Defaults to False.
    """
    for i, (y_data, predictions, expected) in enumerate(test_cases, start=1):
        try:
            result = input_function(y_data, predictions)
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, numbers.Number):
                print(
                    f"Failed: {message_infix}. Expected number as return type, "
                    f"but got value `{result}` with type `{type(result)}`. "
                )
                return
            if not np.isclose(result, expected, atol=1e-6):
                print(
                    f"Failed: {message_infix}. Test number `{i}` with input y_data=`{y_data}`, "
                    f"predictions=`{predictions}`. Expected `{expected}`, got `{result}`. "
                )
                return
        except Exception as e:
            print(f"Failed: {message_infix}. Test number `{i}` got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")


def test_calculate_multiclass_cross_entropy_loss(input_function, message_on_pass=False):
    """
    Loops over test cases for the function `input_function` (multiclass cross-entropy loss).
    Each test case consists of integer targets and logits (not softmaxed).
    Ensures that the function returns the correct scalar loss value.

    Args:
        input_function (callable): The loss function to test.
        message_on_pass (bool, optional): If `True`, prints a message when all tests pass.
            Defaults to False.
    """
    message_infix = "`test_calculate_multiclass_cross_entropy_loss`"

    # Each test case: (y_true, logits, expected_loss)
    # Expected losses are computed from the exact softmax values provided.

    test_cases = [
        (
            np.array([0, 1]),
            np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
            0.0359762955725428,
        ),
        (
            np.array([0, 2, 2]),
            np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]]),
            2.702643041017243,
        ),
        (
            np.array([0, 1, 2]),
            np.zeros((3, 3)),
            1.0986122886681098,
        ),
        (
            np.array([0, 3, 1, 2]),
            np.array([[3.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 2.0, 0.5, 0.0], [0.0, 0.0, 2.0, 1.0]]),
            0.6438389517485247,
        ),
    ]

    _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass)


def test_calculate_accuracy(input_function, message_on_pass=False):
    """
    Loops over test cases for the function `input_function` (classification accuracy).
    There are several hardcoded unit tests with input to the function and expected returns.
    Loops over the test cases and asserts with a suitable message if the test fails.
    Catches any unexpected exceptions and fails with them.

    Args:
        input_function (callable): The function to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes. If not,
            prints nothing if all the tests passes. Defaults to False.
    """
    message_infix = "`test_calculate_accuracy`"

    test_cases = [
        (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), 1.0),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), 0.0),
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.5),
        (np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1]), 0.8),
        (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 1]), 0.75),
        (np.array([1]), np.array([1]), 1.0),
    ]

    _run_loss_function_tests(input_function, message_infix, test_cases, message_on_pass)
