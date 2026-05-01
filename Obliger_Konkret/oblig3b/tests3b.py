import numbers

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def test_random_policy(input_function, message_on_pass=False):
    """
    Tests the random policy function `input_function`.
    It runs multiple trials to ensure that all actions in the valid action space
    are chosen at least once, and that no invalid actions are ever chosen.

    Args:
        input_function (callable): The policy function to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes.
            If not, prints nothing if all the tests passes. Defaults to False.

    Returns:
        bool / None: True if all tests passes, `None` if not. This is use by `test_epsilon_greedy_policy()`.
    """
    np.random.seed(seed=57)
    n_calls = 1000  # Call the input-function many times, so that it is very likely to get all of the retun values
    message_infix = "`test_random_policy`"

    test_cases = [
        np.zeros((1, 2)),  # two actions
        np.zeros((1, 3)),  # three actions
        np.zeros((5, 5)),  # five actions
    ]

    for i, q_table in enumerate(test_cases, start=1):
        n_actions = q_table.shape[1]
        try:
            results = [input_function(q_table, 0) for _ in range(n_calls)]

            for result in results:
                if result is None:
                    print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                    return
                if not isinstance(result, (int, np.integer)):
                    print(f"Failed: {message_infix}. Test {i} returned non-integer action: `{result}`. ")
                    return
                if not (0 <= result < n_actions):
                    print(f"Failed: {message_infix}. Test {i} returned out-of-range action: `{result}`. ")
                    return

            # All possible actions should appear at least once (with a very high probabability)
            observed_actions = set(results)
            expected_actions = set(range(n_actions))
            if observed_actions != expected_actions:
                print(
                    f"Failed: {message_infix}. Test {i} did not produce all possible actions. "
                    f"Expected {sorted(expected_actions)}, got {sorted(observed_actions)}. "
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test {i} got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")

    return True


def test_greedy_policy(input_function, message_on_pass=False):
    """
    Tests the greedy policy function `input_function`.
    It verifies that the function always returns the action index
    with the highest Q-value for the given state.

    Args:
        input_function (callable): The policy function to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes.
            If not, prints nothing if all the tests passes. Defaults to False.

    Returns:
        bool / None: True if all tests passes, `None` if not. This is use by `test_epsilon_greedy_policy()`.
    """
    message_infix = "`test_greedy_policy`"

    test_cases = [
        (np.array([[1.0, 2.0, 3.0]]), 0, 2),
        (np.array([[0.0, 1.0], [5.0, 4.0]]), 0, 1),
        (np.array([[0.0, 1.0], [5.0, 4.0]]), 1, 0),
        (np.array([[2.0, -1.0, 1.0]]), 0, 0),
        (np.array([[-3.0, -5.0, -1.0]]), 0, 2),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ),
            1,
            1,
        ),
    ]

    for i, (q_table, state, expected_action) in enumerate(test_cases, start=1):
        try:
            result = input_function(q_table, state)

            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, (int, np.integer)):
                print(f"Failed: {message_infix}. Expected int as return type, but got `{type(result)}`. ")
                return
            if result != expected_action:
                print(
                    f"Failed: {message_infix}. Test {i} with state={state}, "
                    f"expected `{expected_action}`, got `{result}`. "
                )
                return

        except Exception as e:
            print(f"Failed: {message_infix}. Test {i} got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        n = len(test_cases)
        print(f"Passed: {message_infix}. All [{n}/{n}] tests passed. ")

    return True


def test_epsilon_greedy_policy(input_class, message_on_pass=False):
    """
    Tests the epsilon-greedy policy class `input_class`.

    Args:
        input_class (class): The epsilon-greedy policy class to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the tests pass.
            If not, prints nothing if all the tests pass. Defaults to False.
    """
    message_infix = "`test_epsilon_greedy_policy`"

    try:
        # Test that it behaves like the greedy policy with epsilon equal to 0.
        epsilon_zero_policy = input_class(epsilon=0.0)
        test1 = test_greedy_policy(epsilon_zero_policy)
        if test1 is None:
            return

        # Run the random test when epsilon is over zero.
        epsilon_policy = input_class(epsilon=0.5)
        test2 = test_random_policy(epsilon_policy)
        if test2 is None:
            return

        epsilon_policy = input_class(epsilon=1)
        test3 = test_random_policy(epsilon_policy)
        if test3 is None:
            return

    except Exception as e:
        print(f"Failed: {message_infix}. Unexpected error: `{e}`. ")
        return

    if all([test1, test2, test3]) and message_on_pass:
        print(f"Passed: {message_infix}. All [3/3] tests passed. ")


def test_softmax_policy(input_class, message_on_pass=False):
    """
    Tests the softmax (Boltzmann) policy class `input_class`.

    It checks that:
      - The returned actions are valid integers within the correct range.
      - The function is implemented (not None).
      - With a low temperature, the policy acts greedily most of the time.
      - With a high temperature, all possible actions are sometimes chosen.

    Args:
        input_class (class): The SoftmaxPolicy class to test.
        message_on_pass (bool, optional): If `True`, will print a message if all tests pass.
            If not, prints nothing if all the tests pass. Defaults to False.
    """
    np.random.seed(seed=57)
    n_calls = 1000
    message_infix = "`test_softmax_policy`"

    # Test 1: low temperature (greedy-like behavior)
    try:
        q_table = np.array([[1.0, 2.0, 3.0]])  # Greedy action = 2
        policy_lowT = input_class(temperature=0.05)
        n_actions = q_table.shape[1]

        results = [policy_lowT(q_table, 0) for _ in range(n_calls)]

        for result in results:
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, (int, np.integer)):
                print(f"Failed: {message_infix}. Returned non-integer action `{result}`. ")
                return
            if not (0 <= result < n_actions):
                print(f"Failed: {message_infix}. Returned out-of-range action `{result}`. ")
                return

        # The greedy action should dominate for low temperature
        counts = np.bincount(results, minlength=n_actions)
        greedy_action = np.argmax(q_table[0])
        most_frequent_action = np.argmax(counts)
        if most_frequent_action != greedy_action:
            print(
                f"Failed: {message_infix}. Low-temperature test did not behave greedily. "
                f"Greedy action `{greedy_action}` vs most frequent `{most_frequent_action}`. "
            )
            return
        if counts[greedy_action] / n_calls < 0.8:
            print(
                f"Failed: {message_infix}. Low-temperature test too random. "
                f"Greedy action chosen only {counts[greedy_action] / n_calls:.2f} of the time. "
            )
            return

    except Exception as e:
        print(f"Failed: {message_infix}. Low-temperature test got unexpected error: `{e}`. ")
        return

    # High temperature (should explore uniformly)
    try:
        q_table = np.array([[10.0, 0.0, -10.0]])  # Large Q spread
        policy_highT = input_class(temperature=20.0)
        n_actions = q_table.shape[1]

        results = [policy_highT(q_table, 0) for _ in range(n_calls)]

        for result in results:
            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, (int, np.integer)):
                print(f"Failed: {message_infix}. Returned non-integer action `{result}`. ")
                return
            if not (0 <= result < n_actions):
                print(f"Failed: {message_infix}. Returned out-of-range action `{result}`. ")
                return

        observed_actions = set(results)
        expected_actions = set(range(n_actions))
        if observed_actions != expected_actions:
            print(
                f"Failed: {message_infix}. High-temperature test did not produce all actions. "
                f"Expected {sorted(expected_actions)}, got {sorted(observed_actions)}. "
            )
            return

    except Exception as e:
        print(f"Failed: {message_infix}. High-temperature test got unexpected error: `{e}`. ")
        return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [2/2] tests passed. ")


def test_sarsa_update(input_function, message_on_pass=False):
    """
    Tests the SARSA update function `input_function`.

    Args:
        input_function (callable): The sarsa_update function to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes.
            If not, prints nothing if all the tests passes. Defaults to False.
    """
    message_infix = "`test_sarsa_update`"

    test_cases = [
        (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            (0, 0, 1.0, 1, 1),
            0.5,
            1.0,
            np.array([[0.5, 0.0], [0.0, 0.0]]),
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            (0, 1, 1.0, 1, 0),
            0.5,
            0.5,
            np.array([[1.0, 2.25], [3.0, 4.0]]),
        ),
        (
            np.array([[5.0, 0.0, -2.0], [1.0, 2.0, 3.0]]),
            (0, 2, -1.0, 1, 1),
            0.3,
            0.9,
            np.array([[5.0, 0.0, -1.16], [1.0, 2.0, 3.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            (1, 0, 0.5, 2, 2),
            0.1,
            0.9,
            np.array([[1.0, 1.0, 1.0], [2.12, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        ),
    ]

    for i, (q_table, transition, alpha, gamma, expected) in enumerate(test_cases, start=1):
        try:
            q_copy = q_table.copy()
            result = input_function(q_copy, transition, alpha, gamma)

            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, np.ndarray):
                print(f"Failed: {message_infix}. Expected np.ndarray, got `{type(result)}`. ")
                return
            if not np.allclose(result, expected, atol=1e-6):
                print(f"Failed: {message_infix}. Test {i} failed.\nExpected:\n{expected}\nGot:\n{result}")
                return
        except Exception as e:
            print(f"Failed: {message_infix}. Test {i} got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [{len(test_cases)}/{len(test_cases)}] tests passed. ")


def test_q_learning_update(input_function, message_on_pass=False):
    """
    Tests the Q-learning update function `input_function`.

    Args:
        input_function (callable): The q_learning_update function to test.
        message_on_pass (bool, optional): If `True`, will print a message if all the test passes.
            If not, prints nothing if all the tests passes. Defaults to False.
    """
    message_infix = "`test_q_learning_update`"

    test_cases = [
        (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            (0, 0, 1.0, 1, None),  # (s, a, r, s', _)
            0.5,
            1.0,
            np.array([[0.5, 0.0], [0.0, 0.0]]),
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            (0, 1, 1.0, 1, None),
            0.5,
            0.5,
            np.array([[1.0, 2.5], [3.0, 4.0]]),
        ),
        (
            np.array([[5.0, 0.0, -2.0], [1.0, 2.0, 3.0]]),
            (0, 2, -1.0, 1, None),
            0.3,
            0.9,
            np.array([[5.0, 0.0, -0.89], [1.0, 2.0, 3.0]]),
        ),
        (
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            (1, 0, 0.5, 2, None),
            0.1,
            0.9,
            np.array([[1.0, 1.0, 1.0], [2.12, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        ),
    ]

    for i, (q_table, transition, alpha, gamma, expected) in enumerate(test_cases, start=1):
        try:
            q_copy = q_table.copy()
            result = input_function(q_copy, transition, alpha, gamma)

            if result is None:
                print(f"Failed: {message_infix}. Not implemented (returned `None`). ")
                return
            if not isinstance(result, np.ndarray):
                print(f"Failed: {message_infix}. Expected np.ndarray, got `{type(result)}`. ")
                return
            if not np.allclose(result, expected, atol=1e-6):
                print(f"Failed: {message_infix}. Test {i} failed.\nExpected:\n{expected}\nGot:\n{result}")
                return
        except Exception as e:
            print(f"Failed: {message_infix}. Test {i} got unexpected error: `{e}`. ")
            return

    if message_on_pass:
        print(f"Passed: {message_infix}. All [{len(test_cases)}/{len(test_cases)}] tests passed. ")
