from agent.subgraphs.unit_test import route_after_validate, route_after_execution

def test_infinite_loop_fix():
    # Scenario 1: Validate fails, retry < max
    state1 = {"validation_passed": False, "retry_count": 0, "max_retries": 3}
    result1 = route_after_validate(state1)
    print(f"Scenario 1 (Validate Fail, Count 0): Expected 'retry', got '{result1}'")
    assert result1 == "retry"

    # Scenario 2: Validate fails, retry == max
    state2 = {"validation_passed": False, "retry_count": 3, "max_retries": 3}
    result2 = route_after_validate(state2)
    print(f"Scenario 2 (Validate Fail, Count 3): Expected 'max_retries', got '{result2}'")
    assert result2 == "max_retries"

    # Scenario 3: Execution fails, retry < max
    state3 = {"execution_passed": False, "retry_count": 0, "max_retries": 3}
    result3 = route_after_execution(state3)
    print(f"Scenario 3 (Execution Fail, Count 0): Expected 'retry', got '{result3}'")
    assert result3 == "retry"

    # Scenario 4: Execution fails, retry == max
    state4 = {"execution_passed": False, "retry_count": 3, "max_retries": 3}
    result4 = route_after_execution(state4)
    print(f"Scenario 4 (Execution Fail, Count 3): Expected 'max_retries', got '{result4}'")
    assert result4 == "max_retries"

    print("\nVerification successful! Routing logic correctly handles retry limits.")

if __name__ == "__main__":
    test_infinite_loop_fix()
