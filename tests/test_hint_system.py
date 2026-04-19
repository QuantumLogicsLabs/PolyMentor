from src.reasoning_engine.hint_system import HintSystem

def test_assignment_hint():
    hs = HintSystem()

    hints = hs.generate_hints(
        "assignment_in_condition",
        "if x = 5:"
    )

    assert len(hints) >= 3
    assert any("==" in h for h in hints)

def test_backward_compatibility():
    hs = HintSystem()
    hints = hs.get_hints("syntax_error", "beginner")
    
    assert len(hints) >= 3
    assert "colons" in hints[1] or "brackets" in hints[1]

def test_advanced_level_hint():
    hs = HintSystem()
    hints = hs.get_hints("assignment_in_condition", "advanced")
    
    assert len(hints) == 1
    assert "==" in hints[0]