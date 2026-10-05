import pytest
from calculator import add, divide

def test_add():
    assert add(2, 3) == 5

def test_divide():
    assert divide(10, 2) == 5

# The tester agent will use this to verify the Coder's fix
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)