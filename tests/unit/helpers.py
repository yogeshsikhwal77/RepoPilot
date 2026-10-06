"""Stand-in chunker for tests until Vibhav's tree-sitter parser lands (stdlib ast, Python only)."""
import ast
from pathlib import Path

from repopilot.orchestrator.state import Chunk


def py_chunks(repo_root, commit="HEAD") -> list[Chunk]:
    root = Path(repo_root)
    chunks: list[Chunk] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        src = p.read_text()
        lines = src.splitlines()
        tree = ast.parse(src)

        def add(node, symbol):
            s, e = node.lineno, node.end_lineno
            chunks.append(Chunk(id=f"{rel}:{s}-{e}", path=rel, start_line=s, end_line=e,
                                symbol=symbol, text="\n".join(lines[s - 1:e]), commit=commit))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node, node.name)
            elif isinstance(node, ast.ClassDef):
                add(node, node.name)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(sub, f"{node.name}.{sub.name}")
    return chunks


SAMPLE_FILES = {
    "shop/__init__.py": "",
    "shop/cart.py": '''class Cart:
    def __init__(self):
        self.items = {}

    def add_item(self, name, price, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        old = self.items.get(name, (price, 0))[1]
        self.items[name] = (price, old + qty)

    def total(self):
        return sum(p * q for p, q in self.items.values())


def format_price(x):
    return f"${x:.2f}"
''',
    "shop/pricing.py": '''def apply_discount(total, pct):
    return total * (1 - pct / 100)
''',
    "tests/test_cart.py": '''from shop.cart import Cart, format_price


def test_add_item():
    c = Cart()
    c.add_item("a", 2.0, 3)
    assert c.items["a"] == (2.0, 3)


def test_total():
    c = Cart()
    c.add_item("a", 2.0, 2)
    assert c.total() == 4.0


def test_format_price():
    assert format_price(3) == "$3.00"
''',
    "tests/test_pricing.py": '''from shop.pricing import apply_discount


def test_discount():
    assert apply_discount(100, 10) == 90
''',
}


def write_sample_repo(root) -> Path:
    root = Path(root)
    for rel, content in SAMPLE_FILES.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return root