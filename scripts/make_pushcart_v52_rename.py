"""One-off: rename V5.1 identifiers inside the copied HomieRL/homie_pushcart_v52 package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "HomieRL/homie_pushcart_v52"
REPLACEMENTS = (("homie_pushcart_v51", "homie_pushcart_v52"), ("V51", "V52"), ("v51", "v52"), ("V5.1", "V5.2"),
                ("version = 51", "version = 52"), ('get("version") != 51', 'get("version") != 52'),
                ("task.version=51", "task.version=52"), ('"version": 51', '"version": 52'))
for path in sorted(ROOT.rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    new = text
    for old, repl in REPLACEMENTS:
        new = new.replace(old, repl)
    if new != text:
        path.write_text(new, encoding="utf-8")
        print("renamed in", path.relative_to(ROOT))
