"""Extract public declarations and inherent methods from cargo doc --no-deps --lib."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path


class Declarations(HTMLParser):
    def __init__(self):
        super().__init__()
        self.declaration = ""
        self.methods = []
        self.method = False
        self.capture = ""
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "section":
            self.method = "method" in classes and "trait-impl" not in classes
        if tag == "pre" and "item-decl" in classes or tag == "h4" and self.method:
            self.capture = tag
            self.parts = []

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "section":
            self.method = False
        if tag == self.capture:
            signature = " ".join("".join(self.parts).split())
            if tag == "pre":
                self.declaration = signature
            elif signature:
                self.methods.append(signature)
            self.capture = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crate_docs", type=Path, help="the generated doc/fast_audio directory")
    args = parser.parse_args()
    entries = []
    for path in sorted(args.crate_docs.rglob("*.html")):
        parts = path.stem.split(".", 1)
        if len(parts) != 2:
            continue
        content = Declarations()
        content.feed(path.read_text(encoding="utf-8"))
        if content.declaration:
            module = "::".join(path.parent.relative_to(args.crate_docs).parts)
            entries.append({"name": parts[1], "module": module, "decl": content.declaration, "methods": content.methods})
    if not any(item["name"] == "Tensor" and item["methods"] for item in entries):
        raise ValueError("The input must contain public fast_audio Rustdoc declarations.")
    output = Path(__file__).resolve().parents[1] / "src/data/api-index.json"
    output.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
