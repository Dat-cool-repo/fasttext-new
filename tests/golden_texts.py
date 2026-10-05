"""Edge-case inputs shared by the parity tests and the golden files."""

EDGE_CASES = [
    "",
    " ",
    "\t \t",
    "a\x0bb",  # vertical tab is a separator in C++
    "a\x00b",  # so is NUL
    "a\rb c\x0cd",
    "a b",  # NBSP is *not* a separator
    "a　b",
    "__label__en hello world",
    "hello </s> world after eos is ignored",
    "   leading and trailing   ",
    "😀😃😄 emoji only",
    "x" * 3000,
    " ".join(["word"] * 2000),
]
