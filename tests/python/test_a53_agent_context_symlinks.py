"""A53-R38: a symlink inside the root must not hand a private file to a worker.

safe_context applied the private-path filter (.env*, *secret*, *credential*, .git, .runs, ...)
and the extension filter to the requested name only, then followed symlinks and checked only
that the target stayed inside the root: `notes.txt -> .env` returned the .env content.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from vf_agents import safe_context  # noqa: E402


class ContextSymlinks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src/a.rs").write_text("fn main() {}\n")
        (self.root / ".env").write_text("TOKEN=dummy\n")
        (self.root / "secrets.yaml").write_text("key: dummy\n")
        (self.root / ".runs/run1").mkdir(parents=True)
        (self.root / ".runs/run1/x.json").write_text("{}\n")
        (self.root / ".git").mkdir()
        (self.root / ".git/config.txt").write_text("[core]\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_link_to_a_private_file_is_refused(self):
        for link, target in (
            ("notes.txt", ".env"),
            ("config.yaml", "secrets.yaml"),
            ("ledger.json", ".runs/run1/x.json"),
        ):
            (self.root / link).symlink_to(target)
            with self.assertRaises(ValueError, msg=link):
                safe_context(self.root, [link])

    def test_a_directory_link_into_a_private_tree_is_refused(self):
        (self.root / "docs").symlink_to(".git", target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_context(self.root, ["docs/config.txt"])

    def test_a_link_to_an_ordinary_source_file_still_works(self):
        (self.root / "src/b.rs").symlink_to("a.rs")
        self.assertEqual(
            safe_context(self.root, ["src/b.rs"]), {"src/b.rs": "fn main() {}\n"}
        )


if __name__ == "__main__":
    unittest.main()
