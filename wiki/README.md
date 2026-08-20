# Wiki sources

These Markdown files are the source of truth for the
[SAMLB wiki](https://github.com/TechyNilesh/samlb/wiki). They live in the repo
so documentation changes can be reviewed in a pull request like any other
change, instead of being edited straight into the wiki where nothing is
tracked.

## Layout

One file per wiki page. GitHub maps a filename to a page title by replacing
hyphens with spaces, so `Quick-Start.md` becomes the page **Quick Start**, and
links between pages use the title form: `[[Quick Start]]`.

| File | Page |
|------|------|
| `Home.md` | Landing page and index |
| `Installation.md` | Install, optional backends, build troubleshooting |
| `Quick-Start.md` | First benchmark, the model contract, CLI |
| `Benchmark-API.md` | `BenchmarkSuite`, evaluator, `RunResult`, output formats |
| `Datasets.md` | The 30 bundled streams, `stream()` / `load()`, adding your own |
| `Frameworks.md` | Bundled AutoML methods and their configuration |
| `Base-Algorithms.md` | C++ learners, fused pipelines, metrics, drift detectors |
| `External-Algorithms.md` | River and CapyMOA adapters |
| `Extending-SAMLB.md` | Writing a framework, adapter, dataset or C++ learner |
| `FAQ.md` | Common questions and failure modes |

`Home.md` is the landing page; GitHub builds the sidebar from the page list
unless a `_Sidebar.md` is added.

## Publishing

```bash
./scripts/publish_wiki.sh
```

The script clones `https://github.com/TechyNilesh/samlb.wiki.git`, copies every
`.md` from this directory over it, and pushes. Only pages that changed produce
a commit.

**First time only:** GitHub creates the wiki's git repository when the first
page is saved through the web UI. Until then the clone fails with *Repository
not found*. Open <https://github.com/TechyNilesh/samlb/wiki>, click **Create
the first page**, save anything, then run the script — it overwrites that
placeholder with `Home.md`.

## Editing

Edit here, open a PR, publish after it merges. Editing a page directly on
GitHub is not wrong, but the change will be overwritten by the next publish
unless it is copied back into this directory.
