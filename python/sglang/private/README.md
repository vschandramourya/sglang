# README

Lightweight mechanism for shipping drop-in overrides of upstream sglang.srt.* modules and keeping all one-off monkey patches in one place.

- Mirror overrides live under `sglang/private/*` with the same path as the upstream you’re replacing.

- No symbols or package names are exposed to users beyond the public `sglang.srt.*` API.

## Quick start

1. Enable the override finder early (before anything imports sglang.srt):

   ```python
   # E.g. python/sglang/launch_server.py
   import sglang.private.patches
    ```

2. Any sglang.srt.X that has a twin at sglang.private.X will be replaced automatically. Everything else comes from upstream unchanged.

## Layout

```
sglang/
  srt/                       # upstream (public) modules
    ...
  private/
    patches/                 # put ALL monkey patches here
      __init__.py
      auto_override.py       # the meta-path finder
      000_fix_foo.py
      010_warn_deprecated.py
    layers/
      linear.py              # example mirror override (replaces srt.layers.linear)
    entrypoints/
      http_server.py         # example mirror override (replaces srt.entrypoints.http_server)
```
