# Runtime

CLI entrypoint, config loading, adapter wiring, lifecycle, and stop conditions live here.

The runtime owns the loop timing, but it should call interfaces rather than concrete implementations directly.

