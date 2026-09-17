# Game tutorial demonstrations

Put each game's tutorial recording pair in a directory named after its profile file:

```text
tutorials/<game_id>/tutorial.mp4
tutorials/<game_id>/tutorial.json
```

For example, `configs/profiles/merge_dragons.yaml` uses
`tutorials/merge_dragons/`. File basenames may differ as long as the directory
contains one MP4 and one JSON file. On the first server start, GameAgent samples
the video at recorded action times, asks the configured API model to extract
rules, and writes `configs/profiles/tutorials/<game_id>.yaml`. Later starts reuse
that prior. Runtime discoveries remain separate in `state/<game_id>/rules.json`.
