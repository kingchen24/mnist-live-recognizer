# Source Assets / 原始素材

This folder contains the **original source assets** used to generate the demo GIFs and screenshots in `docs/`.

## Files

| File | Used For | In README |
|------|----------|-----------|
| `camra.png` | Real camera mode screenshot showing hand-written "5" with 87.7% confidence | `docs/images/10_real_camera_screenshot.png` |
| `mouse mode.mp4` | Real screen recording of mouse drawing mode | `docs/gifs/mouse_mode_demo.gif` (converted) |

## Notes

- The MP4 is the **uncompressed** source. The README uses an optimized GIF (3MB) converted from this video.
- These are kept here so the README assets can be regenerated if needed.
- To regenerate the GIF: use `ffmpeg` or PIL with a frame skip of 2-3 and reduced color palette.

```bash
# Example: regenerate mouse_mode_demo.gif
python scripts/convert_mp4_to_gif.py "pic/mouse mode.mp4" docs/gifs/mouse_mode_demo.gif
```
