# Experimental ideas

## Phlash

 * Modularise `main()` to help readability, development of interactive features (on/off, flip, etc.)
 * Isolate Linux specifics (V4L2 in particular), to help porting efforts to Windows/OSX/etc.
   `loopback.cc` is already a useful start here, adopting modern (C++17) abstractions also helpful.
 * Integrate to [GStreamer](https://github.com/GStreamer/gstreamer) as a filter plugin. Currently
   investigating the basics of writing a GStreamer plugin & considering how much processing _should_
   remain within the plugin: mask creation via TF - yep; alpha blending foreground/background - maybe.
 * Investigate if [NNStreamer](https://github.com/nnstreamer/nnstreamer) makes everything redundant?
