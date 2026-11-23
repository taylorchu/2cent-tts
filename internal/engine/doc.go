// Package engine speaks text with the models of one release file: the tokenizer, the speech model, the codec and the
// speech detector.
//
// A caller loads the release once with Load, then for each request prepares the text with Prepare, makes a voice with
// NewReference and NewVoice, or takes NoVoice, and speaks with Speak. An Engine, its voices and its scripts never
// change once made, so any number of goroutines may share them; each Speak works in memory of its own.
//
// Speech comes out as mono float samples at audio.SampleRate, nominally between -1 and 1 and not clipped.
//
// The package uses these words:
//
//   - unit: a word or a sound tag with the text up to the next one. The model is fed text a unit at a time.
//   - step: model.StepFrames SNAC frames, the codes the model writes at once.
//   - start: the model's decision, after a step, that a unit started being spoken in it. Each start feeds the
//     next unit; the start that finds no unit left ends the speech.
//   - heard: the steps both passes have read, written or fed.
//   - chunk: a stretch of text spoken from a fresh copy of the voice's prompt, at most a few sentences long. A chunk
//     after the first opens with the last units of the one before and feeds their steps before it draws, so the
//     speech carries on across the join; tail.fill holds the rule.
//   - voice's prompt: <s>, then a reference clip's text and codes. A voice reads it once.
//   - static prompt: every position before a target opens, which stays in view for the whole chunk. It is the
//     voice's prompt, then the chunk's whole text, which each chunk reads. The markers type holds the layout.
package engine
