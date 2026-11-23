// Package audio holds the model's codec and what prepares audio for it: SNAC 24 kHz with its convolution stages,
// the Silero speech detector, resampling and WAV files. It depends on the tensor package for its kernels and
// weights. Of the model it knows only the slot order frames are packed in, which its caller gives.
package audio

const (
	// SampleRate is SNAC 24 kHz's samples a second.
	SampleRate = 24000
	// FrameSamples is the samples one frame of codes decodes to.
	FrameSamples = 2048
	// FrameCodes is the codes in a frame: one of level 0, two of level 1 and four of level 2.
	FrameCodes = 7
	// LevelSize is the codes each of SNAC's levels holds.
	LevelSize = 4096
	// LevelCount is how many levels SNAC 24 kHz codes a frame in.
	LevelCount = 3
	// CodeCount is how many codes the levels hold, each level in its own range.
	CodeCount = LevelCount * LevelSize
	// VADRate is the speech detector's sample rate.
	VADRate = 16000
)
