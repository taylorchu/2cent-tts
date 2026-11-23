package engine_test

import (
	"context"
	"log"
	"math/rand/v2"
	"os"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/engine"
)

// Example speaks a text in a voice cloned from a reference clip and writes the speech as a WAV file.
func Example() {
	e, err := engine.Load("build/2cent.safetensors", engine.Options{})
	if err != nil {
		log.Fatal(err)
	}

	// The text is checked first: it costs no model work, and a voice costs an encode.
	script, err := e.Prepare("Hello there. How are you today?")
	if err != nil {
		log.Fatal(err)
	}

	clip, err := os.ReadFile("reference.wav")
	if err != nil {
		log.Fatal(err)
	}

	reference, err := e.NewReference(clip, "What the reference clip says.")
	if err != nil {
		log.Fatal(err)
	}

	voice, err := e.NewVoice(context.Background(), reference)
	if err != nil {
		log.Fatal(err)
	}

	var speech []float32
	_, err = e.Speak(context.Background(), voice, script, rand.NewPCG(1, 2), func(span []float32) error {
		speech = append(speech, span...)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	err = os.WriteFile("out.wav", audio.EncodeWAV(speech), 0o644)
	if err != nil {
		log.Fatal(err)
	}
}
