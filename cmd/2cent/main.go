// Command 2cent speaks text with the 2cent-tts model, once from the command line or as an OpenAI compatible speech
// API. The model writes SNAC audio codes a few frames at a time in the voice of a reference clip, reading the text a
// few words ahead of the speech. The SNAC decoder turns the codes into 24 kHz speech.
//
//	GOEXPERIMENT=simd go run ./cmd/2cent -text "Hello there." -voice default -o out.wav
//	GOEXPERIMENT=simd go run ./cmd/2cent -text "Hi." -ref-wav ref.wav -ref-text "What the reference says." -o out.wav
//	GOEXPERIMENT=simd go run ./cmd/2cent -listen :8080
//
// Without GOEXPERIMENT=simd the engine builds with its plain Go kernels, at less than half the speed.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"maps"
	"math/rand/v2"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/engine"
)

// maxRequests bounds -max-requests well past what the cores serve, since the engine keeps memory for each. The engine
// has a looser limit of its own for library callers.
const maxRequests = 64

// options are the command's flags.
type options struct {
	modelPath     string
	voicesDir     string
	text          string
	voiceName     string
	referenceWAV  string
	referenceText string
	output        string
	seed          uint64
	listen        string
	maxRequests   int
	queueWait     time.Duration
}

func main() {
	var o options
	flag.StringVar(&o.modelPath, "model", "build/2cent.safetensors", "release weights")
	flag.StringVar(&o.voicesDir, "voices", "build/voices", "directory of <name>.wav clips, each with a <name>.txt")
	flag.StringVar(&o.listen, "listen", "", "serve the speech API on this address instead of speaking -text")
	flag.StringVar(&o.text, "text", "", "text to speak")
	flag.StringVar(&o.voiceName, "voice", "", "voice from -voices to speak with")
	flag.StringVar(&o.referenceWAV, "ref-wav", "", "reference clip to clone the voice of")
	flag.StringVar(&o.referenceText, "ref-text", "", "what the reference clip says")
	flag.StringVar(&o.output, "o", "out.wav", "output wav")
	flag.Uint64Var(&o.seed, "seed", 0, "sampling seed, random when 0")
	flag.IntVar(&o.maxRequests, "max-requests", 4, "requests served at once; more wait for a slot")
	flag.DurationVar(&o.queueWait, "queue-wait", 10*time.Second, "how long a request waits for a slot before a 503")
	flag.Parse()

	// The flags are checked before the model loads, which takes the longest.
	err := o.check()
	if err != nil {
		log.Fatal(err)
	}

	if o.maxRequests > engine.ParallelRequests {
		log.Printf("-max-requests %d is more than the %d requests whose parallel work shares every core; the rest run "+
			"some of theirs on one core", o.maxRequests, engine.ParallelRequests)
	}

	// TWOCENT_SETTINGS replaces release settings for an experiment that measures the model with other choices.
	e, err := engine.Load(o.modelPath,
		engine.Options{KeptRequests: o.maxRequests, Settings: []byte(os.Getenv("TWOCENT_SETTINGS"))})
	if err != nil {
		log.Fatal(err)
	}

	if o.listen != "" {
		err = serve(e, o)
	} else {
		err = speak(e, o)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func (o options) check() error {
	err := o.checkCombination()
	if err != nil {
		return err
	}

	if o.voiceName != "" {
		_, err = os.Stat(o.voicePath())
		if err != nil {
			return fmt.Errorf("voice %q is not in %s", o.voiceName, o.voicesDir)
		}
	}

	if o.referenceWAV != "" {
		_, err = os.Stat(o.referenceWAV)
		if err != nil {
			return fmt.Errorf("-ref-wav: %w", err)
		}
	}
	return nil
}

// checkCombination refuses flags that contradict each other, or that the mode they are given in ignores.
func (o options) checkCombination() error {
	speaksOnce := o.text != "" || o.voiceName != "" || o.referenceWAV != ""
	switch {
	case o.maxRequests < 1 || o.maxRequests > maxRequests:
		return fmt.Errorf("-max-requests must be 1 to %d", maxRequests)
	case o.listen == "" && o.text == "":
		return errors.New("-text or -listen is required")
	case o.listen != "" && speaksOnce:
		return errors.New("-listen serves requests, which name their own text and voice")
	case o.listen == "" && o.voiceName == "" && o.referenceWAV == "":
		return errors.New("-voice or -ref-wav is required")
	case o.referenceWAV != "" && o.voiceName != "":
		return errors.New("-ref-wav and -voice cannot be used together")
	case (o.referenceWAV == "") != (o.referenceText == ""):
		return errors.New("-ref-wav and -ref-text go together")
	}
	return nil
}

func (o options) voicePath() string {
	return filepath.Join(o.voicesDir, o.voiceName+".wav")
}

// serve answers the speech API until the server fails, or until a stop signal and the requests in flight end.
func serve(e *engine.Engine, o options) error {
	voices, err := loadVoices(e, o.voicesDir)
	if err != nil {
		return err
	}

	if len(voices) == 0 {
		log.Printf("no voices in %s: /v1/audio/speech refuses every request, and only voice cloning speaks",
			o.voicesDir)
	}

	log.Printf("listening on %s with voices %v", o.listen, slices.Sorted(maps.Keys(voices)))
	server := &http.Server{
		Addr:    o.listen,
		Handler: newServer(e, voices, newAdmission(o.maxRequests, o.queueWait)),
		// A request's body is read whole before it is served; its speech streams out with no time limit, each write
		// bounded on its own.
		ReadTimeout:       time.Minute,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	// A container stops the server with SIGTERM; the requests in flight finish their speech before it exits.
	stopped, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	served := make(chan error, 1)
	go func() { served <- server.ListenAndServe() }()
	select {
	case err := <-served:
		return err
	case <-stopped.Done():
	}

	log.Printf("stopping after the requests in flight, waiting at most %v", shutdownWait)
	ctx, cancel := context.WithTimeout(context.Background(), shutdownWait)
	defer cancel()
	return server.Shutdown(ctx)
}

// shutdownWait bounds how long a stopping server waits for its requests in flight; a long text streams for longer.
const shutdownWait = time.Minute

// speak writes -text in the voice the flags name to -o. The text is checked before a reference is encoded.
func speak(e *engine.Engine, o options) error {
	script, err := e.Prepare(o.text)
	if err != nil {
		return err
	}

	var voice *engine.Voice
	if o.referenceWAV != "" {
		voice, err = loadVoice(e, o.referenceWAV, o.referenceText)
	} else {
		voice, err = loadVoiceFile(e, o.voicePath())
	}
	if err != nil {
		return err
	}

	var samples []float32
	report, err := e.Speak(context.Background(), voice, script, newSource(o.seed), func(span []float32) error {
		samples = append(samples, span...)
		return nil
	})
	if err != nil {
		return err
	}

	// A capped chunk's speech up to the cap is still written.
	logCapped(o.output, report)

	err = os.WriteFile(o.output, audio.EncodeWAV(samples), 0o644)
	if err != nil {
		return err
	}

	log.Printf("wrote %s: %.2f s", o.output, float64(len(samples))/audio.SampleRate)
	return nil
}

// loadVoices reads each <name>.wav in dir with its <name>.txt, skipping a clip that cannot be a voice; a missing
// directory holds no voices.
func loadVoices(e *engine.Engine, dir string) (map[string]*engine.Voice, error) {
	voices := map[string]*engine.Voice{}
	paths, err := filepath.Glob(filepath.Join(dir, "*.wav"))
	if err != nil {
		return nil, err
	}

	for _, path := range paths {
		v, err := loadVoiceFile(e, path)
		if err != nil {
			// One bad clip should not keep the server from speaking in the others.
			log.Printf("skipping voice %s: %v", path, err)
			continue
		}
		voices[strings.TrimSuffix(filepath.Base(path), ".wav")] = v
	}
	return voices, nil
}

// loadVoiceFile reads a <name>.wav clip with the <name>.txt that says what it says.
func loadVoiceFile(e *engine.Engine, path string) (*engine.Voice, error) {
	referenceText, err := os.ReadFile(strings.TrimSuffix(path, ".wav") + ".txt")
	if err != nil {
		return nil, err
	}
	return loadVoice(e, path, string(referenceText))
}

// loadVoice reads the WAV clip at path, which says referenceText.
func loadVoice(e *engine.Engine, path, referenceText string) (*engine.Voice, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	reference, err := e.NewReference(data, referenceText)
	if err != nil {
		return nil, err
	}
	return e.NewVoice(context.Background(), reference)
}

// newSource is a random source of seed, or of a random seed when seed is 0.
func newSource(seed uint64) rand.Source {
	if seed == 0 {
		seed = rand.Uint64()
	}
	// PCG takes two words; the second is the seed mixed with the golden ratio's bits, so seeds near each other differ.
	return rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)
}

// logCapped logs how many chunks of the speech for name reached the length cap, if any.
func logCapped(name string, report engine.Report) {
	if report.Capped > 0 {
		log.Printf("%s: %d chunks reached the length cap before their speech ended", name, report.Capped)
	}
}
