package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/engine"
)

const (
	// maxRequestBytes is the largest body read. It holds a reference clip as long as a voice takes several times over,
	// as 16-bit 48 kHz mono WAV in base64. A clip of more channels or a higher rate must be shorter.
	maxRequestBytes = 32 << 20
	// writeTimeout is how long a write of speech may wait on the client: one that stops reading fails its request
	// instead of holding its slot.
	writeTimeout = 30 * time.Second
)

// speechRequest is OpenAI's speech request; voice cloning adds the reference clip and what it says.
type speechRequest struct {
	Input          string          `json:"input"`
	Voice          json.RawMessage `json:"voice"`
	ResponseFormat string          `json:"response_format"`
	Speed          float64         `json:"speed"`
	StreamFormat   string          `json:"stream_format"`
	Audio          string          `json:"audio"`
	AudioText      string          `json:"audio_text"`
}

// voiceName reads voice as OpenAI sends it: a name, or an object holding the name as its id.
func (r *speechRequest) voiceName() string {
	var name string
	if json.Unmarshal(r.Voice, &name) == nil {
		return name
	}

	var object struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(r.Voice, &object)
	return object.ID
}

// admission bounds the requests the engine works on at once and the request bodies held in memory. Requests share the
// engine's cores, so each one more slows every other. A request keeps only its reference's samples, not its body,
// while it waits for a slot.
type admission struct {
	slots, reads chan struct{}
	wait         time.Duration
	waiting      atomic.Int64
}

func newAdmission(requests int, wait time.Duration) *admission {
	return &admission{slots: make(chan struct{}, requests), reads: make(chan struct{}, 2*requests), wait: wait}
}

// admitRead takes a read slot, which the caller gives back with releaseRead. At most twice as many bodies as slots are
// read at once. A request that finds none free waits as long as it would for a slot to speak in, since a slow upload
// holds its read slot for as long as it sends.
func (a *admission) admitRead(ctx context.Context) error {
	if tryTake(a.reads) {
		return nil
	}
	return a.waitFor(ctx, a.reads)
}

func (a *admission) releaseRead() {
	<-a.reads
}

// admit takes a slot, which the caller gives back with release. A request past the bound waits for one. It is refused
// after wait, or at once when as many already wait as there are slots, so a burst fails fast instead of slowing every
// stream.
func (a *admission) admit(ctx context.Context) error {
	if tryTake(a.slots) {
		return nil
	}

	if a.waiting.Add(1) > int64(cap(a.slots)) {
		a.waiting.Add(-1)
		return a.busy()
	}
	defer a.waiting.Add(-1)
	return a.waitFor(ctx, a.slots)
}

// tryTake takes one of slots if one is free.
func tryTake(slots chan struct{}) bool {
	select {
	case slots <- struct{}{}:
		return true
	default:
		return false
	}
}

// waitFor takes one of slots once one is free, or refuses the request after waiting a.wait, or when ctx ends.
func (a *admission) waitFor(ctx context.Context, slots chan struct{}) error {
	// A select whose cases are both ready picks one at random, so a client already gone takes no slot.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	timer := time.NewTimer(a.wait)
	defer timer.Stop()
	select {
	case slots <- struct{}{}:
		return nil
	case <-timer.C:
		return a.busy()
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (a *admission) release() {
	<-a.slots
}

// busy refuses a request, telling it to retry after about as long as a request may wait.
func (a *admission) busy() error {
	return &apiError{status: http.StatusServiceUnavailable, message: "the server is busy; retry later",
		retryAfter: max(time.Second, a.wait)}
}

type apiError struct {
	status     int
	message    string
	retryAfter time.Duration
}

func (e *apiError) Error() string {
	return e.message
}

func badRequest(format string, args ...any) error {
	return &apiError{status: http.StatusBadRequest, message: fmt.Sprintf(format, args...)}
}

// serverErrorMessage answers every error that is the server's fault; the log keeps the detail.
const serverErrorMessage = "the server failed to speak the request"

// newServer serves POST /v1/audio/speech as OpenAI's speech API does, with voice naming a clip in -voices,
// POST /v1/audio/voice-cloning, which takes the voice's clip and text in the request, and GET /health, which answers
// once the model is loaded.
func newServer(e *engine.Engine, voices map[string]*engine.Voice, a *admission) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.Handle("POST /v1/audio/speech", &speechHandler{engine: e, admission: a, voices: voices})
	mux.Handle("POST /v1/audio/voice-cloning", &speechHandler{engine: e, admission: a, cloning: true})
	return mux
}

// speechHandler answers a speech request in a voice of voices, or, when cloning, in the voice of the reference clip
// the request holds.
type speechHandler struct {
	engine    *engine.Engine
	admission *admission
	voices    map[string]*engine.Voice
	cloning   bool
}

// voice is a request's voice: the named voice OpenAI's request gives, or for voice cloning the checked reference clip
// and text the request holds. A reference becomes a voice only once the request has a slot, since encoding it is work
// for the cores.
func (h *speechHandler) voice(req *speechRequest) (*engine.Voice, engine.Reference, error) {
	if h.cloning {
		// The clip is the voice; a named voice sent here would be dropped without a word.
		if len(req.Voice) > 0 && string(req.Voice) != "null" {
			err := badRequest("voice is taken only by /v1/audio/speech; here the clip is the voice")
			return nil, engine.Reference{}, err
		}

		if req.Audio == "" || req.AudioText == "" {
			return nil, engine.Reference{}, badRequest("audio and audio_text are required")
		}

		clip, err := base64.StdEncoding.DecodeString(req.Audio)
		if err != nil {
			return nil, engine.Reference{}, badRequest("audio is not base64: %v", err)
		}

		reference, err := h.engine.NewReference(clip, req.AudioText)
		return nil, reference, err
	}

	// A clip sent here would be dropped without a word, and the request spoken in a named voice.
	if req.Audio != "" || req.AudioText != "" {
		return nil, engine.Reference{}, badRequest("audio and audio_text are taken only by /v1/audio/voice-cloning")
	}

	name := req.voiceName()
	if name == "" {
		return nil, engine.Reference{}, badRequest("voice is required")
	}

	v := h.voices[name]
	if v == nil {
		return nil, engine.Reference{}, badRequest("unsupported voice %q", name)
	}
	return v, engine.Reference{}, nil
}

// parsedRequest is what a request asks for, every part checked: a named voice or a reference to clone one from.
type parsedRequest struct {
	wav       bool
	named     *engine.Voice
	reference engine.Reference
	script    engine.Script
}

func (h *speechHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	out := &speechWriter{w: w, control: http.NewResponseController(w), timeout: writeTimeout}
	err := h.answer(r, out)
	if err == nil {
		return
	}

	// A client that went away hears nothing, and its leaving is no fault of the server's.
	if r.Context().Err() != nil {
		return
	}

	log.Printf("%s: %v", r.URL.Path, err)
	if !out.started {
		writeError(w, err)
	}
}

// answer reads and checks a request, then, once it has a slot, speaks it into out. A bad text, WAV header or reference
// text is refused before the request waits; a clip whose speech is missing or out of range is refused once it has a
// slot, since finding its speech is work for the cores.
func (h *speechHandler) answer(r *http.Request, out *speechWriter) error {
	req, err := h.read(out.w, r)
	if err != nil {
		return err
	}

	out.wav = req.wav

	err = h.admission.admit(r.Context())
	if err != nil {
		return err
	}
	defer h.admission.release()

	voice := req.named
	if voice == nil {
		voice, err = h.engine.NewVoice(r.Context(), req.reference)
		if err != nil {
			return err
		}
	}

	report, err := h.engine.Speak(r.Context(), voice, req.script, newSource(0), out.write)
	if err != nil {
		return err
	}

	// A capped chunk's speech up to the cap is still sent.
	logCapped(r.URL.Path, report)
	return nil
}

// read decodes a request's body on a read slot and checks its text, then its voice, as the command line does.
// Preparing the longest input takes tens of milliseconds, so it runs before the request waits for a slot.
func (h *speechHandler) read(w http.ResponseWriter, r *http.Request) (parsedRequest, error) {
	err := h.admission.admitRead(r.Context())
	if err != nil {
		return parsedRequest{}, err
	}
	defer h.admission.releaseRead()

	req, err := decodeBody(w, r)
	if err != nil {
		return parsedRequest{}, err
	}

	script, err := h.engine.Prepare(req.Input)
	if err != nil {
		return parsedRequest{}, err
	}

	named, reference, err := h.voice(&req)
	if err != nil {
		return parsedRequest{}, err
	}
	return parsedRequest{wav: req.ResponseFormat == "wav", named: named, reference: reference, script: script}, nil
}

// speechWriter sends speech as it is made, a span at a time: 16-bit PCM, behind a WAV header for a WAV response.
type speechWriter struct {
	w       http.ResponseWriter
	control *http.ResponseController
	timeout time.Duration
	wav     bool
	pcm     []byte
	started bool
}

func (s *speechWriter) write(samples []float32) error {
	s.pcm = s.pcm[:0]
	if !s.started {
		contentType := "audio/pcm"
		if s.wav {
			contentType = "audio/wav"
			s.pcm = audio.StreamingWAVHeader()
		}
		s.w.Header().Set("Content-Type", contentType)
		s.started = true
	}

	// A writer that cannot take a deadline, such as a test's recorder, never stalls.
	err := s.control.SetWriteDeadline(time.Now().Add(s.timeout))
	if err != nil && !errors.Is(err, http.ErrNotSupported) {
		return err
	}

	s.pcm = audio.AppendPCM16(s.pcm, samples)
	_, err = s.w.Write(s.pcm)
	if err != nil {
		return err
	}

	err = s.control.Flush()
	if errors.Is(err, http.ErrNotSupported) {
		return nil
	}
	return err
}

// decodeBody decodes and checks a request's body, one JSON object, with its response format defaulting to PCM.
func decodeBody(w http.ResponseWriter, r *http.Request) (speechRequest, error) {
	var req speechRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	err := decoder.Decode(&req)
	if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
		return req, &apiError{status: http.StatusRequestEntityTooLarge,
			message: fmt.Sprintf("the request is larger than %d bytes", maxRequestBytes)}
	}

	// The server's read timeout ran out while the client was still sending.
	if errors.Is(err, os.ErrDeadlineExceeded) {
		return req, &apiError{status: http.StatusRequestTimeout, message: "the request took too long to send"}
	}

	if err != nil {
		return req, badRequest("decode request: %v", err)
	}

	if decoder.More() {
		return req, badRequest("decode request: more than one JSON value")
	}

	// A blank input is a missing parameter of OpenAI's request, refused before any rule of the engine's.
	switch {
	case strings.TrimSpace(req.Input) == "":
		return req, badRequest("input is required")
	case utf8.RuneCountInString(req.Input) > engine.MaxTextCharacters:
		return req, badRequest("input is longer than %d characters", engine.MaxTextCharacters)
	case req.Speed != 0 && req.Speed != 1:
		return req, badRequest("unsupported speed %g; only 1 is supported", req.Speed)
	case req.StreamFormat != "" && req.StreamFormat != "audio":
		return req, badRequest("unsupported stream_format %q; only audio is supported", req.StreamFormat)
	}

	if req.ResponseFormat == "" {
		req.ResponseFormat = "pcm"
	}

	if req.ResponseFormat != "pcm" && req.ResponseFormat != "wav" {
		return req, badRequest("unsupported response_format %q; use pcm or wav", req.ResponseFormat)
	}
	return req, nil
}

// writeError answers with err in OpenAI's error body: the engine's refusals of a text or a voice as a bad request,
// and an error that is the server's fault with a fixed message.
func writeError(w http.ResponseWriter, err error) {
	status, kind, message := http.StatusInternalServerError, "server_error", serverErrorMessage
	apiErr, ok := errors.AsType[*apiError](err)
	switch {
	case ok:
		status, message = apiErr.status, apiErr.message
		if apiErr.retryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(int(apiErr.retryAfter.Seconds())))
		}
	case errors.Is(err, engine.ErrBadText), errors.Is(err, engine.ErrBadReference):
		status, message = http.StatusBadRequest, err.Error()
	}

	if status < http.StatusInternalServerError {
		kind = "invalid_request_error"
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	encodeErr := encoder.Encode(errorBody{Error: errorDetail{Message: message, Type: kind}})
	if encodeErr != nil {
		log.Printf("error body: %v", encodeErr)
	}
}

// errorBody is OpenAI's error body. Its param and code are always null, as the server names neither.
type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}
