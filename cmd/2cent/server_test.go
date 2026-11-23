package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/engine"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestServerRejectsBadRequests answers each request the API or the engine refuses with a 400 in OpenAI's error body,
// and a body over the limit with a 413.
func TestServerRejectsBadRequests(t *testing.T) {
	e, _ := releaseEngine(t)
	server := newServer(e, map[string]*engine.Voice{"default": e.NoVoice()}, newAdmission(1, 0))
	for name, body := range map[string]string{
		"bad json":       `{bad`,
		"empty input":    `{"input": " ", "voice": "default"}`,
		"long input":     `{"input": "` + strings.Repeat("a", engine.MaxTextCharacters+1) + `", "voice": "default"}`,
		"format":         `{"input": "Hi.", "voice": "default", "response_format": "mp3"}`,
		"speed":          `{"input": "Hi.", "voice": "default", "speed": 1.5}`,
		"sse":            `{"input": "Hi.", "voice": "default", "stream_format": "sse"}`,
		"unknown voice":  `{"input": "Hi.", "voice": "nobody"}`,
		"no voice":       `{"input": "Hi."}`,
		"nothing to say": `{"input": "___", "voice": "default"}`,
		"only markup":    `{"input": "<br><b></b>", "voice": {"id": "default"}}`,
		"other script":   `{"input": "Hi \u65e5\u672c.", "voice": "default"}`,
		"two values":     `{"input": "Hi.", "voice": "default"} {}`,
		"a clip":         `{"input": "Hi.", "voice": "default", "audio": "AAAA", "audio_text": "Hi."}`,
	} {
		w := post(server, "/v1/audio/speech", body)
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), `"invalid_request_error"`) {
			t.Errorf("%s: %d %.200q", name, w.Code, w.Body.String())
		}
	}

	for name, body := range map[string]string{
		"no clip":    `{"input": "Hi.", "audio_text": "Hi."}`,
		"no text":    `{"input": "Hi.", "audio": "AAAA"}`,
		"not base64": `{"input": "Hi.", "audio": "not base64!", "audio_text": "Hi."}`,
		"a voice":    `{"input": "Hi.", "voice": "default", "audio": "AAAA", "audio_text": "Hi."}`,
	} {
		w := post(server, "/v1/audio/voice-cloning", body)
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), `"invalid_request_error"`) {
			t.Errorf("voice cloning, %s: %d %.200q", name, w.Code, w.Body.String())
		}
	}

	w := post(server, "/v1/audio/voice-cloning",
		`{"input": "Hi.", "audio_text": "Hi.", "audio": "`+strings.Repeat("A", maxRequestBytes)+`"}`)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("a body over the limit: %d %.200s", w.Code, w.Body.String())
	}

	w = post(server, "/v1/audio/voice-cloning",
		`{"input": "Hi.", "audio": "bm90IGEgd2F2", "audio_text": "`+
			strings.Repeat("a", engine.MaxReferenceTextCharacters+1)+`"}`)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "characters long") {
		t.Errorf("voice cloning with a long audio_text: %d %s", w.Code, w.Body.String())
	}

	w = post(server, "/v1/audio/voice-cloning", `{"input": "Hi.", "audio": "bm90IGEgd2F2", "audio_text": "Hi."}`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("voice cloning with a bad clip: %d %s", w.Code, w.Body.String())
	}
}

// TestServerAnswersHealth answers the health check of a loaded model.
func TestServerAnswersHealth(t *testing.T) {
	e, _ := releaseEngine(t)
	w := httptest.NewRecorder()
	newServer(e, nil, newAdmission(1, 0)).ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/health", http.NoBody))
	if w.Code != http.StatusOK {
		t.Errorf("health: %d %s", w.Code, w.Body.String())
	}
}

// TestServerServesConcurrentRequests sends speech and voice-cloning requests at once, more than the server serves at
// once, so the engine speaks and clones a voice for several requests side by side: each gets whole speech back. Run
// under the race detector, it checks the engine's sharing between requests.
func TestServerServesConcurrentRequests(t *testing.T) {
	e, g := releaseEngine(t)
	server := newServer(e, map[string]*engine.Voice{"default": e.NoVoice()}, newAdmission(2, 10*time.Minute))
	clip := base64.StdEncoding.EncodeToString(audio.EncodeWAV(fixtures.Floats(t, g.Reference.WAV24)))
	cloning, err := json.Marshal(map[string]string{"input": "Nice to meet you.", "audio": clip,
		"audio_text": g.Prompts.ReferenceText})
	if err != nil {
		t.Fatal(err)
	}

	requests := []struct{ name, path, format, body string }{
		{"speech 1", "/v1/audio/speech", "pcm", `{"input": "Hello there.", "voice": "default"}`},
		{"speech 2", "/v1/audio/speech", "pcm", `{"input": "How are you today?", "voice": "default"}`},
		{"speech 3", "/v1/audio/speech", "wav",
			`{"input": "Good morning.", "voice": "default", "response_format": "wav"}`},
		{"cloning", "/v1/audio/voice-cloning", "pcm", string(cloning)},
	}
	var wg sync.WaitGroup
	for _, r := range requests {
		wg.Go(func() {
			w := post(server, r.path, r.body)
			samples := speechSamples(w, r.format)
			if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "audio/"+r.format || samples == 0 ||
				samples%audio.FrameSamples != 0 {
				t.Errorf("%s: status %d, type %q, %d samples", r.name, w.Code, w.Header().Get("Content-Type"),
					samples)
			}
		})
	}
	wg.Wait()
}

// TestServerRefusesWhenBusy fills every slot: a request that may not wait is refused with a 503 to retry, and one that
// cannot be served, its reference clip included, is refused with its own 400 at once, without waiting for a slot.
func TestServerRefusesWhenBusy(t *testing.T) {
	e, _ := releaseEngine(t)
	a := newAdmission(1, 0)
	a.slots <- struct{}{}
	server := newServer(e, map[string]*engine.Voice{"default": e.NoVoice()}, a)
	w := post(server, "/v1/audio/speech", `{"input": "Hi.", "voice": "default"}`)
	if w.Code != http.StatusServiceUnavailable || w.Header().Get("Retry-After") == "" ||
		!strings.Contains(w.Body.String(), `"server_error"`) {
		t.Errorf("busy: %d, Retry-After %q, %.200q", w.Code, w.Header().Get("Retry-After"), w.Body.String())
	}

	silence := base64.StdEncoding.EncodeToString(audio.EncodeWAV(make([]float32, 40*audio.SampleRate)))
	for name, request := range map[string][2]string{
		"unknown voice":  {"/v1/audio/speech", `{"input": "Hi.", "voice": "nobody"}`},
		"nothing to say": {"/v1/audio/speech", `{"input": "___", "voice": "default"}`},
		"long clip": {"/v1/audio/voice-cloning",
			`{"input": "Hi.", "audio_text": "Hi there.", "audio": "` + silence + `"}`},
		"clip text": {"/v1/audio/voice-cloning", `{"input": "Hi.", "audio_text": "<br>", "audio": "` + silence + `"}`},
	} {
		w := post(server, request[0], request[1])
		if w.Code != http.StatusBadRequest {
			t.Errorf("%s while busy: %d %.200q", name, w.Code, w.Body.String())
		}
	}
}

// TestServerStopsWithItsClient speaks to a client whose second write fails and to one that goes away after its first:
// the server writes nothing more, adds no error body to the audio it sent, and frees the slot for the next request.
func TestServerStopsWithItsClient(t *testing.T) {
	e, _ := releaseEngine(t)
	server := newServer(e, map[string]*engine.Voice{"default": e.NoVoice()}, newAdmission(1, 0))
	body := `{"input": "Hello there. How are you? I am fine. Thank you.", "voice": "default"}`
	for name, failAt := range map[string]int{"a failed write": 2, "a client gone": 1} {
		ctx, cancel := context.WithCancel(t.Context())
		client := &failingClient{header: http.Header{}, failAt: failAt, cancel: cancel}
		if name == "a failed write" {
			client.cancel = nil
		}

		server.ServeHTTP(client, httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/audio/speech",
			strings.NewReader(body)))
		cancel()
		if client.writes != failAt || strings.Contains(client.body.String(), `"error"`) {
			t.Errorf("%s: %d writes, want %d, body holds an error: %v", name, client.writes, failAt,
				strings.Contains(client.body.String(), `"error"`))
		}

		if w := post(server, "/v1/audio/speech", `{"input": "Hi.", "voice": "default"}`); w.Code != http.StatusOK {
			t.Errorf("after %s: the next request got %d", name, w.Code)
		}
	}
}

// failingClient takes writes until its failAt-th, which fails; with cancel set, that write succeeds and the client goes
// away instead.
type failingClient struct {
	header http.Header
	body   strings.Builder
	writes int
	failAt int
	cancel context.CancelFunc
}

func (c *failingClient) Header() http.Header { return c.header }

func (*failingClient) WriteHeader(int) {}

func (c *failingClient) Write(b []byte) (int, error) {
	c.writes++
	if c.writes == c.failAt && c.cancel == nil {
		return 0, errors.New("connection reset")
	}

	if c.writes == c.failAt {
		c.cancel()
	}
	return c.body.Write(b)
}

// TestServerAnswersASlowUpload answers a body the read timeout cut off with a 408.
func TestServerAnswersASlowUpload(t *testing.T) {
	e, _ := releaseEngine(t)
	w := httptest.NewRecorder()
	newServer(e, nil, newAdmission(1, 0)).ServeHTTP(w,
		httptest.NewRequest(http.MethodPost, "/v1/audio/speech", timedOutBody{}))
	if w.Code != http.StatusRequestTimeout {
		t.Errorf("a body cut off by the read timeout: %d %.200q", w.Code, w.Body.String())
	}
}

// timedOutBody is a body whose read hit the server's read deadline.
type timedOutBody struct{}

func (timedOutBody) Read([]byte) (int, error) { return 0, os.ErrDeadlineExceeded }

// TestAdmissionWaitsForASlot takes the only slot and gives it back later: a waiting request gets it, and only then.
func TestAdmissionWaitsForASlot(t *testing.T) {
	a := newAdmission(1, time.Minute)
	if err := a.admit(context.Background()); err != nil {
		t.Fatal(err)
	}

	var released atomic.Bool
	go func() {
		time.Sleep(20 * time.Millisecond)
		released.Store(true)
		a.release()
	}()
	if err := a.admit(context.Background()); err != nil || !released.Load() {
		t.Errorf("waiting for a freed slot: error %v, admitted before the release: %v", err, !released.Load())
	}
}

// TestAdmissionRefuses checks each way a request that would wait is refused: its wait runs out, it finds as many
// requests waiting as there are slots, or its client goes away.
func TestAdmissionRefuses(t *testing.T) {
	a := newAdmission(1, 10*time.Millisecond)
	if err := a.admit(context.Background()); err != nil {
		t.Fatal(err)
	}

	err := a.admit(context.Background())
	if apiErr, ok := errors.AsType[*apiError](err); !ok || apiErr.status != http.StatusServiceUnavailable ||
		apiErr.retryAfter < time.Second {
		t.Errorf("a wait that ran out: %v", err)
	}

	long := newAdmission(1, time.Minute)
	if err := long.admit(context.Background()); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	waited := make(chan error, 1)
	go func() { waited <- long.admit(ctx) }()
	for deadline := time.Now().Add(5 * time.Second); long.waiting.Load() == 0; time.Sleep(time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the second request never waited for a slot")
		}
	}

	err = long.admit(context.Background())
	if apiErr, ok := errors.AsType[*apiError](err); !ok || apiErr.status != http.StatusServiceUnavailable {
		t.Errorf("a full queue: %v", err)
	}

	cancel()
	if err := <-waited; !errors.Is(err, context.Canceled) {
		t.Errorf("a client that went away: %v", err)
	}
}

// TestServerBoundsTheBodiesItReads refuses a request whose wait for a reading turn runs out, so a burst of large bodies
// is never held in memory all at once.
func TestServerBoundsTheBodiesItReads(t *testing.T) {
	e, _ := releaseEngine(t)
	a := newAdmission(1, 10*time.Millisecond)
	for range cap(a.reads) {
		a.reads <- struct{}{}
	}

	w := httptest.NewRecorder()
	newServer(e, map[string]*engine.Voice{"default": e.NoVoice()}, a).ServeHTTP(w,
		httptest.NewRequest(http.MethodPost, "/v1/audio/speech",
			strings.NewReader(`{"input": "Hi.", "voice": "default"}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("with every reading turn taken: %d %.200q", w.Code, w.Body.String())
	}
}

// TestSpeechWriterGivesUpOnAStalledClient writes to a client that stopped reading: the write must fail at its deadline,
// so the request ends and frees its slot instead of holding it for as long as the connection stays open.
func TestSpeechWriterGivesUpOnAStalledClient(t *testing.T) {
	client := &stalledClient{header: http.Header{}}
	out := &speechWriter{w: client, control: http.NewResponseController(client), timeout: 50 * time.Millisecond}
	start := time.Now()
	if err := out.write(make([]float32, audio.FrameSamples)); err == nil || time.Since(start) > 5*time.Second {
		t.Errorf("a write to a stalled client returned %v after %v", err, time.Since(start))
	}
}

// stalledClient is a client that stopped reading: a write blocks until its deadline passes, as a full socket's does.
type stalledClient struct {
	header   http.Header
	deadline time.Time
}

func (c *stalledClient) Header() http.Header { return c.header }

func (*stalledClient) WriteHeader(int) {}

func (c *stalledClient) Write([]byte) (int, error) {
	time.Sleep(time.Until(c.deadline))
	return 0, os.ErrDeadlineExceeded
}

//nolint:unparam // http.ResponseController finds a writer's deadline by this signature.
func (c *stalledClient) SetWriteDeadline(deadline time.Time) error {
	c.deadline = deadline
	return nil
}

// speechSamples is how many 16-bit samples a speech response in format holds, its WAV header aside.
func speechSamples(w *httptest.ResponseRecorder, format string) int {
	header := 0
	if format == "wav" {
		header = len(audio.EncodeWAV(nil))
	}
	return (w.Body.Len() - header) / 2
}

var (
	loadOnce    sync.Once
	testEngine  *engine.Engine
	errTestLoad error
)

func releaseEngine(tb testing.TB) (*engine.Engine, *fixtures.Golden) {
	tb.Helper()
	path, g := fixtures.WeightsPath(tb), fixtures.LoadGolden(tb)
	loadOnce.Do(func() {
		// The concurrent tests serve two requests at once.
		testEngine, errTestLoad = engine.Load(path, engine.Options{KeptRequests: 2})
	})
	if errTestLoad != nil {
		tb.Fatal(errTestLoad)
	}
	return testEngine, g
}

// TestWriteErrorKeepsServerFaultsPrivate answers the engine's refusals of a text or a voice as bad requests in their
// own words, and any other error with the fixed message, whatever detail it holds.
func TestWriteErrorKeepsServerFaultsPrivate(t *testing.T) {
	for _, tc := range []struct {
		err    error
		status int
		says   string
	}{
		{fmt.Errorf("%w: nothing to say", engine.ErrBadText), http.StatusBadRequest, "nothing to say"},
		{fmt.Errorf("%w: no word", engine.ErrBadReference), http.StatusBadRequest, "no word"},
		{errors.New("weights: tensor x"), http.StatusInternalServerError, serverErrorMessage},
	} {
		w := httptest.NewRecorder()
		writeError(w, tc.err)
		body := w.Body.String()
		if w.Code != tc.status || !strings.Contains(body, tc.says) || strings.Contains(body, "tensor") {
			t.Errorf("%v: %d %s", tc.err, w.Code, w.Body.String())
		}
	}
}

// post sends body to the server at path and returns its answer.
func post(server http.Handler, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	server.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return w
}
