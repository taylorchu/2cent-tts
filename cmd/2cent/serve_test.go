package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestServeFinishesItsRequestsOnSIGTERM stops a server with SIGTERM while it streams speech: the request in flight
// gets its whole speech, serve returns without an error, and the server takes no new connection.
func TestServeFinishesItsRequestsOnSIGTERM(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a process cannot send itself SIGTERM on Windows")
	}

	e, g := releaseEngine(t)
	addr := freeAddress(t)
	served := make(chan error, 1)
	go func() {
		served <- serve(e, options{listen: addr, voicesDir: t.TempDir(), maxRequests: 1, queueWait: time.Minute})
	}()
	waitForHealth(t, "http://"+addr+"/health")

	clip := base64.StdEncoding.EncodeToString(audio.EncodeWAV(fixtures.Floats(t, g.Reference.WAV24)))
	body, err := json.Marshal(map[string]string{"input": "Hello there. How are you today? I am fine, thank you.",
		"audio": clip, "audio_text": g.Prompts.ReferenceText})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.Post("http://"+addr+"/v1/audio/voice-cloning", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	// The signal comes once speech streams, so the request is in flight.
	first := make([]byte, 2)
	if _, err := io.ReadFull(resp.Body, first); err != nil {
		t.Fatalf("status %d: %v", resp.StatusCode, err)
	}

	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}

	if err := self.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}

	rest, err := io.ReadAll(resp.Body)
	samples := (len(first) + len(rest)) / 2
	if err != nil || resp.StatusCode != http.StatusOK || samples%audio.FrameSamples != 0 {
		t.Errorf("status %d, %d samples, read error %v", resp.StatusCode, samples, err)
	}

	select {
	case err := <-served:
		if err != nil {
			t.Errorf("serve returned %v", err)
		}
	case <-time.After(shutdownWait):
		t.Fatal("serve did not return after its request finished")
	}

	if conn, err := net.DialTimeout("tcp", addr, time.Second); err == nil {
		_ = conn.Close()
		t.Error("the stopped server took a new connection")
	}
}

// freeAddress is a local address no listener holds right now.
func freeAddress(t *testing.T) string {
	t.Helper()
	free, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	addr := free.Addr().String()
	if err := free.Close(); err != nil {
		t.Fatal(err)
	}
	return addr
}

// waitForHealth returns once url answers 200, which serve's handler does only after the signal is registered.
func waitForHealth(t *testing.T, url string) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("%s did not answer", url)
}
