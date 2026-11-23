package main

import (
	"testing"
)

// TestCheckCombination accepts each mode's flags and refuses flags that contradict each other or that a mode ignores.
func TestCheckCombination(t *testing.T) {
	speak := options{text: "Hi.", voiceName: "default", maxRequests: 1}
	for name, tc := range map[string]struct {
		o  options
		ok bool
	}{
		"speak":              {speak, true},
		"serve":              {options{listen: ":8080", maxRequests: 4}, true},
		"clone":              {options{text: "Hi.", referenceWAV: "a.wav", referenceText: "Hi.", maxRequests: 1}, true},
		"nothing to do":      {options{maxRequests: 1}, false},
		"no voice":           {options{text: "Hi.", maxRequests: 1}, false},
		"serve and speak":    {options{listen: ":8080", text: "Hi.", maxRequests: 1}, false},
		"serve with a voice": {options{listen: ":8080", voiceName: "default", maxRequests: 1}, false},
		"voice and clip": {options{text: "Hi.", voiceName: "default", referenceWAV: "a.wav", referenceText: "Hi.",
			maxRequests: 1}, false},
		"clip without text": {options{text: "Hi.", referenceWAV: "a.wav", maxRequests: 1}, false},
		"text without clip": {options{text: "Hi.", voiceName: "default", referenceText: "Hi.", maxRequests: 1}, false},
		"no requests":       {options{text: "Hi.", voiceName: "default"}, false},
		"too many requests": {options{text: "Hi.", voiceName: "default", maxRequests: maxRequests + 1}, false},
	} {
		if err := tc.o.checkCombination(); (err == nil) != tc.ok {
			t.Errorf("%s: %v", name, err)
		}
	}
}
