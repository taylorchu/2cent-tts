package tensor

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// writeSafetensors writes a safetensors file of header and body, and returns its path.
func writeSafetensors(tb testing.TB, header string, body []byte) string {
	tb.Helper()
	path := filepath.Join(tb.TempDir(), "weights.safetensors")
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	data = append(append(data, header...), body...)
	err := os.WriteFile(path, data, 0o600)
	if err != nil {
		tb.Fatal(err)
	}
	return path
}

// TestOpenWeightsRefusesBadHeaders refuses entries whose shape and bytes disagree, however the sizes are chosen.
func TestOpenWeightsRefusesBadHeaders(t *testing.T) {
	for name, entry := range map[string]string{
		"negative sizes": `{"dtype": "F32", "shape": [-1, -2], "data_offsets": [0, 8]}`,
		"overflow":       `{"dtype": "BF16", "shape": [4294967296, 4294967296, 4], "data_offsets": [0, 8]}`,
		"zero size":      `{"dtype": "F32", "shape": [0, 2], "data_offsets": [0, 8]}`,
		"short":          `{"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}`,
		"past the end":   `{"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}`,
		"other dtype":    `{"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}`,
	} {
		w, err := OpenWeights(writeSafetensors(t, `{"x": `+entry+`}`, make([]byte, 8)))
		if err == nil {
			_ = w.Close()
			t.Errorf("%s: the header was accepted", name)
		}
	}
}

// TestLoaderReadsEachTensorInItsType reads f32 and bf16 tensors, refuses one asked for in another type or shape, and
// names the tensors nothing read.
func TestLoaderReadsEachTensorInItsType(t *testing.T) {
	body := binary.LittleEndian.AppendUint32(nil, 0x3f800000)
	body = binary.LittleEndian.AppendUint32(body, 0x40000000)
	for _, v := range []uint16{0x3f80, 0x4000, 0x4040, 0x4080} {
		body = binary.LittleEndian.AppendUint16(body, v)
	}
	w, err := OpenWeights(writeSafetensors(t, `{"f": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
		"b": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]},
		"unread": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}`, body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })

	l := NewLoader(w)
	if got := l.Float32s("f", 2); !slices.Equal(got, []float32{1, 2}) {
		t.Errorf("f32 tensor: %v", got)
	}

	if m := l.Matrices(2, MatrixPart{"b", 1}, MatrixPart{"b", 1}); l.Err() == nil || m.RowCount() != 0 {
		t.Error("a part whose rows do not match the tensor's was read")
	}

	l = NewLoader(w)
	m := l.Matrix("b", 2, 2)
	row := make([]float32, 2)
	m.Row(row, 1)
	if !slices.Equal(row, []float32{3, 4}) || l.Err() != nil {
		t.Errorf("bf16 matrix row: %v, %v", row, l.Err())
	}

	if v := l.Values("b", BF16); v.bf16 == nil || v.At(3) != 4 {
		t.Errorf("bf16 values: %+v", v)
	}

	if unread := l.Unread(); !slices.Equal(unread, []string{"f", "unread"}) {
		t.Errorf("unread tensors: %v", unread)
	}

	l.Float32s("b")
	if l.Err() == nil || !strings.Contains(l.Err().Error(), "BF16") {
		t.Errorf("a bf16 tensor read as f32: %v", l.Err())
	}

	l = NewLoader(w)
	if v := l.Values("b", F32); l.Err() == nil || v.f32 != nil {
		t.Error("bf16 values read as the f32 the caller requires")
	}
}
