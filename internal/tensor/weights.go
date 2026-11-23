package tensor

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
)

// DType is a tensor's stored type, as a safetensors header names it.
type DType string

// The stored types a release holds.
const (
	F32  DType = "F32"
	BF16 DType = "BF16"
)

// Weights is an open safetensors file: its header's tensors, by name, and the strings of its metadata. A Loader reads
// each tensor from the file into the memory that keeps it, so loading never holds a copy of the file.
type Weights struct {
	file     *os.File
	body     *io.SectionReader
	infos    map[string]tensorInfo
	Metadata map[string]string
}

// tensorInfo is a tensor's entry in a safetensors header.
type tensorInfo struct {
	DType   DType  `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// OpenWeights reads a safetensors file's header and keeps the file open for a Loader. Close it once loading is done.
func OpenWeights(path string) (*Weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	w, err := readHeader(f)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return w, nil
}

func readHeader(f *os.File) (*Weights, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	var size [8]byte
	_, err = io.ReadFull(f, size[:])
	n := binary.LittleEndian.Uint64(size[:])
	if err != nil || n > uint64(stat.Size()-8) {
		return nil, errors.New("not a safetensors file")
	}

	header := make([]byte, n)
	_, err = io.ReadFull(f, header)
	if err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}

	var entries map[string]json.RawMessage
	err = json.Unmarshal(header, &entries)
	if err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}

	w := &Weights{file: f, body: io.NewSectionReader(f, int64(8+n), stat.Size()-int64(8+n)),
		infos: map[string]tensorInfo{}}
	for name, msg := range entries {
		if name == "__metadata__" {
			err = json.Unmarshal(msg, &w.Metadata)
			if err != nil {
				return nil, fmt.Errorf("metadata: %w", err)
			}
			continue
		}

		var info tensorInfo
		err = json.Unmarshal(msg, &info)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}

		err = info.check(w.body.Size())
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		w.infos[name] = info
	}
	return w, nil
}

// check refuses an entry whose bytes lie outside a body of size bytes or do not hold its shape's values in its dtype.
func (info tensorInfo) check(size int64) error {
	if info.Offsets[0] < 0 || info.Offsets[0] > info.Offsets[1] || int64(info.Offsets[1]) > size {
		return errors.New("offsets out of range")
	}

	width := info.DType.width()
	if width == 0 {
		return fmt.Errorf("unsupported dtype %s", info.DType)
	}

	// Every size is checked against the bytes before it is multiplied, so a product cannot overflow.
	values := 1
	for _, d := range info.Shape {
		if d <= 0 || d > (info.Offsets[1]-info.Offsets[0])/width/values {
			return fmt.Errorf("shape %v does not fit %d bytes", info.Shape, info.Offsets[1]-info.Offsets[0])
		}
		values *= d
	}

	if bytes := info.Offsets[1] - info.Offsets[0]; bytes != width*values {
		return fmt.Errorf("%d bytes for shape %v in %s", bytes, info.Shape, info.DType)
	}
	return nil
}

// width is the bytes one value of the type takes, or zero for a type no reader reads.
func (d DType) width() int {
	switch d {
	case F32:
		return 4
	case BF16:
		return 2
	}
	return 0
}

// count is how many values the entry's bytes hold; check has matched it to the shape.
func (info tensorInfo) count() int {
	return (info.Offsets[1] - info.Offsets[0]) / info.DType.width()
}

// Close closes the file.
func (w *Weights) Close() error {
	return w.file.Close()
}

// Loader keeps the first missing, misshapen or mistyped tensor as its error, so model setup reads as a list of names.
type Loader struct {
	weights *Weights
	read    map[string]bool
	// raw holds a tensor's bytes on their way to its values, kept from tensor to tensor.
	raw []byte
	err error
}

// NewLoader reads tensors from w.
func NewLoader(w *Weights) *Loader {
	return &Loader{weights: w, read: map[string]bool{}}
}

// Err is the first missing, misshapen or mistyped tensor the loader was asked for, or a read that failed.
func (l *Loader) Err() error {
	return l.err
}

// Unread is the file's tensors nothing asked for, sorted.
func (l *Loader) Unread() []string {
	var unread []string
	for name := range l.weights.infos {
		if !l.read[name] {
			unread = append(unread, name)
		}
	}
	slices.Sort(unread)
	return unread
}

// Float32s reads an f32 tensor, checking its shape when one is given.
func (l *Loader) Float32s(name string, shape ...int) []float32 {
	return l.Values(name, F32, shape...).f32
}

// Values reads a tensor stored as dtype and keeps it in that type, checking its shape when one is given.
func (l *Loader) Values(name string, dtype DType, shape ...int) Values {
	info, ok := l.info(name, dtype, shape)
	if !ok {
		return Values{}
	}

	if info.DType == BF16 {
		return Values{bf16: read(l, name, info, make([]uint16, info.count()))}
	}
	return Values{f32: read(l, name, info, make([]float32, info.count()))}
}

// Matrix reads a bf16 tensor of rows by columns.
func (l *Loader) Matrix(name string, rows, cols int) Matrix {
	return l.Matrices(cols, MatrixPart{name, rows})
}

// MatrixPart is a bf16 tensor's name and how many rows it holds.
type MatrixPart struct {
	Name string
	Rows int
}

// Matrices reads bf16 tensors of the same columns as one matrix, the rows of each in turn, such as the projections
// one product computes together.
func (l *Loader) Matrices(cols int, parts ...MatrixPart) Matrix {
	// Every part is checked before any memory is taken, so sizes a bad config gives allocate nothing.
	infos := make([]tensorInfo, len(parts))
	rows := 0
	for i, part := range parts {
		info, ok := l.info(part.Name, BF16, []int{part.Rows, cols})
		if !ok {
			return Matrix{cols: cols}
		}
		infos[i], rows = info, rows+part.Rows
	}

	m := Matrix{cols: cols, bf16: make([]uint16, rows*cols)}
	at := 0
	for i, part := range parts {
		read(l, part.Name, infos[i], m.bf16[at:at+part.Rows*cols])
		at += part.Rows * cols
	}
	return m
}

// Dims is the shape of a tensor of the given rank, whatever its dtype. A missing tensor, or one of another rank, sets
// the loader's error and reads as zeros, so constructors can index the shape without a check of their own.
func (l *Loader) Dims(name string, rank int) []int {
	info, ok := l.info(name, "", nil)
	if ok && len(info.Shape) != rank {
		l.fail(fmt.Errorf("tensor %s has shape %v, want rank %d", name, info.Shape, rank))
		ok = false
	}

	if !ok {
		return make([]int, rank)
	}
	return info.Shape
}

// info is a tensor's entry, checked to hold dtype, any dtype when it is empty, and shape, when one is given.
func (l *Loader) info(name string, dtype DType, shape []int) (tensorInfo, bool) {
	info, ok := l.weights.infos[name]
	l.read[name] = true
	switch {
	case !ok:
		l.fail(fmt.Errorf("missing tensor %s", name))
	case dtype != "" && info.DType != dtype:
		l.fail(fmt.Errorf("tensor %s is %s, want %s", name, info.DType, dtype))
	case len(shape) > 0 && !slices.Equal(info.Shape, shape):
		l.fail(fmt.Errorf("tensor %s has shape %v, want %v", name, info.Shape, shape))
	default:
		return info, true
	}
	return tensorInfo{}, false
}

// read decodes a tensor's little-endian bytes into out, which holds exactly its values.
func read[T float32 | uint16](l *Loader, name string, info tensorInfo, out []T) []T {
	_, _ = binary.Decode(l.bytes(name, info), binary.LittleEndian, out)
	return out
}

// bytes is a tensor's bytes, in raw's memory until the next read. A failed read sets the loader's error and reads as
// zeros.
func (l *Loader) bytes(name string, info tensorInfo) []byte {
	n := info.Offsets[1] - info.Offsets[0]
	Sized(&l.raw, n)
	_, err := l.weights.body.ReadAt(l.raw, int64(info.Offsets[0]))
	if err != nil {
		l.fail(fmt.Errorf("tensor %s: %w", name, err))
		clear(l.raw)
	}
	return l.raw
}

func (l *Loader) fail(err error) {
	if l.err == nil {
		l.err = err
	}
}
