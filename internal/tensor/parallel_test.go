package tensor

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// runnerFunc is a function as a Runner.
type runnerFunc func(lo, hi, worker int)

func (f runnerFunc) Run(lo, hi, worker int) {
	f(lo, hi, worker)
}

// A short spin sends the pool's goroutines to sleep between most calls, so every test also runs the wake path.
func TestMain(m *testing.M) {
	spinTime = time.Microsecond
	m.Run()
}

// parallelCovers runs parallel over n items with parts of about cost operations each and reports any item not run
// exactly once.
func parallelCovers(t *testing.T, n, cost int) {
	t.Helper()
	counts := make([]atomic.Int32, n)
	parallel(n, cost, runnerFunc(func(lo, hi, _ int) {
		for i := lo; i < hi; i++ {
			counts[i].Add(1)
		}
	}))
	for i := range counts {
		if got := counts[i].Load(); got != 1 {
			t.Errorf("n=%d: item %d ran %d times", n, i, got)
		}
	}
}

// TestParallelEmpty runs a call of no items: it runs no part and does not divide by zero.
func TestParallelEmpty(t *testing.T) {
	parallelCovers(t, 0, 1<<20)
}

// TestParallelConcurrentCalls makes calls from more goroutines than the pool has slots: each call runs every item
// exactly once.
func TestParallelConcurrentCalls(t *testing.T) {
	var wg sync.WaitGroup
	for caller := range 8 {
		wg.Go(func() {
			for call := range 200 {
				parallelCovers(t, 1+(caller*31+call)%64, 1<<16)
			}
		})
	}
	wg.Wait()
}

// TestParallelNested calls from inside a part: the call takes a slot of its own, or runs on its caller when none is
// free.
func TestParallelNested(t *testing.T) {
	parallel(4, 1<<17, runnerFunc(func(lo, hi, _ int) {
		for range hi - lo {
			parallelCovers(t, 16, 1<<17)
		}
	}))
}

// TestParallelWorkersAreExclusive checks that no two parts of one call run on one worker at once and every worker
// number is below workers, which per-worker scratch relies on.
func TestParallelWorkersAreExclusive(t *testing.T) {
	for _, cost := range []int{1 << 12, 1 << 23} {
		busy := make([]atomic.Bool, workers)
		parallel(64, cost, runnerFunc(func(_, _, worker int) {
			if worker < 0 || worker >= workers {
				t.Errorf("worker %d outside [0, %d)", worker, workers)
				return
			}

			if !busy[worker].CompareAndSwap(false, true) {
				t.Errorf("two parts ran on worker %d at once", worker)
			}
			time.Sleep(time.Millisecond)
			busy[worker].Store(false)
		}))
	}
}

// TestParallelAllocatesNothing makes calls whose Runner is kept between calls: they allocate nothing.
func TestParallelAllocatesNothing(t *testing.T) {
	var total atomic.Int64
	runner := runnerFunc(func(lo, hi, _ int) { total.Add(int64(hi - lo)) })
	if allocs := testing.AllocsPerRun(100, func() { parallel(64, 1<<16, runner) }); allocs != 0 {
		t.Errorf("a call allocated %v times", allocs)
	}
}

// TestParallelWakesSleepers makes calls whose parts wait for each other, so a call finishes only once enough goroutines
// take a part. The pauses outlast spinTime, so the pool's goroutines are asleep when each call comes. A small call of
// two parts needs one goroutine that takes small calls; a large call of one part a worker needs every goroutine, the
// ones that take only large calls among them.
func TestParallelWakesSleepers(t *testing.T) {
	if workers == 1 {
		t.Skip("one thread runs the parts in turn")
	}

	for _, call := range []struct{ parts, cost int }{{workers, 1 << 23}, {2, 1 << 17}} {
		for range 20 {
			time.Sleep(20 * spinTime)
			var started sync.WaitGroup
			started.Add(call.parts)
			parallel(call.parts, call.cost, runnerFunc(func(int, int, int) {
				started.Done()
				started.Wait()
			}))
		}
	}
}

// TestParallelFreesASlotAfterAPanic panics in the caller's part of a call on the pool: once the panic is recovered,
// every slot is free again, so a server that recovers a handler's panic keeps its whole pool.
func TestParallelFreesASlotAfterAPanic(t *testing.T) {
	if workers == 1 {
		t.Skip("with one thread a call runs on its caller and takes no slot")
	}

	func() {
		defer func() { _ = recover() }()
		parallel(64, 1<<16, runnerFunc(func(_, _, worker int) {
			if worker == 0 {
				panic("a part failed")
			}
		}))
	}()

	for i := range pool.taken {
		if pool.taken[i].Load() {
			t.Errorf("slot %d is still taken after its call panicked", i)
		}
	}
}
