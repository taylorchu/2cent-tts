package tensor

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// workers is read once, at start: the pool starts a goroutine for each worker, and SizedPerWorker sizes a buffer for
// each.
var workers = runtime.GOMAXPROCS(0)

// smallWorkers is how many workers, the caller among them, take a small call's parts. More spinning threads than the
// cap land on efficiency cores or hyperthreads, where the part one takes holds up its call, and a step's calls are too
// small to gain from them. Up to the cap every thread takes parts: holding one back for the work beside a step,
// another request's decoding or a reference's encoding, slowed a step more than it sped that work.
var smallWorkers = min(workers, 8)

// spinTime is how long a pool goroutine that takes small calls waits for the next one before it sleeps. The calls of a
// step come sooner, and waking a sleeping thread can take longer than a call's work.
var spinTime = 50 * time.Microsecond

// Runner is a call's work: Run runs items lo to hi on worker. A part runs on worker 0, its caller, or on one of the
// pool's goroutines. Two parts of one call never run on one worker at once, so a Runner can keep scratch memory for
// each worker, sized by SizedPerWorker. A call made from inside a part numbers its workers afresh, so it needs scratch
// of its own.
type Runner interface {
	Run(lo, hi, worker int)
}

// MaxParallelCalls is how many calls the pool runs at once. A call that finds every slot taken runs on its caller
// alone.
const MaxParallelCalls = 8

// job is one parallel call. Whoever is free claims its next part, so a slow core holds up only the part it took.
type job struct {
	runner Runner
	n      int
	parts  int
	// large marks a call that every worker may take, which waits behind the small ones; limit is how many workers
	// may take parts. Up to the small cap, both kinds have the same limit.
	large   bool
	limit   int
	next    atomic.Int64
	pending atomic.Int64
	// active counts the pool goroutines inside the job, so its slot is reused only once none is.
	active atomic.Int64
}

// pool runs calls in up to MaxParallelCalls slots. Its goroutines start with the first call and live as long as the
// program, asleep while no calls come. Each slot's job is reused call after call, so a call allocates nothing.
var pool struct {
	start sync.Once
	taken [MaxParallelCalls]atomic.Bool
	slots [MaxParallelCalls]atomic.Pointer[job]
	jobs  [MaxParallelCalls]job

	// calls counts the calls published, so a goroutine that slept knows whether one came meanwhile.
	calls atomic.Int64
	mu    sync.Mutex
	// A small call wakes only the goroutines that take small calls; a large one wakes every goroutine. The sleeper
	// counts are atomic so a call with no one asleep skips the lock.
	smallWake, largeWake         *sync.Cond
	smallSleeping, largeSleeping atomic.Int64
}

func init() {
	pool.smallWake, pool.largeWake = sync.NewCond(&pool.mu), sync.NewCond(&pool.mu)
}

// RunParallel sets task to value, runs it on parts of items 0 to n, the caller's goroutine taking parts too, and
// clears it, so the memory a task lives in holds no caller's buffers between calls. costPerItem is about how many
// operations an item takes: it sets how many parts the items split into and how many workers may take them. The task
// lives in its caller's memory, such as a field of the caller's state, so a call allocates nothing.
func RunParallel[T any, P interface {
	*T
	Runner
}](task P, value T, n, costPerItem int) {
	*task = value
	parallel(n, costPerItem, task)
	var zero T
	*task = zero
}

// parallel splits [0, n) into parts worth at least minWork operations each and runs them at once, the caller taking
// parts too.
func parallel(n, costPerItem int, runner Runner) {
	const minWork = 1 << 12
	// A call this large outlasts waking a thread, so every worker may take a part, and its parts outnumber the
	// workers so a slow core holds up little of it. A small call takes one part a worker: more parts cost more in
	// claiming them than they gave back in balance.
	const largeWork = 1 << 23
	large := n*costPerItem >= largeWork
	limit, perWorker := smallWorkers, 1
	if large {
		limit, perWorker = workers, 4
	}
	parts := min(n, max(1, n*costPerItem/minWork), perWorker*max(limit, 1))

	slot := -1
	if parts > 1 && limit > 1 {
		slot = takeSlot()
	}

	if slot < 0 {
		runner.Run(0, n, 0)
		return
	}

	pool.start.Do(func() {
		for worker := 1; worker < workers; worker++ {
			go work(worker)
		}
	})

	j := &pool.jobs[slot]
	j.runner, j.n, j.parts, j.large, j.limit = runner, n, parts, large, limit
	j.next.Store(0)
	j.pending.Store(int64(parts))
	pool.slots[slot].Store(j)
	pool.calls.Add(1)
	// A goroutine counts itself asleep before it checks calls, so a sleeper this misses saw the new call.
	wakeSmall, wakeLarge := pool.smallSleeping.Load() > 0, large && pool.largeSleeping.Load() > 0
	if wakeSmall || wakeLarge {
		pool.mu.Lock()
		if wakeSmall {
			pool.smallWake.Broadcast()
		}

		if wakeLarge {
			pool.largeWake.Broadcast()
		}
		pool.mu.Unlock()
	}

	// A part that panics on the caller still frees the slot, so a server that recovers the panic keeps its pool.
	defer releaseSlot(slot, j)
	j.run(0)
	for i := 1; j.pending.Load() > 0; i++ {
		if i%1024 == 0 {
			runtime.Gosched()
		}
	}
}

// releaseSlot unpublishes a call's job and frees its slot once no pool goroutine is inside the job.
func releaseSlot(slot int, j *job) {
	pool.slots[slot].Store(nil)
	for j.active.Load() > 0 {
		runtime.Gosched()
	}
	j.runner = nil
	pool.taken[slot].Store(false)
}

func takeSlot() int {
	for i := range pool.taken {
		if pool.taken[i].CompareAndSwap(false, true) {
			return i
		}
	}
	return -1
}

func (j *job) run(worker int) {
	for {
		part := int(j.next.Add(1) - 1)
		if part >= j.parts {
			return
		}

		j.runner.Run(j.n*part/j.parts, j.n*(part+1)/j.parts, worker)
		j.pending.Add(-1)
	}
}

// work runs parts of the published calls it may take, then waits for the next call. Small calls go first: a step's
// calls wait on each part, while a large call, the decoder's, has every other worker to finish it.
func work(worker int) {
	for {
		seen := pool.calls.Load()
		joinCalls(worker, true)
		joinCalls(worker, false)
		waitForCall(worker, seen)
	}
}

// joinCalls runs parts of the published small calls, or of the large ones, that worker may take.
func joinCalls(worker int, small bool) {
	for i := range pool.slots {
		j := pool.slots[i].Load()
		if j == nil {
			continue
		}

		j.active.Add(1)
		// The job's fields are read only once it is known to be still published, so a slot reused meanwhile is read
		// whole.
		if pool.slots[i].Load() == j && worker < j.limit && j.large != small {
			j.run(worker)
		}
		j.active.Add(-1)
	}
}

// waitForCall returns once a call is published after the count seen: a goroutine that takes small calls spins for
// spinTime first, and every goroutine then sleeps. A goroutine outside the pool waits for a free thread meanwhile;
// yielding in the spin let it in sooner but slowed each step more than that saved.
func waitForCall(worker int, seen int64) {
	small := worker < smallWorkers
	if small {
		start := time.Now()
		for i := 1; i%1024 != 0 || time.Since(start) < spinTime; i++ {
			if pool.calls.Load() != seen {
				return
			}
		}
	}

	sleeping, wake := &pool.largeSleeping, pool.largeWake
	if small {
		sleeping, wake = &pool.smallSleeping, pool.smallWake
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()
	sleeping.Add(1)
	for pool.calls.Load() == seen {
		wake.Wait()
	}
	sleeping.Add(-1)
}
