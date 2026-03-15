package aggregator

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"live-coding/model"
)

// mockStorage реализует интерфейс Storage для тестов.
type mockStorage struct {
	mu      sync.Mutex
	calls   []map[string]int64
	err     error
	onFlush func() // хук, вызываемый при каждом UpsertCounts
}

func (m *mockStorage) UpsertCounts(_ context.Context, counts map[string]int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// копируем map, чтобы snapshot не мутировался
	cp := make(map[string]int64, len(counts))
	for k, v := range counts {
		cp[k] = v
	}
	m.calls = append(m.calls, cp)

	if m.onFlush != nil {
		m.onFlush()
	}
	return m.err
}

func (m *mockStorage) getCalls() []map[string]int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func TestHandle_SingleEvent(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour) // большой интервал — flush не сработает сам

	agg.Handle(model.Event{UserID: "u1", Action: "click"})

	agg.mu.Lock()
	defer agg.mu.Unlock()

	if agg.counts["u1"] != 1 {
		t.Errorf("expected count 1 for u1, got %d", agg.counts["u1"])
	}
}

func TestHandle_MultipleEventsSameUser(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour)

	for i := 0; i < 5; i++ {
		agg.Handle(model.Event{UserID: "u1", Action: "click"})
	}

	agg.mu.Lock()
	defer agg.mu.Unlock()

	if agg.counts["u1"] != 5 {
		t.Errorf("expected count 5 for u1, got %d", agg.counts["u1"])
	}
}

func TestHandle_MultipleUsers(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour)

	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u2"})

	agg.mu.Lock()
	defer agg.mu.Unlock()

	if agg.counts["u1"] != 2 {
		t.Errorf("expected count 2 for u1, got %d", agg.counts["u1"])
	}
	if agg.counts["u2"] != 1 {
		t.Errorf("expected count 1 for u2, got %d", agg.counts["u2"])
	}
}

func TestFlush_SendsDataToStorage(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour)

	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u2"})

	agg.flush(context.Background())

	calls := store.getCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 flush call, got %d", len(calls))
	}
	if calls[0]["u1"] != 2 {
		t.Errorf("expected u1=2, got %d", calls[0]["u1"])
	}
	if calls[0]["u2"] != 1 {
		t.Errorf("expected u2=1, got %d", calls[0]["u2"])
	}

	// буфер должен быть пуст после flush
	agg.mu.Lock()
	defer agg.mu.Unlock()
	if len(agg.counts) != 0 {
		t.Errorf("expected empty counts after flush, got %v", agg.counts)
	}
}

func TestFlush_EmptyBuffer_NoCall(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour)

	agg.flush(context.Background())

	calls := store.getCalls()
	if len(calls) != 0 {
		t.Errorf("expected 0 flush calls for empty buffer, got %d", len(calls))
	}
}

func TestFlush_Error_ReturnsDataToBuffer(t *testing.T) {
	store := &mockStorage{err: fmt.Errorf("db unavailable")}
	agg := New(store, time.Hour)

	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u2"})

	agg.flush(context.Background())

	// данные должны вернуться в буфер
	agg.mu.Lock()
	defer agg.mu.Unlock()
	if agg.counts["u1"] != 2 {
		t.Errorf("expected u1=2 returned to buffer, got %d", agg.counts["u1"])
	}
	if agg.counts["u2"] != 1 {
		t.Errorf("expected u2=1 returned to buffer, got %d", agg.counts["u2"])
	}
}

func TestFlush_Error_MergesWithNewEvents(t *testing.T) {
	store := &mockStorage{err: fmt.Errorf("db unavailable")}
	agg := New(store, time.Hour)

	agg.Handle(model.Event{UserID: "u1"})
	agg.Handle(model.Event{UserID: "u1"})

	// flush вызовет onFlush, внутри которого мы добавим новое событие,
	// имитируя параллельное поступление данных во время flush
	store.onFlush = func() {
		agg.Handle(model.Event{UserID: "u1"})
	}

	agg.flush(context.Background())

	// u1: 2 (возвращённые) + 1 (новое во время flush) = 3
	agg.mu.Lock()
	defer agg.mu.Unlock()
	if agg.counts["u1"] != 3 {
		t.Errorf("expected u1=3 after merge, got %d", agg.counts["u1"])
	}
}

func TestRun_FlushesOnTick(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, 50*time.Millisecond)

	agg.Handle(model.Event{UserID: "u1"})

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	agg.Run(ctx)

	calls := store.getCalls()
	if len(calls) == 0 {
		t.Fatal("expected at least 1 flush call from ticker")
	}
}

func TestRun_FinalFlushOnCancel(t *testing.T) {
	flushed := make(chan struct{}, 1)
	store := &mockStorage{
		onFlush: func() {
			select {
			case flushed <- struct{}{}:
			default:
			}
		},
	}
	agg := New(store, time.Hour) // большой интервал — тикер не сработает

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() {
		agg.Run(ctx)
		close(done)
	}()

	// добавляем событие и отменяем контекст
	agg.Handle(model.Event{UserID: "u1"})
	cancel()

	<-done

	calls := store.getCalls()
	if len(calls) != 1 {
		t.Fatalf("expected exactly 1 final flush, got %d", len(calls))
	}
	if calls[0]["u1"] != 1 {
		t.Errorf("expected u1=1 in final flush, got %d", calls[0]["u1"])
	}
}

func TestHandle_Concurrent(t *testing.T) {
	store := &mockStorage{}
	agg := New(store, time.Hour)

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			agg.Handle(model.Event{UserID: "u1"})
		}()
	}
	wg.Wait()

	agg.mu.Lock()
	defer agg.mu.Unlock()

	if agg.counts["u1"] != 100 {
		t.Errorf("expected count 100 for u1, got %d", agg.counts["u1"])
	}
}
